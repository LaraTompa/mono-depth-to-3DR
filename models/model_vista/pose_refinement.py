"""
pose_refinement.py — SE(3) iterative pose refinement module.

Replaces the single-shot RelativePoseHead with a ConvGRU-based loop that
refines pose by comparing geometrically-warped cross-attention features.

Architecture
------------
Coarse head
    cat(cam_embed1, cam_embed2) → MLP → 9-dim (6D-rot + 3-trans) → T_0

Refinement loop  (num_iters times)
    1. Warp feat2 → view1 using T_cur + depth_prior + K       [F.grid_sample]
    2. residual = (feat2_warped − feat1) × conf1              [confidence gate]
    3. inp = project(cat[residual, feat1, conf1, flow])        [1×1 conv]
    4. h   = ConvGRU(inp, h)                                  [spatial GRU]
    5. Δξ  = global_avg_pool(h) → MLP → 6-dim                [se(3) update]
    6. T_cur = exp(Δξ̂) @ T_cur                               [SE(3) left-mult]

T_cur is NOT detached between steps so that gradients from camera_pose_loss
(on the final T_N) and pose_iters_loss flow back through the full SE(3)
chain to the coarse head.

Approximate parameter count  (token_dim=768, feat_dim=128, hidden_dim=128)
    feat_proj   768×128×1×1            =   98 304
    inp_proj    259×128×1×1 + bias     =   33 280
    ConvGRU     3 gates × 384×128×3×3  =  884 736
    readout     128→64→6               =    8 262
    coarse_head 1536→256→9             =  395 785
    log_conf    1536→1                 =    1 537
    ─────────────────────────────────────────────
    Total                              ≈  1.42 M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_image_depth.geometry import rot6d_to_matrix, se3_inv


# ---------------------------------------------------------------------------
# SE(3) exponential map
# ---------------------------------------------------------------------------

def se3_exp_map(xi: torch.Tensor) -> torch.Tensor:
    """
    Lift a se(3) twist ξ = [ω, v] ∈ ℝ^(B,6) to SE(3).

    Uses the Rodrigues formula for the rotation part and the left Jacobian
    for the translation coupling.  Taylor series at θ → 0 prevents NaN/Inf.

    Parameters
    ----------
    xi : (B, 6)   [ω₁ ω₂ ω₃  v₁ v₂ v₃]  ω = axis×angle, v = velocity

    Returns
    -------
    T : (B, 4, 4)  SE(3) rigid body transform
    """
    B     = xi.shape[0]
    omega = xi[:, :3]
    v     = xi[:, 3:]

    theta_sq = (omega * omega).sum(dim=-1, keepdim=True).clamp(min=0.0)
    # add eps before sqrt — same reasoning as in geodesic_rotation_loss:
    # sqrt'(0) = inf which can leak through torch.where on some PyTorch builds
    theta    = theta_sq.add(1e-8).sqrt()

    # Skew-symmetric matrix  ω̂  (B, 3, 3)
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    Z = torch.zeros_like(wx)
    W = torch.stack([
        Z,  -wz,  wy,
        wz,   Z, -wx,
       -wy,  wx,   Z,
    ], dim=-1).reshape(B, 3, 3)
    W2 = torch.bmm(W, W)

    I3 = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(B, -1, -1)

    # Coefficients — Taylor series selected per-sample for small θ.
    # Both branches always computed; clamps prevent NaN in inactive branch.
    eps   = 1e-4
    small = theta < eps
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    a = torch.where(small,
                    1.0 - theta_sq / 6.0  + theta_sq ** 2 / 120.0,
                    sin_t / theta.clamp(min=eps))

    b = torch.where(small,
                    0.5 - theta_sq / 24.0 + theta_sq ** 2 / 720.0,
                    (1.0 - cos_t) / theta_sq.clamp(min=eps ** 2))

    c = torch.where(small,
                    1.0 / 6.0 - theta_sq / 120.0 + theta_sq ** 2 / 5040.0,
                    (theta - sin_t) / (theta_sq * theta).clamp(min=eps ** 3))

    a, b, c = a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1)

    R = I3 + a * W + b * W2                        # Rodrigues  (B,3,3)
    J = I3 + b * W + c * W2                        # left Jacobian
    t = torch.bmm(J, v.unsqueeze(-1)).squeeze(-1)  # (B, 3)

    T = torch.zeros(B, 4, 4, device=xi.device, dtype=xi.dtype)
    T[:, :3, :3] = R
    T[:, :3,  3] = t
    T[:,  3,  3] = 1.0
    return T


# ---------------------------------------------------------------------------
# ConvGRU cell
# ---------------------------------------------------------------------------

class ConvGRUCell(nn.Module):
    """Spatial GRU cell with 3×3 convolutional gates on (B, C, H, W) maps."""

    def __init__(self, inp_dim: int, hid_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.reset  = nn.Conv2d(inp_dim + hid_dim, hid_dim, kernel_size, padding=pad)
        self.update = nn.Conv2d(inp_dim + hid_dim, hid_dim, kernel_size, padding=pad)
        self.new    = nn.Conv2d(inp_dim + hid_dim, hid_dim, kernel_size, padding=pad)
        for layer in (self.reset, self.update, self.new):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h], dim=1)
        r  = torch.sigmoid(self.reset(xh))
        z  = torch.sigmoid(self.update(xh))
        n  = torch.tanh(self.new(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * n


# ---------------------------------------------------------------------------
# Pose refinement module
# ---------------------------------------------------------------------------

class PoseRefinementModule(nn.Module):
    """
    Iterative SE(3) pose refinement on the ViT patch grid (h14 × w14).

    Parameters
    ----------
    token_dim     : cross-attention spatial token dimension (768)
    feat_dim      : projected feature dim fed to the GRU (128)
    hidden_dim    : ConvGRU hidden state channels (128)
    num_iters     : number of refinement iterations (4)
    coarse_hidden : MLP width for the coarse initial-pose head (256)
    """

    def __init__(
        self,
        token_dim:     int = 768,
        feat_dim:      int = 128,
        hidden_dim:    int = 128,
        num_iters:     int = 4,
        coarse_hidden: int = 256,
    ):
        super().__init__()
        self.num_iters  = num_iters
        self.hidden_dim = hidden_dim

        # Feature projection: token_dim → feat_dim  (weight-shared both views)
        self.feat_proj = nn.Conv2d(token_dim, feat_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.feat_proj.weight)

        # GRU input projector: [residual | feat1 | conf | flow] → hidden_dim
        inp_dim = feat_dim * 2 + 1 + 2   # 259 for defaults
        self.inp_proj = nn.Conv2d(inp_dim, hidden_dim, 1)
        nn.init.xavier_uniform_(self.inp_proj.weight)
        nn.init.zeros_(self.inp_proj.bias)

        self.gru = ConvGRUCell(hidden_dim, hidden_dim)

        # Δξ readout: global-avg-pool → MLP → 6-dim se(3) update
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        nn.init.zeros_(self.readout[-1].weight)  # identity update at step 0
        nn.init.zeros_(self.readout[-1].bias)

        # Coarse head: cat(cam_embed1, cam_embed2) → T_0
        self.coarse_head = nn.Sequential(
            nn.Linear(token_dim * 2, coarse_hidden),
            nn.GELU(),
            nn.Linear(coarse_hidden, 9),  # 6D Gram-Schmidt rot + 3D trans
        )
        nn.init.zeros_(self.coarse_head[-1].weight)
        with torch.no_grad():
            self.coarse_head[-1].bias.copy_(
                torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 1e-2])
            )  # identity rotation; small but non-zero z-translation so the
               # direction gradient is defined from the very first step

        # Per-sample log-confidence for the predicted pose
        self.log_conf_head = nn.Linear(token_dim * 2, 1)
        nn.init.zeros_(self.log_conf_head.weight)
        nn.init.zeros_(self.log_conf_head.bias)

    # ------------------------------------------------------------------

    @staticmethod
    def _raw_to_T(raw: torch.Tensor) -> torch.Tensor:
        """(B, 9) → (B, 4, 4) via 6D Gram-Schmidt rotation."""
        B = raw.shape[0]
        R = rot6d_to_matrix(raw[:, :6])
        t = raw[:, 6:]
        T = torch.zeros(B, 4, 4, device=raw.device, dtype=raw.dtype)
        T[:, :3, :3] = R
        T[:, :3,  3] = t
        T[:,  3,  3] = 1.0
        return T

    @staticmethod
    def _build_warp_grid(
        depth:  torch.Tensor,   # (B, 1, h, w)
        K:      torch.Tensor,   # (B, 3, 3)  intrinsics at h×w resolution
        T_cur:  torch.Tensor,   # (B, 4, 4)  current cam1→cam2 estimate
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build F.grid_sample grid + 2D flow for warping view-2 features
        into view-1 space using the current pose estimate.

        Returns
        -------
        grid : (B, h, w, 2)  normalised [-1,1] coords (align_corners=True)
        flow : (B, 2, h, w)  pixel displacement (u_proj−u_src, v_proj−v_src)
        """
        B, _, h, w = depth.shape
        dev, dtype = depth.device, depth.dtype
        N = h * w

        ys = torch.arange(h, device=dev, dtype=dtype)
        xs = torch.arange(w, device=dev, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        xy1 = torch.stack([
            gx.flatten(), gy.flatten(),
            torch.ones(N, device=dev, dtype=dtype),
        ], dim=0).unsqueeze(0).expand(B, -1, -1)   # (B, 3, N)

        # Analytical K⁻¹
        fx = K[:, 0, 0].clamp(min=1e-8)
        fy = K[:, 1, 1].clamp(min=1e-8)
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        Ki = torch.zeros_like(K)
        Ki[:, 0, 0] =  1.0 / fx;  Ki[:, 0, 2] = -cx / fx
        Ki[:, 1, 1] =  1.0 / fy;  Ki[:, 1, 2] = -cy / fy
        Ki[:, 2, 2] =  1.0

        d  = depth.reshape(B, 1, N).clamp(min=0.01)
        P1 = torch.bmm(Ki, xy1) * d                           # (B, 3, N)
        R  = T_cur[:, :3, :3]
        t  = T_cur[:, :3,  3]
        P2 = torch.bmm(R, P1) + t.unsqueeze(-1)              # (B, 3, N)

        proj = torch.bmm(K, P2)                               # (B, 3, N)
        z    = proj[:, 2, :].clamp(min=1e-8)
        uj   = proj[:, 0, :] / z                              # (B, N)
        vj   = proj[:, 1, :] / z

        u_src = gx.flatten().unsqueeze(0).expand(B, -1)
        v_src = gy.flatten().unsqueeze(0).expand(B, -1)
        flow  = torch.stack([uj - u_src,
                              vj - v_src], dim=1).reshape(B, 2, h, w)

        un   = 2.0 * uj / max(w - 1, 1) - 1.0
        vn   = 2.0 * vj / max(h - 1, 1) - 1.0
        grid = torch.stack([un, vn], dim=-1).reshape(B, h, w, 2)

        return grid, flow

    # ------------------------------------------------------------------

    def forward(
        self,
        cam_embed1:  torch.Tensor,   # (B, token_dim)
        cam_embed2:  torch.Tensor,
        spatial1:    torch.Tensor,   # (B, N, token_dim)  N = h14×w14
        spatial2:    torch.Tensor,
        depth_mono1: torch.Tensor,   # (B, 1, H, W)  monocular prior, full res
        K_pred:      torch.Tensor,   # (B, 3, 3)  at full resolution H×W
        conf1:       torch.Tensor,   # (B, 1, Hp, Wp)  depth confidence
        H: int, W: int,
        h14: int, w14: int,
    ) -> tuple[list, list, torch.Tensor]:
        """
        Returns
        -------
        T_12_iters    : list[(B,4,4)] length num_iters — T_1 … T_N
        T_21_iters    : list[(B,4,4)] length num_iters — se3_inv of each
        log_conf_pose : (B,)
        """
        B   = cam_embed1.shape[0]
        dev, dtype = cam_embed1.device, cam_embed1.dtype

        # ── Coarse initial pose from global embeddings ─────────────────────
        embed_12      = torch.cat([cam_embed1, cam_embed2], dim=-1)
        T_cur         = self._raw_to_T(self.coarse_head(embed_12))    # (B,4,4)
        log_conf_pose = self.log_conf_head(embed_12).squeeze(-1)       # (B,)

        # ── Reshape tokens → 2-D feature maps at patch grid ───────────────
        # Detach spatial tokens so pose_iters gradients do NOT flow back into
        # the cross-attention decoder.  The decoder is already supervised by
        # depth loss and the final camera-pose loss; letting pose_iters also
        # modify it causes gradient interference that prevents both from converging.
        feat1 = self.feat_proj(
            spatial1.detach().reshape(B, h14, w14, -1).permute(0, 3, 1, 2)
        )   # (B, feat_dim, h14, w14)
        feat2 = self.feat_proj(
            spatial2.detach().reshape(B, h14, w14, -1).permute(0, 3, 1, 2)
        )

        # ── Downsample depth + confidence to patch grid ────────────────────
        d1 = F.interpolate(depth_mono1, (h14, w14),
                           mode='bilinear', align_corners=False)
        c1 = F.interpolate(conf1.detach(), (h14, w14),
                           mode='bilinear', align_corners=False)

        # ── Scale K to patch grid resolution ──────────────────────────────
        K_patch = K_pred.clone()
        K_patch[:, 0, :] = K_patch[:, 0, :] * (w14 / W)
        K_patch[:, 1, :] = K_patch[:, 1, :] * (h14 / H)

        # ── GRU hidden state ───────────────────────────────────────────────
        h_state = torch.zeros(B, self.hidden_dim, h14, w14,
                              device=dev, dtype=dtype)

        T_12_iters: list = []
        T_21_iters: list = []

        for _ in range(self.num_iters):
            # No detach on T_cur: gradients flow back through the full pose
            # chain to the coarse head.  With T_cur.detach() the coarse head
            # received gradient only from pose_iters[0] at weight ≈ 0.033,
            # which was too weak to move it away from identity — causing every
            # warp to be identity and leaving the GRU with nothing to refine.
            T_sg = T_cur

            grid, flow = self._build_warp_grid(d1, K_patch, T_sg)

            # Warp view-2 features into view-1 coordinate frame
            feat2_w = F.grid_sample(
                feat2, grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            )   # (B, feat_dim, h14, w14)

            # Confidence-weighted geometric residual
            residual = (feat2_w - feat1) * c1

            # ConvGRU step
            inp     = self.inp_proj(
                torch.cat([residual, feat1, c1, flow], dim=1)
            )
            h_state = self.gru(inp, h_state)

            # Predict Δξ ∈ se(3) from spatially-pooled hidden state
            delta_xi = self.readout(h_state.mean(dim=[-2, -1]))   # (B, 6)

            # Left-multiply SE(3) update: T_{k+1} = exp(Δξ̂) ⊕ T_k
            T_cur = se3_exp_map(delta_xi) @ T_sg

            T_12_iters.append(T_cur)
            T_21_iters.append(se3_inv(T_cur))

        return T_12_iters, T_21_iters, log_conf_pose


# ---------------------------------------------------------------------------
# Quick self-test  (python -m models.model_vista.pose_refinement)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

    B, D, h, w = 2, 768, 28, 37
    N  = h * w
    H, W = h * 14, w * 14
    fx = float(max(H, W)) * 0.9
    K  = torch.tensor([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]],
                      dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).contiguous()

    mod = PoseRefinementModule(token_dim=D, feat_dim=128, hidden_dim=128, num_iters=4)
    mod.eval()
    with torch.no_grad():
        T12, T21, lc = mod(
            torch.randn(B, D), torch.randn(B, D),
            torch.randn(B, N, D), torch.randn(B, N, D),
            torch.rand(B, 1, H, W) + 0.5,
            K,
            torch.rand(B, 1, H // 2, W // 2),
            H, W, h, w,
        )

    print(f"Iterations: {len(T12)},  T_12[-1]: {T12[-1].shape},  log_conf: {lc.shape}")
    for i, T in enumerate(T12):
        print(f"  iter {i+1}  det(R) = {T[:, :3, :3].det().tolist()}")

    total = sum(p.numel() for p in mod.parameters())
    print(f"\nParams: {total/1e6:.2f}M")

    # exp(0) = I
    err = (se3_exp_map(torch.zeros(2, 6)) - torch.eye(4)).abs().max()
    print(f"se3_exp_map(0) error from I: {err:.2e}  (expect < 1e-6)")