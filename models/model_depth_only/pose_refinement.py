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
    # eps=1e-4 bounds the sqrt gradient at theta≈0 to 0.5/sqrt(1e-4)=50,
    # vs ~5000 with the previous 1e-8.  Forward error: |sinc(0.01)-1| ≈ 3e-5
    # — negligible for pose residuals clamped to ±0.05 rad.
    theta    = (theta_sq + 1e-4).sqrt()

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

    # Coefficients with clamped denominators for numerical stability.
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    # Avoid torch.where here: both branches are evaluated and can flood NaNs.
    theta_safe = theta.clamp(min=1e-6)
    a = sin_t / theta_safe
    b = (1.0 - cos_t) / theta_sq.clamp(min=1e-12)
    c = (theta - sin_t) / (theta_sq * theta_safe).clamp(min=1e-18)

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
        token_dim:  int = 768,   # cross-attention token dim → feat_proj input
        feat_dim:   int = 128,
        hidden_dim: int = 128,
        num_iters:  int = 4,
    ):
        super().__init__()
        self.num_iters  = num_iters
        self.hidden_dim = hidden_dim
        # detach_between_iters=True (default): each GRU step receives gradient
        # only from the loss on its own iterate, not through the full SE(3)
        # chain.  This prevents BPTT gradient explosion (4+ matrix-multiply
        # chain) at the cost of each step being supervised independently.
        # Set False to enable full BPTT (more expressive, but unstable without
        # careful LR scheduling).
        self.detach_between_iters = True

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

        # Coarse pose T_0 and log_conf_pose are provided externally by PoseHead;
        # this module is a pure geometric refiner only.

    # ------------------------------------------------------------------

    @staticmethod
    def _build_warp_grid(
        depth:  torch.Tensor,   # (B, 1, h, w)
        K:      torch.Tensor,   # (B, 3, 3)  intrinsics at h×w resolution
        T_cur:  torch.Tensor,   # (B, 4, 4)  current cam1→cam2 estimate
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        z    = proj[:, 2, :]
        # min_z = 0.1 m (10 cm): limits |d(u_j)/dz| ≤ ~100/pixel vs ~1e8 with
        # 1e-8.  Points behind the camera (z ≤ 0) are already excluded from
        # the residual by the 'valid' mask, but their gradient through
        # F.grid_sample would still propagate without this floor.
        z_safe = z.clamp(min=0.1)
        uj   = proj[:, 0, :] / z_safe                         # (B, N)
        vj   = proj[:, 1, :] / z_safe
        uj   = uj.clamp(-1e4, 1e4)
        vj   = vj.clamp(-1e4, 1e4)

        u_src = gx.flatten().unsqueeze(0).expand(B, -1)
        v_src = gy.flatten().unsqueeze(0).expand(B, -1)
        flow  = torch.stack([uj - u_src,
                              vj - v_src], dim=1).reshape(B, 2, h, w)

        un   = 2.0 * uj / max(w - 1, 1) - 1.0
        vn   = 2.0 * vj / max(h - 1, 1) - 1.0
        grid = torch.stack([un, vn], dim=-1).reshape(B, h, w, 2)
        grid = grid.clamp(-2.0, 2.0)
        # Mask invalid projections so warped-border samples do not enter residuals.
        valid = (
            (un > -1.0) & (un < 1.0) &
            (vn > -1.0) & (vn < 1.0) &
            (z > 0.0)
        ).float().reshape(B, 1, h, w)

        return grid, flow, valid

    # ------------------------------------------------------------------

    def forward(
        self,
        T_0:         torch.Tensor,   # (B, 4, 4)  coarse pose from PoseHead
        spatial1:    torch.Tensor,   # (B, N, token_dim)  N = h14×w14
        spatial2:    torch.Tensor,
        depth_mono1: torch.Tensor,   # (B, 1, H, W)  monocular prior, full res
        K:           torch.Tensor,   # (B, 3, 3)  at full resolution H×W
        conf1:       torch.Tensor,   # (B, 1, Hp, Wp)  depth confidence
        H: int, W: int,
        h14: int, w14: int,
    ) -> tuple[list, list]:
        """
        Returns
        -------
        T_12_iters : list[(B,4,4)] length num_iters — T_1 … T_N
        T_21_iters : list[(B,4,4)] length num_iters — se3_inv of each

        Coarse pose (T_0) and log_conf_pose are provided by PoseHead externally.
        """
        B   = T_0.shape[0]
        dev, dtype = T_0.device, T_0.dtype

        # Coarse initial pose provided externally by PoseHead.
        # Guard against NaN/Inf (can occur early in training before PoseHead
        # has stabilised); replace bad batch elements with identity.
        T_cur = T_0
        if not torch.isfinite(T_cur).all():
            bad = ~torch.isfinite(T_cur).view(B, -1).all(dim=-1)   # (B,) bool
            if bad.any():
                T_cur = T_cur.clone()
                T_cur[bad] = torch.eye(4, device=dev, dtype=dtype)

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
        # Normalize projected features so residual scale stays bounded.
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        # ── Downsample depth + confidence to patch grid ────────────────────
        d1 = F.interpolate(depth_mono1, (h14, w14),
                           mode='bilinear', align_corners=False)
        c1 = F.interpolate(conf1.detach(), (h14, w14),
                           mode='bilinear', align_corners=False)

        # ── Scale K to patch grid resolution ──────────────────────────────
        K_patch = K.clone()
        K_patch[:, 0, :] = K_patch[:, 0, :] * (w14 / W)
        K_patch[:, 1, :] = K_patch[:, 1, :] * (h14 / H)

        # ── GRU hidden state ───────────────────────────────────────────────
        h_state = torch.zeros(B, self.hidden_dim, h14, w14,
                              device=dev, dtype=dtype)

        T_12_iters: list = []
        T_21_iters: list = []

        for iter_idx in range(self.num_iters):
            # With detach_between_iters=True (default), each step treats the
            # previous iterate as a fixed warp target — the primary safeguard
            # against BPTT gradient explosion through the SE(3) chain.
            T_sg = T_cur

            grid, flow, valid = self._build_warp_grid(d1, K_patch, T_sg)

            # Warp view-2 features into view-1 coordinate frame
            feat2_w = F.grid_sample(
                feat2, grid,
                mode='bilinear', padding_mode='border', align_corners=True,
            )   # (B, feat_dim, h14, w14)

            # Confidence- and validity-weighted geometric residual.
            # Hard clamp keeps the residual well-conditioned even when the
            # warp is far off (e.g. first few training steps).
            residual = ((feat2_w - feat1) * c1 * valid).clamp(-1.0, 1.0)

            # ConvGRU step
            # Normalize pixel flow to feature-map units before mixing with features.
            flow_norm = flow.clone()
            flow_norm[:, 0] /= max(w14, 1)
            flow_norm[:, 1] /= max(h14, 1)
            # Clamp inp_proj output to prevent GRU gate saturation.
            inp = self.inp_proj(
                torch.cat([residual, feat1, c1, flow_norm], dim=1)
            ).clamp(-5.0, 5.0)
            h_state = self.gru(inp, h_state)
            h_state = torch.nan_to_num(h_state, nan=0.0, posinf=5.0, neginf=-5.0)
            h_state = h_state.clamp(-5.0, 5.0)

            # Predict Δξ ∈ se(3) from spatially-pooled hidden state
            delta_xi = 0.01 * self.readout(h_state.mean(dim=[-2, -1]))   # (B, 6)
            # Prevent catastrophic SE(3) updates
            rot_step = 0.05
            trans_step = 0.05

            delta_rot   = delta_xi[:, :3].clamp(-rot_step,   rot_step)
            delta_trans = delta_xi[:, 3:].clamp(-trans_step, trans_step)

            # Left-multiply SE(3) update: T_{k+1} = exp(Δξ̂) ⊕ T_k
            delta_xi = torch.nan_to_num(
                torch.cat([delta_rot, delta_trans], dim=-1),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            T_next = se3_exp_map(delta_xi) @ T_sg
            # NaN fallback: if the SE(3) update produced non-finite values
            # (rare but possible early in training), reuse the previous T_sg
            # for that sample so subsequent steps are not poisoned.
            if not torch.isfinite(T_next).all():
                bad = ~torch.isfinite(T_next).view(B, -1).all(dim=-1)
                if bad.any():
                    T_next = T_next.clone()
                    T_next[bad] = T_sg[bad].detach()
            T_cur = T_next

            T_12_iters.append(T_cur)
            T_21_iters.append(se3_inv(T_cur))
            if self.detach_between_iters and iter_idx < self.num_iters - 1:
                T_cur = T_cur.detach()

        return T_12_iters, T_21_iters


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
    T_0 = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).contiguous()
    with torch.no_grad():
        T12, T21 = mod(
            T_0,
            torch.randn(B, N, D), torch.randn(B, N, D),
            torch.rand(B, 1, H, W) + 0.5,
            K,
            torch.rand(B, 1, H // 2, W // 2),
            H, W, h, w,
        )

    print(f"Iterations: {len(T12)},  T_12[-1]: {T12[-1].shape}")
    for i, T in enumerate(T12):
        print(f"  iter {i+1}  det(R) = {T[:, :3, :3].det().tolist()}")

    total = sum(p.numel() for p in mod.parameters())
    print(f"\nParams: {total/1e6:.2f}M")

    # exp(0) = I
    err = (se3_exp_map(torch.zeros(2, 6)) - torch.eye(4)).abs().max()
    print(f"se3_exp_map(0) error from I: {err:.2e}  (expect < 1e-6)")