"""
encoder.py — Frozen DINOv2 ViT-L/14 RGB encoder.

Loads the public DINOv2 ViT-L/14 model from torch.hub and runs it in
inference mode only.  Returns per-patch tokens (B, N, 1024) and the
CLS token (B, 1024) for downstream cross-attention and pose prediction.

Input normalisation (ImageNet mean/std) is applied internally, so the
caller should pass RGB in [0, 1] float32.

Input resolution must be divisible by 14 in both dimensions.
Recommended: 392 × 518  (28 × 37 = 1 036 patch tokens per view).
"""

import torch
import torch.nn as nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class DinoEncoder(nn.Module):
    """
    Frozen DINOv2 ViT-L/14 RGB encoder.

    Parameters
    ----------
    model_name : str
        torch.hub model identifier.  Default "dinov2_vitl14".
        Also accepts "dinov2_vits14" or "dinov2_vitb14" if a smaller
        model is preferred.
    freeze : bool
        Freeze all DINOv2 parameters (default True).  Set False only
        for fine-tuning experiments.

    Outputs
    -------
    patch_tokens : (B, N, embed_dim)   N = (H/14) * (W/14)
    cls_token    : (B, embed_dim)
    """

    def __init__(self, model_name: str = "dinov2_vitl14", freeze: bool = True):
        super().__init__()

        self.dino = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )
        self.embed_dim: int = self.dino.embed_dim   # 1024 for ViT-L/14

        if freeze:
            for p in self.dino.parameters():
                p.requires_grad = False

        # Register normalisation constants as buffers so they move to the
        # correct device automatically when .to(device) is called.
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std",  std)

    def _normalise(self, rgb: torch.Tensor) -> torch.Tensor:
        """Map (B, 3, H, W) from [0, 1] to ImageNet statistics."""
        return (rgb - self._mean) / self._std

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        rgb : (B, 3, H, W)  float32 in [0, 1].
              H and W must both be multiples of 14.

        Returns
        -------
        patch_tokens : (B, N, 1024)   — patch-level features
        cls_token    : (B, 1024)      — global image feature
        """
        x   = self._normalise(rgb)
        out = self.dino.forward_features(x)
        return out["x_norm_patchtokens"], out["x_norm_clstoken"]
