import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training.losses import pixel_consistency_loss, se3_inv

B, H, W = 2, 60, 80
pred1 = torch.rand(B, 1, H, W) + 0.5
pred2 = torch.rand(B, 1, H, W) + 0.5
gt1   = torch.rand(B, 1, H, W) + 0.5
gt2   = torch.rand(B, 1, H, W) + 0.5
K = torch.tensor([[72., 0., 40.],[0., 72., 30.],[0.,0.,1.]]).unsqueeze(0).expand(B,-1,-1).clone()
T_12 = torch.eye(4).unsqueeze(0).expand(B,-1,-1).clone()
T_12[:, 0, 3] = 0.1  # small translation

l = pixel_consistency_loss(pred1, pred2, gt1, gt2, T_12, K)
print(f"pixel_consistency_loss = {l.item():.6f}")

pred1 = pred1.detach().requires_grad_(True)
l2 = pixel_consistency_loss(pred1, pred2, gt1, gt2, T_12, K)
l2.backward()
print(f"grad norm = {pred1.grad.norm().item():.6f}")

# Sanity: loss should be 0 when pred == gt
l_zero = pixel_consistency_loss(gt1, gt2, gt1, gt2, T_12, K)
print(f"loss when pred==gt = {l_zero.item():.8f}  (should be ~0)")
print("OK")
