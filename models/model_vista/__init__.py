"""
model_vista — ViSTA-SLAM inspired depth alignment model.

Two-stream encoder (frozen DINOv2 ViT-L/14 + trainable ConvNeXt-Small depth
stream) with MASt3R-initialised cross-attention decoder and late fusion.
"""
from .network import DepthAlignNetV2
