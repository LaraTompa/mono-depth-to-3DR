"""
model_depth_only — Depth-only two-view depth alignment model.

Entry point: DepthOnlyNet / build_depth_only_net
"""

from .network import DepthOnlyNet, build_depth_only_net

__all__ = ["DepthOnlyNet", "build_depth_only_net"]
