"""
Compatibility shim cho torchvision.

Thử import torchvision thật trước.
Chỉ inject stub nếu import thật thất bại (C++ extension broken).
"""

import sys
import types
import torch
import torch.nn as nn


def _patch_relative_position_attention():
    """
    Fix: compute_relative_positions tạo index trên CPU nhưng relative_position_bias_table
    ở CUDA → device mismatch → CUDA device-side assert trong PyTorch 2.12.
    Patch để index luôn ở cùng device với table.
    """
    try:
        from sstan.models.transformers.modules.attention import (
            RelativePositionalEncodeMultiHeadSelfAttention,
        )

        def _fixed_compute_relative_positions(self, seq_len):
            device = self.relative_position_bias_table.device
            range_vec = torch.arange(seq_len, device=device)
            rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
            rel_pos_matrix = rel_pos_matrix + seq_len - 1
            return rel_pos_matrix

        RelativePositionalEncodeMultiHeadSelfAttention.compute_relative_positions = (
            _fixed_compute_relative_positions
        )
    except ImportError:
        pass  # sstan chưa có trong sys.path — sẽ được patch sau khi add vào path


class _PureStochasticDepth(nn.Module):
    """Pure-PyTorch StochasticDepth — fallback khi torchvision broken."""
    def __init__(self, p: float, mode: str = "row"):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        survival = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device)
        noise = noise.bernoulli_(survival).div_(survival)
        return x * noise


# ── Thử import torchvision thật ───────────────────────────────────────
try:
    import torchvision
    import torchvision.ops
    # Kiểm tra StochasticDepth có dùng được không
    _ = torchvision.ops.StochasticDepth(0.1, mode="row")
    print(f"[compat] torchvision {torchvision.__version__} OK")

except Exception as e:
    # torchvision thật bị lỗi → inject stub
    import importlib.machinery
    print(f"[compat] torchvision lỗi ({e}) — dùng stub")

    _tv_ops = types.ModuleType("torchvision.ops")
    _tv_ops.StochasticDepth = _PureStochasticDepth
    _tv_ops.__spec__ = importlib.machinery.ModuleSpec("torchvision.ops", None)

    _tv = types.ModuleType("torchvision")
    _tv.ops     = _tv_ops
    _tv.__version__ = "0.0.0+stub"
    _tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)

    sys.modules["torchvision"]     = _tv
    sys.modules["torchvision.ops"] = _tv_ops
