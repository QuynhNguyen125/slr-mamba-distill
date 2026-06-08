"""
Compatibility shim: fake torchvision stub cho torch 2.12.0+cu130.

torchvision C++ extension bị broken — inject stub vào sys.modules
TRƯỚC KHI bất kỳ code nào (kể cả sstan/teacher) có cơ hội import torchvision thật.

Import file này là dòng ĐẦU TIÊN trong mọi entry-point:
    import compat  # PHẢI đứng trước tất cả import khác
"""

import sys
import types
import torch
import torch.nn as nn


# ── Chỉ inject nếu torchvision chưa load thành công ──────────────────
_already_patched = "torchvision" in sys.modules

if not _already_patched:

    class StochasticDepth(nn.Module):
        """Pure-PyTorch StochasticDepth thay thế torchvision.ops.StochasticDepth."""
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

    import importlib.machinery

    # Tạo fake torchvision.ops
    _tv_ops = types.ModuleType("torchvision.ops")
    _tv_ops.StochasticDepth = StochasticDepth
    _tv_ops.__spec__ = importlib.machinery.ModuleSpec("torchvision.ops", None)

    # Tạo fake torchvision root
    _tv = types.ModuleType("torchvision")
    _tv.ops = _tv_ops
    _tv.__version__ = "0.0.0+stub"
    _tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)

    # Inject vào sys.modules — mọi import torchvision sau đây sẽ dùng stub này
    sys.modules["torchvision"]      = _tv
    sys.modules["torchvision.ops"]  = _tv_ops

    print("[compat] torchvision stub injected (torch 2.12 workaround)")

else:
    # torchvision đã load thật → chỉ patch StochasticDepth nếu bị broken
    try:
        import torchvision.ops as _ops
        _ops.StochasticDepth  # test xem có lỗi không
    except Exception:
        class StochasticDepth(nn.Module):
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
        import torchvision.ops as _ops
        _ops.StochasticDepth = StochasticDepth
        print("[compat] torchvision.ops.StochasticDepth patched")
