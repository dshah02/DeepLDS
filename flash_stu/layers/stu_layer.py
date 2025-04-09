import torch
import torch.nn as nn

from flash_stu.modules.stu import STU
from flash_stu.modules.swiglu import MLP

try:
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP as TritonMLP
    triton_mlp = True
except ImportError as e:
    print(
        f"Unable to import Triton-based MLP: {e}. Falling back to vanilla SwiGLU MLP instead."
    )
    triton_mlp = False

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as TritonNorm
    triton_norm = True
except ImportError as e:
    print(
        f"Unable to import Triton-based RMSNorm: {e}. Falling back to PyTorch implementation."
    )
    from torch.nn import RMSNorm
    triton_norm = False


class STULayer(nn.Module):
    def __init__(self, config, stu_filters, future_fill = False):
        super(STULayer, self).__init__()
        self.stu_norm = nn.RMSNorm(config.dim)
        self.stu = STU(config, stu_filters, future_fill)
        self.mlp_norm = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)


    def forward(self, x: torch.Tensor, input_pos, *kargs, **kwargs) -> torch.Tensor:
        x = x + self.stu(self.stu_norm(x), input_pos, *kargs, **kwargs)
        x = x + self.mlp(self.mlp_norm(x))
        return x