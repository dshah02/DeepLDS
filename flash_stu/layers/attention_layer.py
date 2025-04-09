import torch
import torch.nn as nn

from flash_stu.modules.attention import Attention
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
class AttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super(AttentionLayer, self).__init__()
        self.attn_norm = nn.RMSNorm(config.dim)
        self.attn = Attention(config=config)
        self.mlp_norm = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)

    def reset(self):
        """Reset the KV cache in the attention module."""
        self.attn.reset()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor=None, input_pos: torch.Tensor = None) -> torch.Tensor:
        
        x = x + self.attn(x=self.attn_norm(x), freqs_cis=freqs_cis, input_pos=input_pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x