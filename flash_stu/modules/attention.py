import math

import torch
import torch.nn as nn

from flash_stu.utils.numerics import nearest_power_of_two
from flash_stu.utils.attn_utils import precompute_freqs_cis, apply_rotary_emb
try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(
        f"Unable to import Triton-based flash attention: {e}. No alternative currently available."
    )
    # TODO: Add FlexAttention + local attention mask when it's in stable release

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        assert config.dim % config.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.c_proj.SCALE_INIT = 1

        self.alibi_slopes = self._get_alibi_slopes(self.num_heads) if config.use_alibi else None
        self.window_size = config.window_size
        self.softcap = config.softcap

        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)
        
        # Initialize KV cache to None (will be set by setup_caches)
        self.kv_cache = None

    def _generate_slopes(self, n: int):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start**i) for i in range(n)]

    def _get_alibi_slopes(self, num_heads: int, interpolation_factor: float = 0.25):
        # If n_heads is a power of 2, generate slopes directly
        if math.log2(num_heads).is_integer():
            slopes = self._generate_slopes(num_heads)
        else:
            # Get slopes for the nearest power of two
            n = nearest_power_of_two(num_heads, round_up=False)
            slopes_power_of_two = self._generate_slopes(n)

            # Generate extra slopes
            extra_slopes = self._generate_slopes(2 * n)
            extra_slopes_trunc = extra_slopes[0::2][: num_heads - n]
            slopes = slopes_power_of_two + extra_slopes_trunc
        slopes = torch.tensor(slopes, device=torch.device("cuda")) # FA ALiBi must be on CUDA
        slopes = slopes * interpolation_factor  # https://arxiv.org/pdf/2310.13017
        return slopes

    def forward(
        self,
        x: torch.Tensor = None,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        freqs_cis: torch.Tensor = None,
        input_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        if x is not None:
            q = k = v = x
        if any(t is None for t in [q, k, v]):
            raise ValueError("Must provide either x for self-attention or q/k/v for cross-attention.")

        bsz, q_len, dim = q.shape

        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        # FlashAttention expects bsz, input_len, num_heads, head_dim
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_heads, self.head_dim)

        if self.alibi_slopes is None: # Either RoPE or ALiBi for positional embedding, need to make sure this is chill with input_pos
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        
        # Use KV cache if it's available
        if self.kv_cache is not None:
            # Update the KV cache with new keys and values at the specified positions
            # input_pos should be a tensor of shape [batch_size, seq_len] indicating positions
            k, v = self.kv_cache.update(input_pos, k, v)
  
            if len(input_pos) == 1:
                k = k[:, :input_pos+1]
                v = v[:, :input_pos+1]

            else:
                k = k[:, :max(input_pos)+1]
                v = v[:, :max(input_pos)+1]

        y = flash_attn_func(
            q=q, k=k, v=v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),
            alibi_slopes=self.alibi_slopes,
        )

        y = y.contiguous().view(bsz, q_len, -1)
        
        y = self.resid_dropout(self.c_proj(y))
        
        return y