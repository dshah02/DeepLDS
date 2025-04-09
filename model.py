import torch
import torch.nn as nn

from transformers import PreTrainedModel

from flash_stu.modules.stu import STU
from flash_stu.modules.attention import Attention
from flash_stu.utils.numerics import nearest_power_of_two
from config import FlashSTUConfig
from flash_stu.layers.stu_layer import STULayer
from flash_stu.layers.attention_layer import AttentionLayer
from flash_stu.modules.cache import Cache
from flash_stu.modules.kv_cache import KVCache
from flash_stu.utils.attn_utils import precompute_freqs_cis

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as TritonNorm
    triton_norm = True
except ImportError as e:
    print(
        f"Unable to import Triton-based RMSNorm: {e}. Falling back to PyTorch implementation."
    )
    from torch.nn import RMSNorm

    triton_norm = False

class FlashSTU(PreTrainedModel):
    config_class = FlashSTUConfig

    def __init__(self, config, filters, future_fill = False) -> None:
        super(FlashSTU, self).__init__(config)
        self.num_layers = config.num_layers
        assert config.dim % config.num_heads == 0, f"dim ({self.dim}) must be divisible num_heads ({self.num_heads})"
        self.head_dim = config.dim // config.num_heads

        self.future_fill = future_fill

        # From pytorch/pytorch#123411, we set persistent=True for torch.compile and PP compatibility
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                head_dim=self.head_dim,
                max_seq_len=config.seq_len,
                theta=config.theta,
            ),
            persistent=True,
        )

        self.use_approx = config.use_approx
        self.use_hankel_L = config.use_hankel_L

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim, dtype=config.torch_dtype)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList()

        for layer_idx in range(config.num_layers):
            # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
            if layer_idx % 2 == 0:
                self.layers.append(STULayer(config, filters, self.future_fill))
            else:
                self.layers.append(AttentionLayer(config) if config.use_attn else STULayer(config, filters, self.future_fill))

        self.norm = (
            TritonNorm(config.dim)
            if triton_norm
            else RMSNorm(config.dim, dtype=config.torch_dtype)
        )

        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=config.bias)

        if config.weight_tying:
            self.tok_emb.weight = self.lm_head.weight

        self.std = config.dim ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM\n" % (self._get_num_params() / 1e6,))

    
    def setup_caches(self, batch_size: int, dtype: torch.dtype = None, *kargs, **kwargs) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        if dtype is None:
            dtype = self.config.torch_dtype
            
        for layer in self.layers:
            if hasattr(layer, "attn"):
                layer.attn.kv_cache = KVCache(
                    batch_size=batch_size,
                    max_seq_len=self.config.seq_len,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    dtype=dtype,
                ).to(self.device)
            if hasattr(layer, "stu"):
                layer.stu.cache = Cache(
                    batch_size=batch_size,
                    max_seq_len=self.config.seq_len,
                    dim=self.config.dim,
                    dtype=dtype,
                ).to(self.device)
            if hasattr(layer, "lds"):
                layer.lds.reset()
                layer.lds.cache = True


    def caches_are_enabled(self) -> bool:
        """Check if the key value caches are setup."""
        for layer in self.layers:
            if hasattr(layer, "attn") and layer.attn.kv_cache is not None:
                return True
            if hasattr(layer, "stu"):
                if hasattr(layer.stu, "cache"):
                    return True
                elif hasattr(layer.stu, "eff_plus"):
                    return True
        return False

    def reset_caches(self):
        """Reset the key value caches."""
        if not self.caches_are_enabled():
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )

        for layer in self.layers:
            if hasattr(layer, "attn"):
                layer.attn.kv_cache = None
            elif hasattr(layer, 'stu'):
                layer.stu.cache = None
            elif hasattr(layer, 'lds'):
                layer.lds.reset()
                
    def forward(self, x: torch.Tensor, input_pos: torch.Tensor = None, *kargs, **kwargs) -> torch.tensor:
        """
        Forward pass with optional caching support.
        
        Args:
            x (torch.Tensor): Input token ids [batch_size, seq_len]
            input_pos (torch.Tensor, optional): Positions in sequence for the tokens, used with KV cache
            cache (bool, optional): Whether to use KV caching
        """
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            if hasattr(layer, "attn"): # Pass RoPE freq_cis to attention layers
                x = layer(x, freqs_cis=self.freqs_cis, input_pos=input_pos, *kargs, **kwargs)
            else:
                x = layer(x, input_pos = input_pos, *kargs, **kwargs)
            

        y_hat = self.lm_head(self.norm(x))
        return y_hat
    
    def reset(self):
        """Reset all KV caches in the model."""
        for layer in self.layers:
            if hasattr(layer, "attn"):
                layer.reset()
                
    def prefill(self, x, input_pos: torch.Tensor = None):
        assert self.future_fill, "must be using future fill"
        print("WARNING using prefill")
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)

        for layer in self.layers:
            if hasattr(layer, "attn"): # Pass RoPE freq_cis to attention layers
                x = layer(x, freqs_cis=self.freqs_cis, input_pos=input_pos)
            else:
                x = layer.prefill(x, input_pos = input_pos)
        
        y_hat = self.lm_head(self.norm(x))
        return y_hat


    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if hasattr(self, "pos_emb") and self.pos_emb is not None:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, Attention):
            torch.nn.init.xavier_normal_(module.c_attn.weight)
            torch.nn.init.xavier_normal_(module.c_proj.weight)
            if module.c_attn.bias is not None:
                torch.nn.init.zeros_(module.c_attn.bias)
            if module.c_proj.bias is not None:
                torch.nn.init.zeros_(module.c_proj.bias)
        elif isinstance(module, STU):
            if self.use_approx:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                if not self.use_hankel_L:
                    torch.nn.init.xavier_normal_(module.M_phi_minus)
