import torch

from transformers import PretrainedConfig


class FlashSTUConfig(PretrainedConfig):

    model_type = "FlashSTU"

    def __init__(
        self,
        bsz: int = 8,
        dim: int = 896,
        num_heads: int = 8,
        num_layers: int = 12,
        seq_len: int = 8192,
        weight_tying: bool = True,
        window_size: int = 1024,
        vocab_size: int = 200064,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.1,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_flash_fft: bool = True,
        use_approx: bool = True,
        use_attn: bool = True,
        softcap: float = 50.0,
        theta: float = 10000.0,
        use_alibi: bool = False,
        dilation: int = 2,  # For dilated sliding window attention mask, if used
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        **kwargs,
    ):
        
        super().__init__(**kwargs)
        self.bsz = bsz
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.weight_tying = weight_tying
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.hidden_size = dim
        self.mlp_scale = mlp_scale
        self.intermediate_size = self.hidden_size * self.mlp_scale
        self.bias = bias
        self.dropout = dropout
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.use_flash_fft = use_flash_fft
        self.use_approx = use_approx
        self.use_attn = use_attn
        self.softcap = softcap
        self.theta = theta
        self.use_alibi = use_alibi
        self.torch_dtype = torch_dtype
        self.device = device
