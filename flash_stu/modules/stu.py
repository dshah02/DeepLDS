import torch
import torch.nn as nn
from flash_stu.utils.numerics import nearest_power_of_two
from flash_stu.utils.stu_utils import convolve, flash_convolve
from flash_stu.utils.future_fill import EpochedFutureFill
try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False
class STU(nn.Module):
    def __init__(self, config, filters, future_fill = False) -> None:
        super(STU, self).__init__()
        self.config = config
        self.stu_filters = filters
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.K = config.num_eigh
        self.d_in = config.dim
        self.d_out = config.dim
        self.use_hankel_L = config.use_hankel_L
        self.use_approx = config.use_approx
        self.cache = None
        self.future_fill = future_fill        
        
        self.flash_fft = ( # TODO: Buggy with torch.compile, need to write a custom op wrapper
            FlashFFTConv(self.n, dtype=torch.bfloat16)
            if config.use_flash_fft and flash_fft_available
            else None
        )
        if self.use_approx:
            self.M_inputs = nn.Parameter(
                torch.randn(self.d_in, self.d_out, dtype=config.torch_dtype)
            )
            self.M_filters = nn.Parameter(
                torch.randn(self.K, self.d_in, dtype=config.torch_dtype)
            )

        else:
            self.M_phi_plus = nn.Parameter(
                torch.randn(self.K, self.d_in, self.d_out, dtype=config.torch_dtype)
            )
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    torch.randn(self.K, self.d_in, self.d_out, dtype=config.torch_dtype)
                )

        ################################ TO MAKE THINGS COMPARABLE WITH THE BASELINES THAT HAVE BEEN RUN, we compute on every forward ################################
        ################################ If time, re-run without this outside ########################################################################################
        self.phi_proj, self.alt_phi_proj = self.stu_filters @ self.M_filters.to(self.stu_filters.device), None #phi_proj: #(max seq length, dim)
        if not self.use_hankel_L:
            sign = torch.ones(self.phi_proj.size(0), device= self.phi_proj.device )
            sign[1::2] = -1 
            self.alt_phi_proj = self.phi_proj * sign.unsqueeze(-1)
        ########################################################################################################

    def setup_ff(self, K):
        # Before runtime calculation so we are good
        if self.future_fill:
            if not self.use_hankel_L:
                self.eff_minus = EpochedFutureFill(self.alt_phi_proj.T, epoch_length = K, bsz = self.d_in, device = torch.device("cuda"))
                
            #assuming batch size of 1
            self.eff_plus = EpochedFutureFill(self.phi_proj.T, epoch_length = K, bsz = self.d_in, device = torch.device("cuda"))

    def prefill(self, x: torch.Tensor, length = 1000):
        x_proj = x @ self.M_inputs #L, D
        x_proj = x_proj.squeeze(0)
        spectral_plus = self.eff_plus.prefill(x_proj.T, length) # B, L -> B, L
        spectral_minus = self.eff_minus.prefill(x_proj.T, length)
        spectral_plus= spectral_plus.T.unsqueeze(dim = 0,)
        spectral_minus= spectral_minus.T.unsqueeze(dim = 0)

        return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus

    def forward(
        self, x: torch.Tensor, input_pos = None, *args, **kwargs) -> torch.Tensor:
        if self.use_approx:
            # Contract inputs and filters over the K and d_in dimensions, then convolve
            x_proj = x @ self.M_inputs #B, L, D 

            if self.future_fill:
                #assuming x is 1, L, D so batch size is 1
                x_proj = x_proj.squeeze(0) #L, D
                spectral_plus = self.eff_plus(x_proj.T, self.phi_proj.T, *args, **kwargs) # B, L -> B, L
                if not self.use_hankel_L:
                    spectral_minus = self.eff_minus(x_proj.T, self.alt_phi_proj.T, *args, **kwargs)
                    spectral_minus= spectral_minus.T.unsqueeze(dim = 0)
                spectral_plus = spectral_plus.T.unsqueeze(dim = 0)

                return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus

            if self.cache is not None:

                # Update and remove the extra batch dimension
                x_proj = self.cache.update(x_proj.squeeze(dim=0), input_pos)
                
                # Extract the subset of x_proj up to the current position and flip it along the sequence dimension
                pos = input_pos.item()
                subset_seq = x_proj[:, :pos+1, :]
                flipped_seq = torch.flip(subset_seq, dims=[1])
                
                common_length = flipped_seq.size(1)
                
                flipped_seq_clipped = flipped_seq[:, :common_length, :]
                phi_proj_clipped = self.phi_proj[:common_length, :]
                if not self.use_hankel_L:
                    alt_phi_proj_clipped = self.alt_phi_proj[:common_length, :]
                
                spectral_plus = torch.sum(flipped_seq_clipped * phi_proj_clipped.unsqueeze(0), dim=1, keepdim=True)
                if not self.use_hankel_L:
                    spectral_minus = torch.sum(flipped_seq_clipped * alt_phi_proj_clipped.unsqueeze(0), dim=1, keepdim=True)
            
                return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus

            
            if self.flash_fft:
                
                spectral_plus, spectral_minus = flash_convolve(
                    x_proj, self.alt_phi_proj, self.flash_fft, self.use_approx
                )
            else:
                spectral_plus, spectral_minus = convolve(
                    x_proj, self.alt_phi_proj, self.n, self.use_approx
                )
        else:
            # Convolve inputs and filters,
            if self.flash_fft:
                U_plus, U_minus = flash_convolve(
                    x, self.stu_filters, self.flash_fft, self.use_approx
                )
            else:
                U_plus, U_minus = convolve(x, self.stu_filters, self.n, self.use_approx)
            # Then, contract over the K and d_in dimensions
            spectral_plus = torch.tensordot(
                U_plus, self.M_phi_plus, dims=([2, 3], [0, 1])
            )
            if not self.use_hankel_L:
                spectral_minus = torch.tensordot(
                    U_minus, self.M_phi_minus, dims=([2, 3], [0, 1])
                )
        # if input_pos.shape[0] == 1:
            # print((spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus)[:,-1,:])
        return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus