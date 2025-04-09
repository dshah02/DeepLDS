import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union

class FutureFillCache:
    """
    A naive future fill approach
    """
    def __init__(self, dim, dtype, K, batch_size=1, use_hankel_L = False, device=None):
        """
        Args:
            K: epoch length
            dim: dimensionality of your convolved outputs
        """
        self.futurefill_K = K
        self.d_out = dim
        self.dtype = dtype
        self.batch_size = batch_size
        self.device = device
        self.use_hankel_L = use_hankel_L

        self.reset()

    def reset(self):
        self.tau = 0 
        self.epoch_count = 0
        self.all_tokens = []

        self.C_plus = torch.zeros(self.futurefill_K,
                                  self.batch_size,
                                  self.d_out,
                                  dtype=self.dtype,
                                  device=self.device)
        if not self.use_hankel_L:
            self.C_minus = torch.zeros(self.futurefill_K,
                                    self.batch_size,
                                    self.d_out,
                                    dtype=self.dtype,
                                    device=self.device)

    def step(self, new_input: torch.Tensor, phi_proj: torch.Tensor, alt_phi_proj: Union[None, torch.Tensor], *args, **kwargs):
        """
        new_input: [batch_size, 1, d_out]
        phi_proj:  [max_seq_length, d_out]
        alt_phi_proj:  [max_seq_length, d_out]
        Returns:
            [batch_size, d_out]
        """

        num_token_to_generation = kwargs.get("num_token_to_generation")
        if num_token_to_generation is None:
            raise ValueError("Please pass 'num_token_to_generation' as a kwarg for now")
        _rest = num_token_to_generation % self.futurefill_K

        self.all_tokens.append(new_input)

        T = len(self.all_tokens)
        buffer_tensor = torch.cat(self.all_tokens, dim=0)

        # Local portion (the last tau+1 tokens):
        local_tokens = buffer_tensor[-(self.tau + 1):]    
        local_filter = phi_proj[: (self.tau + 1)] 
        flipped_filter = torch.flip(local_filter, dims=[0]) 
        partial_sum = (local_tokens * flipped_filter.unsqueeze(1)).sum(dim=0) 
        partial_sum += self.C_plus[self.tau]

        if alt_phi_proj is not None:
            local_filter_minus = alt_phi_proj[: (self.tau + 1)] 
            flipped_filter_minus = torch.flip(local_filter_minus, dims=[0]) 
            partial_sum_minus = (local_tokens * flipped_filter_minus.unsqueeze(1)).sum(dim=0) 
            partial_sum_minus += self.C_minus[self.tau]
        else:
            partial_sum_minus = None

        if (self.tau + 1) == self.futurefill_K:
            # ######### FFT #########
            # self.C_plus = self.compute_future_fill_fft(buffer_tensor, phi_proj)
            # if alt_phi_proj is not None:
            #     self.C_minus = self.compute_future_fill_fft(buffer_tensor, alt_phi_proj)
            
            # ######## NAIVE CONV #########
            self.C_plus, self.C_minus = self.compute_future_fill_vectorized(buffer_tensor, phi_proj, alt_phi_proj)

            self.tau = 0
            self.epoch_count += 1
        else:
            self.tau += 1

        # When future fill K does not truly divide num generation, we need one last call to compute what is coming next
        if _rest != 0 and T == num_token_to_generation - _rest:
            # ######### FFT #########
            # self.C_plus = self.compute_future_fill_fft(buffer_tensor, phi_proj)
            # if alt_phi_proj is not None:
            #     self.C_minus = self.compute_future_fill_fft(buffer_tensor, alt_phi_proj)

            ######## NAIVE CONV #########
            self.C_plus, self.C_minus = self.compute_future_fill_vectorized(buffer_tensor, phi_proj, alt_phi_proj)

            self.tau = 0
            self.epoch_count += 1

        return partial_sum, partial_sum_minus


    def compute_future_fill_vectorized(self, buffer_tensor, phi_proj, alt_phi_proj):
        T, B, d = buffer_tensor.shape
        L = T + self.futurefill_K
        phi_slice = phi_proj[:L]

        s_idx = torch.arange(T + self.futurefill_K, device=buffer_tensor.device).unsqueeze(1)
        j_idx = torch.arange(T, device=buffer_tensor.device).unsqueeze(0)
        i_idx = s_idx - j_idx

        valid = (i_idx >= 0) & (i_idx < L)
        i_idx_clamped = i_idx.clamp(0, L - 1)

        phi_slice_exp = phi_slice.unsqueeze(0).expand(T + self.futurefill_K, -1, d)
        phi_coeff = torch.gather(phi_slice_exp, 1, i_idx_clamped.unsqueeze(-1).expand(-1, -1, d))
        phi_coeff = phi_coeff * valid.unsqueeze(-1).float()

        full_conv = torch.einsum('std,tbd->sbd', phi_coeff, buffer_tensor)

        if alt_phi_proj is not None:
            alt_phi_slice = alt_phi_proj[:L]
            alt_phi_slice_exp = alt_phi_slice.unsqueeze(0).expand(T + self.futurefill_K, -1, d)
            alt_phi_coeff = torch.gather(alt_phi_slice_exp, 1, i_idx_clamped.unsqueeze(-1).expand(-1, -1, d))
            alt_phi_coeff = alt_phi_coeff * valid.unsqueeze(-1).float()
            full_conv_minus = torch.einsum('std,tbd->sbd', alt_phi_coeff, buffer_tensor)
        else:
            full_conv_minus = None

        return full_conv[-self.futurefill_K:], full_conv_minus[-self.futurefill_K:] if full_conv_minus is not None else None


    def compute_future_fill_naive_convolution(self, buffer_tensor, phi_proj, alt_phi_proj):

        T, B, d = buffer_tensor.shape
        phi_slice = phi_proj[: T + self.futurefill_K]  # [T + K, d]
        full_conv = buffer_tensor.new_zeros(T + self.futurefill_K, B, d)
        full_conv_minus = None

        if alt_phi_proj is not None:
            alt_phi_slice = alt_phi_proj[: T + self.futurefill_K]
            full_conv_minus = buffer_tensor.new_zeros(T + self.futurefill_K, B, d)
        
        for s in range(T + self.futurefill_K):
            for i in range(self.futurefill_K + T):
                if 0 <= s - i < T and i < phi_slice.shape[0]:
                    full_conv[s] += buffer_tensor[s - i] * phi_slice[i]
                    if alt_phi_proj is not None:
                        full_conv_minus[s] += buffer_tensor[s - i] * alt_phi_slice[i]
        
        return full_conv[-self.futurefill_K:], full_conv_minus[-self.futurefill_K:]

    def compute_future_fill_fft(self, buffer_tensor, phi_proj):
        """
        FFT based convolution.
        """

        T, B, d = buffer_tensor.shape
        K = self.futurefill_K

        phi_slice = phi_proj[:T+K]  #[T+K, d]
        n = 2 * T + K - 1

        x_padded = torch.nn.functional.pad(buffer_tensor, (0, 0, 0, 0, 0, n - T)) #pad buffer_tensor on time dimension (dim=0) to length n
        h_padded = torch.nn.functional.pad(phi_slice, (0, 0, 0, n - (T+K))) #pad phi_slice on first dimension (shape is (T+K, d)) to n

        fft_x = torch.fft.rfft(x_padded, n=n, dim=0)
        fft_h = torch.fft.rfft(h_padded, n=n, dim=0)  # [n_fft, d]

        fft_product = fft_x * fft_h.unsqueeze(1)  # [n_fft, B, d]

        conv_full = torch.fft.irfft(fft_product, n=n, dim=0)

        return conv_full[:T+K][-K:]