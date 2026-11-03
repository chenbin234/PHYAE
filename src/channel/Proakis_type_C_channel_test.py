"""
Below is an illustrative (and somewhat simplified) Python/PyTorch function that implements the Proakis type-C channel model (5-tap real-valued impulse response) together with random delay, sampling time offset (STO) via a raised-cosine filter, and AWGN.  The overall flow follows your specified steps:

Explanation of the main steps:

1. Random delay (tau_off):
   • We draw an integer delay for each sample in the batch.
   • We create an output array of length (2n - n_M) filled with zeros and insert the original n complex symbols starting at the offset τ_off.
2. Add AWGN:
   • We create complex Gaussian noise for each output sample with variance σ² total (so real and imaginary parts each get variance σ²/2).
   • This is added to the final convolved result.

This code can be adapted or extended (e.g., different raised-cosine definitions or specific doping for the noise variance in real vs. imaginary parts) to suit your exact simulation needs.
"""

import torch
import torch.nn.functional as F


def proakis_c_channel(x, SNR_db_low, SNR_db_high, beta=0.22):
    """
    Simulate a Proakis type-C 5-tap real channel model with:
      1) Random delay (tau_off).
      2) Addition of white Gaussian noise of variance sigma^2 (added in real & imag. parts).

    Args:
        x (torch.Tensor):  Input of shape (batch_size, n, 2).
        SNR_db_low (float): Lower edge of SNR in dB.
        SNR_db_high (float): Higher edge of SNR in dB.
        beta (float)     : Roll-off factor for raised-cosine (0 < beta <= 1).

    Returns:
        y_out (torch.Tensor): The noisy output, after channel + STO filter + AWGN.
                              Shape will be (batch_size, something, 2).
        true_starting_point_onehot (torch.Tensor): One-hot encoded starting point.
    """
    device = x.device
    Batch_size, n, _ = x.shape

    # # convert x to complex tensor
    # x = x[:, :, 0] + 1j * x[:, :, 1]  # Shape: (batch_size, n), dtype:torch.complex64

    # # generate Rayleigh fading coefficients for each batch, each message share the same coefficients
    # h = torch.randn((Batch_size, 4), dtype=torch.complex64, device=device)
    # h_repeated = h.repeat_interleave(16, dim=1)
    # # h_expended = h.expand(-1, n)  # Shape: (batch_size, n), dtype:torch.complex64

    # # fading
    # x = x * h_repeated  # Shape: (batch_size, n), dtype:torch.complex64
    # # convert x to real number tensor, shape: (batch_size, n, 2), dtype:torch.float32
    # x = torch.stack(
    #     (x.real, x.imag), dim=-1
    # )  # Shape: (batch_size, n, 2), dtype:torch.float32

    # Number of taps in the channel
    n_taps = 5
    n_M = n_taps

    # Range for random delay: tau_off ~ U(0, n - n_M)
    max_delay = max(n - n_M, 1)
    # tau_offs = torch.randint(
    #     low=0, high=max_delay + 1, size=(Batch_size,), device=device
    # )
    # set tau_off to be all zeros
    tau_offs = torch.zeros((Batch_size,), device=device, dtype=torch.int64)

    true_starting_point_onehot = starting_point_onehot(tau_offs, dmax=max_delay + 1).to(
        x.dtype
    )

    # --------------------------------------------------------------------------
    # Step 1: Vectorized zero-padding with the random delay
    # --------------------------------------------------------------------------
    out_len_step1 = 2 * n - n_M
    # out_len_step1 = n

    x_delayed = torch.zeros(
        (Batch_size, out_len_step1, 2), dtype=x.dtype, device=device
    )
    batch_idx = torch.arange(Batch_size, device=device).unsqueeze(1).expand(-1, n)
    sample_idx = torch.arange(n, device=device).unsqueeze(0).expand(
        Batch_size, -1
    ) + tau_offs.unsqueeze(1)
    x_delayed[batch_idx, sample_idx, :] = x

    # --------------------------------------------------------------------------
    # Step 5: Add white Gaussian noise of variance sigma^2
    # --------------------------------------------------------------------------
    sigma_low_squared = torch.tensor(10 ** (-SNR_db_low / 10) / 2.0, device=device)
    sigma_high_squared = torch.tensor(10 ** (-SNR_db_high / 10) / 2.0, device=device)
    sigma_low = torch.sqrt(sigma_low_squared)
    sigma_high = torch.sqrt(sigma_high_squared)
    this_sigma = (
        torch.rand(x_delayed.shape, device=device) * (sigma_low - sigma_high)
        + sigma_high
    )
    noise = this_sigma * torch.randn(x_delayed.shape, dtype=torch.float, device=device)
    output_signal = x_delayed + noise

    return output_signal, true_starting_point_onehot


def starting_point_onehot(tau_offs, dmax):
    """
    Generate the one-hot encoded starting point for each batch.
    Args:
        tau_offs (torch.Tensor): The random delay for each batch.
        dmax (int): The number of classes.
    Returns:
        torch.Tensor: One-hot encoded starting points of shape (batch_size, dmax).
    """
    one_hot = F.one_hot(tau_offs, num_classes=dmax)
    return one_hot  # F.one_hot already returns shape (B, dmax)
