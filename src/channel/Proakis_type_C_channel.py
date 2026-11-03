"""
Below is an illustrative (and somewhat simplified) Python/PyTorch function that implements the Proakis type-C channel model (5-tap real-valued impulse response) together with random delay, sampling time offset (STO) via a raised-cosine filter, and AWGN.  The overall flow follows your specified steps:

Explanation of the main steps:

1. Random delay (tau_off):
   • We draw an integer delay for each sample in the batch.
   • We create an output array of length (2n - n_M) filled with zeros and insert the original n complex symbols starting at the offset τ_off.
2. Random 5-tap “Proakis type C” channel:
   • Generate five real taps per batch element.
   • Each tap hᵢ = w_Proakis,ᵢ x tᵢ where tᵢ ~ N(0,1).
   • w_Proakis = [0.227, 0.46, 0.688, 0.46, 0.227].
   • We then convolve (full convolution) with this 5-tap real-valued impulse response.
3. Random sampling time offset (STO) with raised-cosine filter:
   • We draw a fractional delay τ_STO from a uniform distribution [0,1).
   • Construct a length-16 raised-cosine filter sampled at t + τ_STO.
   • Convolve again (full convolution).
4. Add AWGN:
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
      2) Multipath channel (5 real taps with Proakis weighting).
      3) Convolution with the real channel.
      4) Random sampling time offset (STO) via a raised-cosine filter (length 16).
      5) Addition of white Gaussian noise of variance sigma^2 (added in real & imag. parts).

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

    # Number of taps in the channel
    n_taps = 5
    n_M = n_taps
    # n_M = 0  # ! Debug

    # Proakis weighting
    w_proakis = torch.tensor(
        [0.227, 0.46, 0.688, 0.46, 0.227], dtype=torch.float, device=device
    )
    w_proakis = w_proakis / torch.sum(torch.abs(w_proakis))

    # Range for random delay: tau_off ~ U[0, n - n_M)
    max_delay = 12
    # max_delay = max(n - n_M, 1)
    tau_offs = torch.randint(
        low=0, high=max_delay + 1, size=(Batch_size,), device=device
    )
    # set tau_off to be all zeros
    # tau_offs = torch.zeros((Batch_size,), device=device, dtype=torch.int64)

    # true_starting_point_onehot = starting_point_onehot(tau_offs, dmax=max_delay + 1).to(
    #     x.dtype
    # )
    true_starting_point_onehot = starting_point_onehot(tau_offs, dmax=max_delay + 1).to(
        x.dtype
    )

    # --------------------------------------------------------------------------
    # Step 1: Vectorized zero-padding with the random delay
    # --------------------------------------------------------------------------
    out_len_step1 = n + max_delay
    x_delayed = torch.zeros(
        (Batch_size, out_len_step1, 2), dtype=x.dtype, device=device
    )
    batch_idx = torch.arange(Batch_size, device=device).unsqueeze(1).expand(-1, n)
    sample_idx = torch.arange(n, device=device).unsqueeze(0).expand(
        Batch_size, -1
    ) + tau_offs.unsqueeze(1)
    x_delayed[batch_idx, sample_idx, :] = x

    # Combine real and imaginary parts into a complex tensor
    x_delayed_complex = torch.complex(x_delayed[:, :, 0], x_delayed[:, :, 1])

    # --------------------------------------------------------------------------
    # Step 2: Generate random 5-tap real channel h_i with Proakis weighting
    # --------------------------------------------------------------------------
    h_real = torch.sqrt(w_proakis / 2) * torch.randn(
        (Batch_size, n_taps), dtype=torch.float, device=device
    )
    h_imag = torch.sqrt(w_proakis / 2) * torch.randn(
        (Batch_size, n_taps), dtype=torch.float, device=device
    )

    h_complex = torch.complex(h_real, h_imag)

    # # --------------------------------------------------------------------------
    # # Step 3: Vectorized full complex convolution with the channel h using group conv
    # # --------------------------------------------------------------------------
    # # Reshape inputs for convolution: (B, 1, L) -> (1, B, L)
    seq_r = x_delayed_complex.real.unsqueeze(1).view(1, Batch_size, -1)
    seq_i = x_delayed_complex.imag.unsqueeze(1).view(1, Batch_size, -1)
    # Filters are already shaped as (B, 1, n_taps)
    filt_r = h_complex.real.unsqueeze(1)
    filt_i = h_complex.imag.unsqueeze(1)
    padding_val = n_taps - 1

    conv_rr = F.conv1d(seq_r, filt_r, padding=padding_val, groups=Batch_size).view(
        Batch_size, -1
    )
    conv_ii = F.conv1d(seq_i, filt_i, padding=padding_val, groups=Batch_size).view(
        Batch_size, -1
    )
    conv_ri = F.conv1d(seq_r, filt_i, padding=padding_val, groups=Batch_size).view(
        Batch_size, -1
    )
    conv_ir = F.conv1d(seq_i, filt_r, padding=padding_val, groups=Batch_size).view(
        Batch_size, -1
    )
    y_conv = torch.complex(conv_rr - conv_ii, conv_ri + conv_ir)  # shape: (B, L_full)
    y_channel = torch.stack([y_conv.real, y_conv.imag], dim=2)

    # # --------------------------------------------------------------------------
    # # Step 4: Vectorized STO filter convolution using group conv
    # # --------------------------------------------------------------------------
    # filter_len = 16
    # # Sample tau_STO for each batch, uniformly from [-0.49, 0.49])
    # # tau_STOs = torch.rand(Batch_size, device=device) - 0.5
    # # change tau_STOs to be zeros for debugging
    # tau_STOs = torch.zeros((Batch_size,), device=device)

    # # Build raised-cosine filters for each batch element; shape: (B, filter_len)
    # g_filters = torch.stack(
    #     [
    #         piecewise_raised_cosine(filter_len, beta, tau, device=device)
    #         for tau in tau_STOs
    #     ],
    #     dim=0,
    # )
    # # Convert filters to proper shape: (Batch_size, 1, filter_len)
    # g_filters = g_filters.to(x.dtype).unsqueeze(1)

    # # Prepare input for convolution (for real and imag parts)
    # # Original shape for y_channel_real: (Batch_size, L,  ) after squeeze.
    # y_channel_real = y_channel[:, :, 0].unsqueeze(1)  # shape: (Batch_size, 1, L)
    # y_channel_imag = y_channel[:, :, 1].unsqueeze(1)  # shape: (Batch_size, 1, L)

    # # For grouped conv with groups==Batch_size, we need to reshape the batch axis into the channel dimension.
    # # Transpose so that input becomes of shape (1, Batch_size, L)
    # y_channel_real_grp = y_channel_real.transpose(0, 1)
    # y_channel_imag_grp = y_channel_imag.transpose(0, 1)

    # padding_val_sto = filter_len - 1  # equals 15 for filter_len=16

    # y_sto_real = F.conv1d(
    #     y_channel_real_grp, g_filters, padding=padding_val_sto, groups=Batch_size
    # )  # shape: (1, Batch_size, L_out)
    # y_sto_imag = F.conv1d(
    #     y_channel_imag_grp, g_filters, padding=padding_val_sto, groups=Batch_size
    # )  # shape: (1, Batch_size, L_out)

    # # Convert back to (Batch_size, L_out)
    # y_sto_real = y_sto_real.squeeze(0)
    # y_sto_imag = y_sto_imag.squeeze(0)

    # # Stack real and imaginary parts along the last dimension: (Batch_size, L_out, 2)
    # y_sto = torch.stack([y_sto_real, y_sto_imag], dim=2)
    # # Cut off y_sto to have the shape of (batch_size, 2n, 2)
    # y_sto_cutoff = y_sto[:, : 2 * n, :]

    # --------------------------------------------------------------------------
    # Step 5: Add white Gaussian noise of variance sigma^2
    # --------------------------------------------------------------------------
    sigma_low_squared = torch.tensor(10 ** (-SNR_db_low / 10) / 2.0, device=device)
    sigma_high_squared = torch.tensor(10 ** (-SNR_db_high / 10) / 2.0, device=device)
    sigma_low = torch.sqrt(sigma_low_squared)
    sigma_high = torch.sqrt(sigma_high_squared)
    this_sigma = (
        torch.rand(y_channel.shape, device=device) * (sigma_low - sigma_high)
        + sigma_high
    )
    noise = this_sigma * torch.randn(y_channel.shape, dtype=torch.float, device=device)
    output_signal = y_channel + noise
    # return x_delayed, true_starting_point_onehot
    return output_signal, true_starting_point_onehot


def piecewise_raised_cosine(num_samples, beta, tau_STO, device):
    """
    Construct a length-num_samples raised-cosine filter g(t) sampled at
    t = n - (num_samples-1)/2 + tau_STO, for n = 0..(num_samples-1),
    following the piecewise definition:

    g(t) =
        1,                                          t = 0
        π/4 · sinc( π/(2β) ),                       |t| = 1/(2β)
        ( cos(β t) / (π t) ) * sinc( π t ),         otherwise
    with T=1.
    """
    n = torch.arange(num_samples, dtype=torch.float, device=device)
    center = (num_samples - 1) / 2.0
    t = n - center + tau_STO

    def my_sinc(z):

        # Handle both tensor and scalar inputs
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float, device=device)

        eps = 1e-12
        return torch.where(z.abs() < eps, torch.ones_like(z), torch.sin(z) / (z))

    g = torch.zeros(num_samples, dtype=torch.float, device=device)
    eps = 1e-9
    cond_t0 = t.abs() < eps
    cond_t_half = (t.abs() - (1.0 / (2.0 * beta))).abs() < eps

    val_t0 = 1.0
    val_t_half = (torch.pi / 4.0) * my_sinc(torch.pi / (2.0 * beta))
    out_else = (torch.cos(torch.pi * beta * t) / (torch.pi * t + eps)) * my_sinc(
        torch.pi * t
    )

    g[cond_t0] = val_t0
    g[cond_t_half] = val_t_half
    g[~(cond_t0 | cond_t_half)] = out_else[~(cond_t0 | cond_t_half)]
    return g


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
