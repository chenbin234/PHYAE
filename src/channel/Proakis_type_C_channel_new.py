import torch
import torch.nn.functional as F
import math


def proakis_c_channel(x, output_length, channel, SNR_db_low, SNR_db_high):

    batch = x.size(0)
    device = x.device
    span = 16

    # x is of shape (batch, n, 2), convert x to complex tensor of shape (batch, n), where each element is a complex number
    x = x[:, :, 0] + 1j * x[:, :, 1]  # [batch, n]

    # 1. Channel impulse response
    if channel == "proakis_c_channel":
        h = proakisImpulseResponse(
            batch_size=batch, taps=5, device=device
        )  # [batch, h_len]
    elif channel == "awgn":
        # h = torch.ones(batch, 1, device=device)
        h = torch.ones((batch, 5), dtype=torch.cfloat, device=device)  # [batch, h_len]
    else:
        raise RuntimeError("Unknown channel type.")

    h_len = h.size(1)

    max_offset = max(output_length, x.size(1) + h_len - 1) - x.size(1) - h_len
    max_offset = max(0, max_offset)  # Ensure it's not negative

    # 2. Fractional (sub-symbol) offset
    t_off = torch.empty(batch, 1, device=device).uniform_(-0.49, 0.49)  # [batch,1]

    # 3. Padding before pulse shaping to center h in span
    padding = span - 1
    h_pad = F.pad(h, (padding, padding))

    # 4. Pulse shaping
    h_pulse = pulse_shaping(
        h_pad,
        beta=0.22,
        shape="normal",
        span=span,
        sps=1,
        offsets=t_off,
    )

    # 5. Integer timing offset (random)
    offset = torch.randint(low=0, high=max_offset + 1, size=(batch,), device=device)

    offset_one_hot = F.one_hot(
        offset, num_classes=max_offset + 1
    ).float()  # [batch, max_offset+1]
    offset_one_hot_complex = torch.complex(
        offset_one_hot, torch.zeros_like(offset_one_hot)
    )

    # pad2 = h_pulse.size(1) - 1
    # offset_one_hot_padded = F.pad(
    #     offset_one_hot_complex, (pad2, pad2)
    # )  # [batch, max_offset+1+2*pad2]

    # 6. Create batch-specific time-shifted channel impulse response
    h_offset = complex_convolution_with_batch_filters(offset_one_hot_complex, h_pulse)

    # 7. Pad signal for convolution
    # pad_signal_left = int(h_offset.size(1) // 2)
    # pad_signal_right = int((h_offset.size(1) - 1) // 2)
    # x_padded = F.pad(x, (pad_signal_left, pad_signal_right))

    # 8. Filter the signal by the time-shifted channel
    y = complex_convolution_with_batch_filters(x, h_offset)

    # truncate y to output_length, truncate the middle part
    y = y[:, :output_length]  # [batch, output_length]

    # convert y to real tensor
    y = torch.stack((y.real, y.imag), dim=-1)  # [batch, output_length, 2]

    # 9. AWGN
    y_noisy = awgn(y, SNR_db_low, SNR_db_high, device)

    # 10. Normalize output power to 1
    # rms = y_noisy.abs().pow(2).mean(dim=1, keepdim=True).sqrt()
    # y_normalized = y_noisy / rms

    return y_noisy, offset_one_hot


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


def proakisImpulseResponse(batch_size, taps=5, device=None):
    # Example: simple Proakis type-C impulse response
    w_proakis = torch.tensor(
        [0.227, 0.460, 0.688, 0.460, 0.227], dtype=torch.float32, device=device
    )

    t_i_real = torch.randn(
        (batch_size, taps), dtype=torch.float, device=device
    ) / torch.sqrt(torch.tensor(2.0))
    t_i_imag = torch.randn(
        (batch_size, taps), dtype=torch.float, device=device
    ) / torch.sqrt(torch.tensor(2.0))

    h_real = w_proakis * t_i_real
    h_imag = w_proakis * t_i_imag

    h_complex = torch.complex(h_real, h_imag)

    # h = h / torch.norm(h)
    # h = h.unsqueeze(0).repeat(batch_size, 1)  # [batch, taps]
    return h_complex


# def awgn(y, parameters, std=None):
#     # Adds Gaussian noise with std set according to SNR
#     # For demo: SNR in dB expected in parameters["snr_db"]
#     if std is None:
#         snr_db = parameters.get("snr_db", 20)
#         signal_power = y.abs().pow(2).mean()
#         noise_power = signal_power / (10 ** (snr_db / 10))
#         std = noise_power.sqrt()
#     noise = torch.randn_like(y) * std
#     if y.is_complex():
#         noise = noise + 1j * torch.randn_like(y) * std
#     return y + noise


def awgn(y, SNR_db_low, SNR_db_high, device):
    # Adds Gaussian noise with std set according to SNR
    sigma_low_squared = torch.tensor(10 ** (-SNR_db_low / 10) / 2.0, device=device)
    sigma_high_squared = torch.tensor(10 ** (-SNR_db_high / 10) / 2.0, device=device)
    sigma_low = torch.sqrt(sigma_low_squared)
    sigma_high = torch.sqrt(sigma_high_squared)
    this_sigma = (
        torch.rand(y.shape, device=device) * (sigma_low - sigma_high) + sigma_high
    )
    noise = this_sigma * torch.randn(y.shape, dtype=torch.float, device=device)
    output_signal = y + noise
    return output_signal


def pulse_shaping(h, beta, shape="normal", span=16, sps=1, offsets=None):
    """
    h: [batch, taps]
    offsets: [batch, 1] (fractional offsets in samples)
    Returns: [batch, taps+rc_len-1]
    """
    batch, taps = h.shape
    rc_bank = batch_raised_cosine(beta, span, sps, offsets.view(-1), h.device)
    # Convolve each h with its matching RC pulse
    outputs = []
    for i in range(batch):
        hi = h[i].unsqueeze(0).unsqueeze(0)  # [1,1,taps]
        rci = (
            rc_bank[i].flip(0).unsqueeze(0).unsqueeze(0)
        )  # [1,1,rc_len] (flip for convolution)
        # out_real = F.conv1d(hi.real, rci, padding=span // 2)  # [1,1,out_len]
        # out_imag = F.conv1d(hi.imag, rci, padding=span // 2)  # [1,1,out_len]
        out_real = F.conv1d(hi.real, rci, padding=0)  # [1,1,out_len]
        out_imag = F.conv1d(hi.imag, rci, padding=0)  # [1,1,out_len]
        out = torch.complex(out_real, out_imag)  # [1,1,out_len]
        outputs.append(out.squeeze(0).squeeze(0))
    return torch.stack(outputs, dim=0)  # [batch, out_len]


def batch_raised_cosine(beta, span, sps, offsets, device):
    # Create a bank of RC pulses, each offset by different t_off
    rc_len = span
    mid = rc_len // 2
    t = torch.arange(-mid, rc_len - mid, device=device).float()  # [rc_len]
    rc_pulses = []
    for offset in offsets:
        rc = raised_cosine(t - offset, beta, sps, span)
        rc = rc / rc.norm()  # Normalize energy
        rc_pulses.append(rc)
    rc_pulses = torch.stack(rc_pulses, dim=0)  # [batch, rc_len]
    return rc_pulses


def raised_cosine(t, beta, sps, span):
    # Generates raised cosine pulse at t, fractional sample positions included
    T = sps
    t = t / sps
    sinc = torch.sinc(t)
    cos_part = torch.cos(torch.pi * beta * t)
    denom = 1 - (2 * beta * t) ** 2
    rc = sinc * cos_part / (denom + 1e-10)
    rc[torch.abs(denom) < 1e-8] = (torch.pi / 4) * sinc[torch.abs(denom) < 1e-8]
    return rc


# def complex_convolution_with_batch_filters(x, h):
#     # Batch convolution for real/complex signals
#     # x: [batch, seq], h: [batch, taps]
#     # Returns: [batch, output_len]
#     out = []
#     for i in range(x.size(0)):
#         xi = x[i].unsqueeze(0).unsqueeze(0)  # [1,1,N]
#         hi = (
#             h[i].flip(0).unsqueeze(0).unsqueeze(0)
#         )  # [1,1,K], flip for correlation/convolution
#         yi = F.conv1d(xi, hi, padding=0)  # no padding; manually pad input as needed
#         out.append(yi.squeeze(0).squeeze(0))
#     # Pad output to max_length to form batch
#     max_len = max(y.size(0) for y in out)
#     y_padded = torch.zeros(len(out), max_len, dtype=out[0].dtype, device=out[0].device)
#     for i, y in enumerate(out):
#         y_padded[i, : y.size(0)] = y
#     return y_padded


# def complex_convolution_with_batch_filters(x, h):
#     """
#     x, h: shape [batch, length], each complex (torch.cfloat).
#     Returns: time-domain convolution [batch, length_x + length_h - 1], complex.
#     This is a direct Python loop approach for clarity (not the most efficient).
#     """
#     batch_size = x.size(0)
#     x_len = x.size(1)
#     h_len = h.size(1)
#     out_len = x_len + h_len - 1

#     outputs = []
#     for b in range(batch_size):
#         y_b = torch.zeros(out_len, dtype=torch.cfloat, device=x.device)
#         xb = x[b]  # [x_len] complex
#         hb = h[b]  # [h_len] complex
#         for n in range(out_len):
#             val = 0 + 0j
#             for k in range(n + 1):
#                 if (k < x_len) and ((n - k) < h_len):
#                     val += xb[k] * hb[n - k]
#             y_b[n] = val
#         outputs.append(y_b)

#     return torch.stack(outputs, dim=0)  # [batch, out_len] cfloat


def next_power_of_2(n: int) -> int:
    """
    Returns the smallest power of 2 >= n.
    """
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def complex_convolution_with_batch_filters(
    x: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    """
    Efficiently computes the full (1D) convolution of x with h for each batch item, in the complex domain,
    via FFT multiplication.

    Args:
        x: [batch, length_x], complex (torch.cfloat).
        h: [batch, length_h], complex (torch.cfloat).

    Returns:
        out: [batch, length_x + length_h - 1], complex (torch.cfloat).
    """

    # 1) Determine output length and choose FFT size
    batch_size, length_x = x.shape
    _, length_h = h.shape
    out_len = length_x + length_h - 1

    # 2) Zero-pad length for FFT => next power-of-2 for efficiency
    fft_len = next_power_of_2(out_len)

    # 3) Compute FFT along the time dimension
    X = torch.fft.fft(x, n=fft_len, dim=1)  # [batch, fft_len], cfloat
    H = torch.fft.fft(h, n=fft_len, dim=1)  # [batch, fft_len], cfloat

    # 4) Multiply in the frequency domain
    Y = X * H  # [batch, fft_len], cfloat

    # 5) Inverse FFT to get the time-domain result
    y_time = torch.fft.ifft(Y, n=fft_len, dim=1)  # [batch, fft_len], cfloat

    # 6) Crop to the "full" convolution size (length_x + length_h - 1)
    return y_time[:, :out_len]
