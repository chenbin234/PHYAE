import math

import numpy as np
import torch
import torch.nn.functional as F


def rcosdesign(beta, span, sps, offset=0.0, shape="sqrt", num_taps=None):
    """Discrete, truncated, and normalized (root)-raised cosine filter.

    API modeled after MATLAB's `rcosdesign`.

    Args:
        beta: Float in ``[0, 1]``.
        span: Integer. Number of symbols to which the impulse response is truncated.
        sps: Integer. Samples per symbol.
        offset: Scalar or 1D array/tensor of timing offsets (in symbol durations).
          A scalar offset returns a 1D tensor of shape ``[num_taps]``.
          A 1D offset returns a 2D tensor of shape ``[batch_size, num_taps]``.
        shape: ``"sqrt"`` for root-raised cosine or ``"normal"`` for raised cosine.
        num_taps: Optional integer. If provided, designs a filter with exactly ``num_taps`` taps.
          An even ``num_taps`` corresponds to a half-sample/half-symbol centered time grid.

    Returns:
        ``torch.Tensor`` of shape ``[num_taps]`` (scalar offset) or ``[batch_size, num_taps]``.
    """
    sps = int(sps)
    if sps < 1:
        raise ValueError("sps must be >= 1")

    if (span * sps) % 2 != 0:
        raise ValueError("Product of span and sps must be even.")

    if num_taps is None:
        num_taps = span * sps + 1
    num_taps = int(num_taps)
    if num_taps < 1:
        raise ValueError("num_taps must be >= 1")

    td = 1.0 / float(sps)

    offset = torch.as_tensor(offset, dtype=torch.float32)
    t = td * (
        torch.arange(num_taps, device=offset.device, dtype=torch.float32)
        - (num_taps - 1) / 2.0
    )

    if offset.ndim == 0:
        t = t + offset
    else:
        offset = offset.reshape(-1)
        t = t.unsqueeze(0) + offset.unsqueeze(1)

    # Match legacy TensorFlow implementation which applied a small shift to avoid
    # the removable singularities.
    t = t + 1e-5

    if shape == "sqrt":
        h = g_rrc(t, 1.0, beta)
    elif shape == "normal":
        h = g_rc(t, 1.0, beta)
    else:
        raise ValueError("shape must be either 'sqrt' or 'normal'")

    h = h / torch.sqrt(torch.sum(h**2, dim=-1, keepdim=True))
    return h


def g_rc(t, T, beta):
    """Raised cosine filter impulse response.

    See https://en.wikipedia.org/wiki/Raised-cosine_filter
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    T = float(T)
    beta = float(beta)

    t_over_T = t / T
    if beta == 0.0:
        return torch.sinc(t_over_T)

    denom = 1.0 - (2.0 * beta * t_over_T) ** 2
    h = torch.sinc(t_over_T) * torch.cos(math.pi * beta * t_over_T) / denom

    eps = 1e-8
    h_sing = (math.pi / 4.0) * torch.sinc(
        torch.tensor(1.0 / (2.0 * beta), device=t.device, dtype=t.dtype)
    )
    h = torch.where(torch.abs(denom) < eps, h_sing, h)
    return h


def g_rrc(t, T, beta):
    """Root-raised cosine filter impulse response.

    See https://en.wikipedia.org/wiki/Root-raised-cosine_filter
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    T = float(T)
    beta = float(beta)

    if beta == 0.0:
        return (1.0 / math.sqrt(T)) * torch.sinc(t / T)

    pi = math.pi
    t_over_T = t / T

    numerator = torch.sin(pi * t_over_T * (1.0 - beta)) + (
        4.0 * beta * t_over_T * torch.cos(pi * t_over_T * (1.0 + beta))
    )
    denominator = pi * t_over_T * (1.0 - (4.0 * beta * t_over_T) ** 2)
    h = (1.0 / math.sqrt(T)) * numerator / denominator

    eps = 1e-8
    h0 = (1.0 / math.sqrt(T)) * (1.0 - beta + 4.0 * beta / pi)
    h = torch.where(
        torch.abs(t) < eps, torch.as_tensor(h0, dtype=h.dtype, device=h.device), h
    )

    t1 = T / (4.0 * beta)
    h1 = (
        beta
        / math.sqrt(2.0 * T)
        * (
            (1.0 + 2.0 / pi) * math.sin(pi / (4.0 * beta))
            + (1.0 - 2.0 / pi) * math.cos(pi / (4.0 * beta))
        )
    )
    h = torch.where(
        torch.abs(torch.abs(t) - t1) < eps,
        torch.as_tensor(h1, dtype=h.dtype, device=h.device),
        h,
    )
    return h


def upsample(x, sps):
    """Add ``sps-1`` zeros after each element of ``x`` along the last dimension."""
    if sps < 1:
        raise ValueError("sps must be >= 1")

    x = torch.as_tensor(x)
    out_shape = list(x.shape)
    out_shape[-1] = out_shape[-1] * int(sps)
    y = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    y[..., :: int(sps)] = x
    return y


def upsample_filters(x, pulses, sps):
    """Upsample each row of ``x`` and filter with the corresponding row of ``pulses``."""
    xu = upsample(x, sps)
    return complex_convolution_with_batch_filters(xu, pulses, padding="SAME")


def complex_convolution_with_batch_filters(x, w, padding="SAME"):
    """Cross-correlate complex signals with per-sample complex filters.

    Args:
        x: Tensor of shape ``[batch_size, sequence]`` (real or complex).
        w: Tensor of shape ``[batch_size, filter_sequence]`` or ``[1, filter_sequence]``.
        padding: ``"SAME"`` or ``"VALID"``.

    Returns:
        Complex tensor of shape ``[batch_size, out_len]`` where ``out_len`` matches TF's
        ``depthwise_conv2d`` behavior for the equivalent 1D setup.
    """
    x = torch.as_tensor(x)
    w = torch.as_tensor(w)

    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("x and w must be rank-2 tensors of shape [batch, sequence].")

    x_compl = x.to(torch.complex64)
    w_compl = w.to(dtype=torch.complex64, device=x_compl.device)

    x_batch = int(x_compl.shape[0])
    w_batch = int(w_compl.shape[0])
    if w_batch == 1 and x_batch > 1:
        w_compl = w_compl.repeat(x_batch, 1)
    if int(w_compl.shape[0]) != x_batch:
        raise ValueError("w must have batch size 1 or the same batch size as x.")

    padding = padding.upper()
    if padding not in ("SAME", "VALID"):
        raise ValueError("padding must be 'SAME' or 'VALID'")

    k = int(w_compl.shape[1])
    if padding == "SAME":
        total_pad = max(k - 1, 0)
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
    else:
        pad_left = 0
        pad_right = 0

    def _corr1d_grouped(x_real, w_real):
        x_work = x_real.unsqueeze(0)  # [1, B, N]
        if pad_left or pad_right:
            x_work = F.pad(x_work, (pad_left, pad_right))
        w_weight = w_real.unsqueeze(1)  # [B, 1, K]
        return F.conv1d(x_work, w_weight, groups=x_batch)  # [1, B, out_len]

    x_r = x_compl.real.to(torch.float32)
    x_i = x_compl.imag.to(torch.float32)
    w_r = w_compl.real.to(torch.float32)
    w_i = w_compl.imag.to(torch.float32)

    y_r = _corr1d_grouped(x_r, w_r) - _corr1d_grouped(x_i, w_i)
    y_i = _corr1d_grouped(x_i, w_r) + _corr1d_grouped(x_r, w_i)

    return torch.complex(y_r, y_i).squeeze(0)


def pulse_shaping(
    x, beta=0.22, span=16, sps=4, shape="sqrt", offsets=np.array([0.0]), num_taps=None
):
    """Upsample and apply (root)-raised cosine pulse shaping.

    Args:
        x: Complex tensor of shape ``[batch_size, n]``.
        beta: Roll-off factor.
        span: Filter span (symbols).
        sps: Samples per symbol (upsampling factor).
        shape: ``"sqrt"`` (RRC) or ``"normal"`` (RC).
        offsets: Scalar or 1D offsets (length 1 or ``batch_size``). Non-tensor inputs are
          converted to a float32 tensor on ``x.device``.
        num_taps: Optional filter length.

    Returns:
        Complex tensor of shape ``[batch_size, n * sps]`` (for ``padding="SAME"``).
    """
    x = torch.as_tensor(x)
    offsets_t = torch.as_tensor(offsets, dtype=torch.float32, device=x.device)
    pulses = rcosdesign(
        beta=beta, span=span, sps=sps, offset=offsets_t, shape=shape, num_taps=num_taps
    )
    if pulses.ndim == 1:
        pulses = pulses.unsqueeze(0)

    return upsample_filters(x, pulses, sps)


if __name__ == "__main__":
    batch_size = 4
    n = 64
    max_sampling_time_offset = 0.49
    offsets = np.random.uniform(-max_sampling_time_offset, 0, size=batch_size).astype(
        np.float32
    )

    x = torch.complex(
        torch.randn([batch_size, n], dtype=torch.float32),
        torch.randn([batch_size, n], dtype=torch.float32),
    )
    y = pulse_shaping(
        x, beta=0.22, span=16, sps=1, shape="normal", offsets=offsets, num_taps=16
    )
    print(y.shape, y.dtype)
