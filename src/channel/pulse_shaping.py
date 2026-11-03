# Necessary Imports
import numpy as np
import tensorflow as tf


def rcosdesign(beta, span, sps, offset=0.0, shape="sqrt", num_taps=None):
    """Discrete, Truncated, and Normalized (Root)-Raised Cosine Filter
       API taken from https://fr.mathworks.com/help/signal/ref/rcosdesign.html

    Args:
      beta: Scalar float within `[0,1]`
      span: Scalar integer - Number of symbols to which the impulse response is truncated to
      sps: Scalar integer - Samples per symbol
      offset: Scalar or 1D numpy array of shape `[batch_size]` represeting the timinig offset(s) in multiples of the symbol duration
      shape: String equal to `sqrt`(root-raised cosine) `or `normal`(raised cosine)
      num_taps: Optional integer. If provided, designs a filter with exactly `num_taps` taps. An even
        `num_taps` corresponds to a half-sample/half-symbol centered time grid.

    Returns:
      Numpy array of shape `[batch_size, num_taps]` or `[num_taps]` if `batch_size==1`
    """
    if not span * sps % 2 == 0:
        raise NameError("Product of span and sps must be even.")

    # offsets = np.array(offset, ndmin=1)
    # offsets = np.expand_dims(offsets, axis=1)
    # batch_size = offsets.shape[0]

    td = 1 / sps
    if num_taps is None:
        num_taps: Unknown = span * sps + 1
    num_taps = int(num_taps)
    if num_taps < 1:
        raise ValueError("num_taps must be >= 1")

    # Centered time grid:
    # - odd  num_taps -> includes t=0
    # - even num_taps -> centered at half-sample/half-symbol (linear-phase with half-sample delay)
    t = td * (tf.range(num_taps, dtype=tf.float32) - (num_taps - 1) / 2.0)

    offset = tf.convert_to_tensor(offset, dtype=tf.float32)
    if offset.shape.rank == 0:
        t = t + offset
    else:
        offset = tf.reshape(offset, [-1])
        t = t[tf.newaxis, :] + offset[:, tf.newaxis]
    # t_array = np.repeat(np.expand_dims(t, 0), batch_size, axis=0) + offsets
    if shape == "sqrt":
        h = g_rrc(t + 0.0001, 1, beta)
    elif shape == "normal":
        h = g_rc(t + 0.0001, 1, beta)
    else:
        raise NameError("Shape must be either sqrt or normal")
    h = h / tf.sqrt(tf.reduce_sum(h**2, axis=-1, keepdims=True))

    return h


def g_rc(t, T, beta):
    """Raised Cosine Filter Impulse Response
       see https://en.wikipedia.org/wiki/Raised-cosine_filter

    Args:
      t: Numpy array of arbitrary containing the sample times
      T: Scalar float
      beta : Scalar float

    Results:
      A numpy array of the same shape as `t`
    """
    h = tf.zeros_like(t, dtype=tf.float32)
    # h = np.zeros_like(t,dtype=np.float32)
    if beta == 0:
        h = tf.experimental.numpy.sinc(t / T)
    else:
        # Case t=+-T/(2*beta)
        # ind1 = tf.not_equal(tf.abs(t), T/2/beta)
        # t1 = t[ind1]
        h = (
            tf.experimental.numpy.sinc(t / T)
            * tf.cos(np.pi * beta * t / T)
            / (1 - (2 * beta * t / T) ** 2)
        )
        # Case otherwise
    #         ind2 = np.logical_not(ind1)

    #         h[ind2] = np.pi/4*np.sinc(1/(2*beta))
    return h


def g_rrc(t, T, beta):
    """Root-Raised Cosine Filter Impulse Response
       see https://en.wikipedia.org/wiki/Root-raised-cosine_filter

    Args:
      t: Numpy array of arbitrary shape containing the sample times
      T: Scalar float
      beta : Scalar float

    Results:
      A numpy array of the same shape as `t`
    """

    t = tf.cast(t, tf.float32)
    T = tf.cast(T, tf.float32)
    beta = tf.cast(beta, tf.float32)

    def _beta_zero():
        return (1.0 / tf.sqrt(T)) * tf.experimental.numpy.sinc(t / T)

    def _beta_nonzero():
        pi = tf.constant(np.pi, dtype=tf.float32)
        t_over_T = t / T

        numerator = tf.sin(pi * t_over_T * (1.0 - beta)) + (
            4.0 * beta * t_over_T * tf.cos(pi * t_over_T * (1.0 + beta))
        )
        denominator = pi * t_over_T * (1.0 - (4.0 * beta * t_over_T) ** 2)
        h = (1.0 / tf.sqrt(T)) * numerator / denominator

        eps = tf.constant(1e-8, dtype=tf.float32)
        h0 = (1.0 / tf.sqrt(T)) * (1.0 - beta + 4.0 * beta / pi)
        h = tf.where(tf.abs(t) < eps, h0, h)

        t1 = T / (4.0 * beta)
        h1 = (
            beta
            / tf.sqrt(2.0 * T)
            * (
                (1.0 + 2.0 / pi) * tf.sin(pi / (4.0 * beta))
                + (1.0 - 2.0 / pi) * tf.cos(pi / (4.0 * beta))
            )
        )
        h = tf.where(tf.abs(tf.abs(t) - t1) < eps, h1, h)

        return h

    return tf.cond(tf.equal(beta, 0.0), _beta_zero, _beta_nonzero)


def upsample(x, sps):
    """Add `sps-1` zeros after each element of `x` along the last dimension

    Args:
      x: Tensor of arbitrary shape and dtype
      sps: Integer greater or equal to `1`

    Returns:
      Tensor of the same shape as `x` but with the last dimension multiplied by `sps`.

    Comments:
      If `x` has shape `[n1,n2,n3]`, the output has shape `[n1,n2,n3*sps]`
    """
    if sps < 1:
        raise NameError("sps must be >=1")
    shape = tf.concat([tf.shape(x), [sps - 1]], axis=-1)
    z = tf.zeros(shape=shape, dtype=x.dtype)
    x = tf.expand_dims(x, axis=-1)
    y = tf.concat([x, z], axis=tf.rank(x) - 1)
    newshape = tf.concat([tf.shape(x)[: tf.rank(x) - 2], [-1]], axis=-1)
    y = tf.reshape(y, newshape)
    return y


def upsample_filters(x, pulses, sps):
    """Filter each row of `x` with each row of `pulses` after upsampling of `x`

    Args:
      x: Tensor of shape `[batch_size, n]`
      pulses: Tensor of shape `[bacth_size, L]`
      sps: Integer corresponding to the upsampling factor, i.e. samples per symbol

    Returns:
      Tensor of shape `[batch_size, (n-1)*sps+L]`
    """

    # Dimensions for later computations
    n = tf.shape(x)[1]
    L = tf.shape(pulses)[1]
    n_out = (n - 1) * sps + L  # Check definition of `fading.convolution()`
    pad = (L - 1) // 2
    batch_size_pulse = tf.shape(pulses)[0]
    batch_size = tf.shape(x)[0]
    # Upsampling
    xu = upsample(x, sps)
    # xu = xu[:,:-sps+1]

    # xu = tf.repeat(xu,batch_size_pulse, axis=0)
    # pulses = tf.tile(pulses,[batch_size,1])
    # print([pulses])
    # # #Convolution
    # # Hmap = tf.py_func(convMap, [L, tf.shape(xu)[1], n_out], tf.int32)
    # xu_pad = tf.concat([tf.zeros([batch_size**2,pad],dtype=xu.dtype), xu , tf.zeros([batch_size**2,pad],dtype=xu.dtype)],axis=1)
    y = complex_convolution_with_batch_filters(xu, pulses, padding="SAME")
    # y = tf.reshape(y, [batch_size,batch_size_pulse,-1])
    return y


def complex_convolution_with_batch_filters(x, w, padding="SAME"):
    """Performs convolution with complex inputs and different complex weights per sample in batch

    Parameters
    ----------
    x : tf tensor [batch_size,sequence]
        input tensor
    w : tf tensor [batch_size,filter_sequence]
        weights tensor
    padding: 'SAME' or 'VALID'

    Returns
    -------
    y : `tf.Tensor`
        Result of applying convolution to x
        for 'FULL' pad with zeros
        for 'CYCLIC' pad cyclic

    Notes
    -----
    Uses tf.nn.depthwise_conv2d
    """

    x_compl = tf.cast(x, tf.complex64)
    w_compl = tf.cast(w, tf.complex64)

    # Support a single shared filter for all batch elements, or one filter per sample.
    x_batch = tf.shape(x_compl)[0]
    w_batch = tf.shape(w_compl)[0]
    w_compl = tf.cond(
        tf.equal(w_batch, 1),
        lambda: tf.tile(w_compl, [x_batch, 1]),
        lambda: w_compl,
    )
    tf.debugging.assert_equal(
        tf.shape(w_compl)[0],
        x_batch,
        message="w must have batch size 1 or the same batch size as x.",
    )

    padding = padding.upper()
    if padding not in ("SAME", "VALID"):
        raise ValueError("padding must be 'SAME' or 'VALID'")

    # Using channels of depthwise_conv2d as batch dimension.
    x = tf.expand_dims(
        tf.expand_dims(tf.transpose(x_compl, perm=[1, 0]), axis=0), axis=0
    )
    w = tf.expand_dims(
        tf.expand_dims(tf.transpose(w_compl, perm=[1, 0]), axis=0), axis=3
    )
    x_r = tf.math.real(x)
    x_i = tf.math.imag(x)
    w_r = tf.math.real(w)
    w_i = tf.math.imag(w)
    conv = lambda x, w: tf.nn.depthwise_conv2d(
        x, w, strides=[1, 1, 1, 1], padding=padding
    )
    y_r = conv(x_r, w_r) - conv(x_i, w_i)
    y_i = conv(x_i, w_r) + conv(x_r, w_i)
    y_compl = tf.complex(y_r, y_i)
    y = tf.transpose(tf.squeeze(y_compl, axis=[0, 1]), perm=[1, 0])
    return y


def pulse_shaping(
    x, beta=0.22, span=16, sps=4, shape="sqrt", offsets=np.array([0.0]), num_taps=None
):  # , n_coarse=1, min_offset=-0.5, max_offset=0.5):
    """Upsampling followed by pulse-shaping with a random timing offset

    Args:
      x: Tensor of shape `[batch_size,2,n]` (complex dtypes are not supported)
      beta: Scalar float corresponding to the beta-parameter of the raised-cosine filter
      span: Scalar integer corresponding to the length of the filter in multiples of the symbol duration
      sps: Scalar integer corresponding to the number of samples per symbol
      shape: String equal to `sqrt` or `normal` determining the pulse shape
      min_offset: Scalar float corresponding to the minimum time-offset in multiples of the symbol time
      max_offset: Scalar float corresponding to the maximum time-offset in multiples of the symbol time

    Returns:
      Tensor of shape `[batch_size,2,(n-1)*sps+span*sps+1]`
    """

    pulses = rcosdesign(
        beta=beta, span=span, sps=sps, offset=offsets, shape=shape, num_taps=num_taps
    )
    if pulses.shape.rank == 1:
        pulses = pulses[tf.newaxis, :]

    y = upsample_filters(x, pulses, sps)

    return y


if __name__ == "__main__":
    # generate a signal x of shape [batch_size, n], where each element is a complex symbol represented by its real and imaginary parts in the last dimension
    batch_size = 4
    n = 64
    max_sampling_time_offset = 0.49
    # generate offset of shape [batch_size] with values uniformly distributed in [-max_sampling_time_offset, 0)
    offsets = np.random.uniform(-max_sampling_time_offset, 0, size=batch_size)

    x = tf.complex(
        tf.random.normal([batch_size, n], dtype=tf.float32),
        tf.random.normal([batch_size, n], dtype=tf.float32),
    )
    y = pulse_shaping(
        x, beta=0.22, span=16, sps=1, shape="normal", offsets=offsets, num_taps=16
    )
    print(y.shape)
