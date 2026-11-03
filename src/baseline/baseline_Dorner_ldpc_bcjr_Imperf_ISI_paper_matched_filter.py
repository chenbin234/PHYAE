import itertools
import sys

import numpy as np
import sionna as sn

# Import TensorFlow and NumPy
import tensorflow as tf
from loguru import logger
from sionna.phy.utils.plotting import sim_ber

# For the implementation of the Keras models
from tensorflow.keras import Model

import wandb

# Configure NumPy print options for better logging
np.set_printoptions(
    linewidth=250,  # Increase line width
    threshold=1200,  # Show more array elements before truncating
    edgeitems=10,  # Show more items at beginning and end
    suppress=False,  # Don't suppress small values
    precision=6,  # Number of decimal places
)

# Set the random seed for reproducibility
tf.random.set_seed(5)

# Set the log level to ERROR
logger.remove()  # Remove the default logger
logger.add(sys.stderr, level="ERROR")  # Set to "DEBUG" to see all messages
logger.add("./src/baseline/log_file.log", level="ERROR", mode="w")


FRACTIONAL_FILTER_SPAN = 16


def modulation_bits_per_symbol(modulation):
    """Return the bit width for a supported modulation (case-insensitive)."""
    bit_widths = {"4qam": 2, "16qam": 4, "8psk": 3}
    try:
        return bit_widths[modulation.lower()]
    except (AttributeError, KeyError):
        raise ValueError("modulation must be '4qam', '16qam', or '8psk'.") from None


def create_constellation(modulation):
    """Create a unit-average-power constellation with Gray bit labels."""
    num_bits = modulation_bits_per_symbol(modulation)
    if modulation.lower() != "8psk":
        return sn.phy.mapping.Constellation("qam", num_bits)

    # Counterclockwise from phase 0, label the eight points as
    # 000, 001, 011, 010, 110, 111, 101, 100 (including Gray wraparound).
    # Sionna uses each point's ARRAY INDEX as its binary bit label, so store
    # the point at phase k*pi/4 at index gray(k), not at index k.
    phase_indices = np.arange(8, dtype=np.int32)
    gray_labels = phase_indices ^ (phase_indices >> 1)
    points = np.empty(8, dtype=np.complex64)
    points[gray_labels] = np.exp(1j * np.pi * phase_indices / 4)
    return sn.phy.mapping.Constellation(
        "custom", num_bits, points=points, normalize=True, center=False
    )


def g_rc(t: tf.Tensor, symbol_period: float = 1.0, beta: float = 0.22) -> tf.Tensor:
    """
    Raised Cosine Filter.

    Link: https://en.wikipedia.org/wiki/Raised-cosine_filter.

    Parameters
    ----------
    t : `tf.Tensor`
        Time points at which to evaluate the filter, in symbol durations.
    symbol_period : `float`
        Symbol duration (T).
    beta : `float`
        Roll-off factor, in [0, 1].

    """
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    symbol_period = tf.cast(symbol_period, tf.float32)
    beta = tf.cast(beta, tf.float32)
    singularity_eps = tf.constant(1e-6, dtype=tf.float32)

    def beta_zero() -> tf.Tensor:
        return tf.experimental.numpy.sinc(t / symbol_period) / symbol_period

    def beta_nonzero() -> tf.Tensor:
        normalized_t = t / symbol_period
        denominator = 1 - (2 * beta * normalized_t) ** 2
        general = (
            tf.experimental.numpy.sinc(normalized_t)
            * tf.cos(np.pi * beta * normalized_t)
            / (symbol_period * denominator)
        )

        singular_t = symbol_period / (2 * beta)
        at_singularity = tf.abs(tf.abs(t) - singular_t) <= singularity_eps
        singular_value = (
            np.pi
            / (4.0 * symbol_period)
            * tf.experimental.numpy.sinc(1.0 / (2.0 * beta))
        )
        return tf.where(at_singularity, singular_value, general)

    return tf.cond(tf.equal(beta, 0.0), beta_zero, beta_nonzero)


def create_fractional_filter(
    eps: tf.Tensor, n_up: int, filter_span: int = FRACTIONAL_FILTER_SPAN
) -> tf.Tensor:
    t = tf.linspace(
        -filter_span * 1.0 / 2,
        filter_span * 1.0 / 2,
        filter_span * n_up + 1,
    )
    t_shifted = tf.subtract(t, tf.expand_dims(eps, axis=1))
    p_e = g_rc(t_shifted, symbol_period=1.0, beta=0.22)
    return p_e / tf.sqrt(tf.reduce_sum(tf.square(p_e), axis=1, keepdims=True))


# ---------------------------------------------------
# Proakis channel impulse response
# ---------------------------------------------------
def proakisImpulseResponse_tf(batch_size, taps=5, debug=False):
    # Proakis type-C impulse response weights
    if taps == 5:
        w_proakis = tf.constant([0.227, 0.460, 0.688, 0.460, 0.227], dtype=tf.float32)
    elif taps == 3:
        w_proakis = tf.constant([0.407, 0.815, 0.407], dtype=tf.float32)
    elif taps == 1:
        w_proakis = tf.constant([1.0], dtype=tf.float32)
    w_proakis = w_proakis / tf.reduce_sum(tf.abs(w_proakis))

    # Truncate if taps < 5
    w_proakis = w_proakis[:taps]

    # Expand to [1, taps] so it can broadcast with (batch_size, taps)
    w_proakis = tf.reshape(w_proakis, [1, -1])

    # # Generate real and imaginary parts
    # if debug:
    #     w_proakis = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0], dtype=tf.float32)
    #     # w_proakis = w_proakis / tf.reduce_sum(tf.abs(w_proakis))

    #     w_proakis = tf.reshape(w_proakis, [1, -1])
    #     h_real = tf.sqrt(w_proakis) * tf.constant(
    #         [0.0, 1.0, 1.0, 1.0, 1.0], dtype=tf.float32
    #     )
    #     h_imag = tf.zeros_like(h_real)
    # else:
    #     h_real = tf.sqrt(w_proakis / 2.0) * tf.random.normal(
    #         shape=(batch_size, taps), dtype=tf.float32
    #     )

    #     h_imag = tf.sqrt(w_proakis / 2.0) * tf.random.normal(
    #         shape=(batch_size, taps), dtype=tf.float32
    #     )
    h_real = tf.sqrt(w_proakis / 2.0) * tf.random.normal(
        shape=(batch_size, taps), dtype=tf.float32
    )

    h_imag = tf.sqrt(w_proakis / 2.0) * tf.random.normal(
        shape=(batch_size, taps), dtype=tf.float32
    )

    # Combine into complex tensor
    h_complex = tf.complex(h_real, h_imag)

    return h_complex


# ---------------------------------------------------
# Complex convolution with batch-specific filters
# ---------------------------------------------------
def complex_convolution_with_batch_filters_tf(x, w, padding="SAME"):
    """
    x: Tensor of shape (batch, N), complex or real
    w: Tensor of shape (batch, K), complex or real
    Returns: Tensor of shape (batch, N), complex
    """

    # Cast to complex
    x = tf.cast(x, tf.complex64)
    w = tf.cast(w, tf.complex64)

    N = tf.shape(x)[1]
    K = tf.shape(w)[1]

    # FFT size for linear convolution
    L = N + K - 1

    # Zero-pad
    x_pad = tf.pad(x, [[0, 0], [0, L - N]])
    w_pad = tf.pad(w, [[0, 0], [0, L - K]])

    # FFT
    X = tf.signal.fft(x_pad)
    W = tf.signal.fft(w_pad)

    # Multiply in frequency domain
    Y = X * W

    # IFFT
    y_full = tf.signal.ifft(Y)

    # SAME padding behavior: center crop to length N
    if padding.upper() == "SAME":
        start = K // 2
        y = y_full[:, start : start + N]
    else:  # VALID
        y = y_full[:, :N]

    return y


def complex_convolution_full_tf(x, w):
    """
    Full linear convolution with batch-specific filters.

    Args:
        x: Tensor of shape (batch, N), complex or real
        w: Tensor of shape (batch, K), complex or real

    Returns:
        Tensor of shape (batch, N+K-1), complex
    """
    x = tf.cast(x, tf.complex64)
    w = tf.cast(w, tf.complex64)

    N = tf.shape(x)[1]
    K = tf.shape(w)[1]
    L = N + K - 1

    x_pad = tf.pad(x, [[0, 0], [0, L - N]])
    w_pad = tf.pad(w, [[0, 0], [0, L - K]])

    X = tf.signal.fft(x_pad)
    W = tf.signal.fft(w_pad)

    y_full = tf.signal.ifft(X * W)
    return y_full


def proakis_c_channel_tf(
    x,
    max_offset,
    taps,
    is_data_interference=True,
    pilot_length=None,
    guard_length=0,
    debug=False,
):
    """
    x: shape (batch, n), complex

    When ``is_data_interference`` is ``False``, ``pilot_length`` must be
    provided so the preamble and payload can be generated without mutual ISI
    while preserving the original output shape. ``guard_length`` extends the
    clean pilot-only region so the pilot tail remains observable.
    """
    x = tf.cast(x, tf.complex64)
    batch = tf.shape(x)[0]

    # 1. Channel impulse response
    h = proakisImpulseResponse_tf(
        batch_size=batch, taps=taps, debug=debug
    )  # shape [batch, h_len], complex

    h_len = tf.shape(h)[1]

    # max_offset = (
    #     tf.maximum(output_length, tf.shape(x)[1] + h_len - 1)
    #     - tf.shape(x)[1]
    #     - h_len
    #     + 1
    # )
    # fix max_offset to 11
    max_offset = tf.constant(max_offset, dtype=tf.int32)

    # 5. Integer timing offset (random)

    offset = tf.random.uniform(
        shape=(batch,), minval=0, maxval=max_offset + 1, dtype=tf.int32
    )
    # debug: use fixed offset to check the error floor issue
    # offset = tf.random.uniform(shape=(batch,), minval=6, maxval=7, dtype=tf.int32)

    offset_one_hot = tf.one_hot(offset, depth=max_offset + 1, dtype=tf.float32)

    if taps == 5:
        offset_one_hot_padded = tf.pad(offset_one_hot, paddings=[[0, 0], [2, 2]])
    elif taps == 3:
        offset_one_hot_padded = tf.pad(offset_one_hot, paddings=[[0, 0], [1, 1]])
    elif taps == 1:
        offset_one_hot_padded = offset_one_hot
    else:
        raise ValueError("Unsupported number of taps. Must be 1, 3, or 5.")

    # 6. Create batch-specific time-shifted channel impulse response
    h_offset = complex_convolution_with_batch_filters_tf(offset_one_hot_padded, h)

    # 7. Pad signal for convolution
    pad_signal_left = tf.shape(h_offset)[1] // 2
    pad_signal_right = (tf.shape(h_offset)[1] - 1) // 2

    x_padded = tf.pad(x, paddings=[[0, 0], [pad_signal_left, pad_signal_right]])

    # 8. Filter the signal by the time-shifted channel
    y = complex_convolution_with_batch_filters_tf(x_padded, h_offset)

    # randomly generate sampling time offsets for imperfect timing case, and apply
    # them to the output signal
    sampling_offsets = tf.random.uniform(
        shape=(batch,), minval=0.0, maxval=1.0, dtype=tf.float32
    )
    pulses = create_fractional_filter(
        sampling_offsets, n_up=1, filter_span=FRACTIONAL_FILTER_SPAN
    )

    def apply_fractional_sampling(signal):
        # Apply the same fractional pulse to each branch; the final splice
        # determines whether pilot/payload interference is preserved or removed.
        pad_left = tf.shape(pulses)[1] // 2
        pad_right = tf.shape(pulses)[1] // 2
        signal = tf.pad(signal, paddings=[[0, 0], [pad_left, pad_right]])
        return complex_convolution_with_batch_filters_tf(signal, pulses)

    if is_data_interference:
        y = apply_fractional_sampling(y)
    else:
        y = apply_fractional_sampling(y)

    return y, h, offset


class CodedSystemBlockFading(Model):  # Inherits from Keras Model
    def __init__(
        self,
        num_bits_per_symbol,
        n,
        Channel_CODERATE,
        perfect_sync=True,
        perfect_csi=True,
        data_interference=False,
        preamble_length=None,
        cyclic_prefix_length=0,
        payload_length=44,
        zc_length=None,
        cyclic_suffix_length=0,
        num_taps=1,
        num_effective_taps=1,
        num_estimation_taps=None,
        pilot_guard_length=None,
        max_delay=12,  # Maximum delay
        bp_iterations=20,
        turbo_iterations=10,
        bp_damping=0.7,
        feedback_weight=0.2,
        llr_max=20.0,
        debug=False,
        sync_detection_alpha=1.15,
        sync_onset_threshold=38.0,
        modulation=None,
    ):
        super().__init__()  # Must call the Keras model initializer

        self.debug = debug
        # Infer QAM for existing callers that only provide the bit width.
        if modulation is None:
            modulation = f"{2**num_bits_per_symbol}qam"
        expected_bits = modulation_bits_per_symbol(modulation)
        if num_bits_per_symbol != expected_bits:
            raise ValueError(
                f"{modulation} requires num_bits_per_symbol={expected_bits}."
            )
        if n != payload_length * expected_bits:
            raise ValueError("n must equal payload_length * num_bits_per_symbol.")
        self.modulation = modulation.lower()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.n = n
        self.k = int(n * Channel_CODERATE)
        self.channel_coderate = Channel_CODERATE
        self.perfect_sync = perfect_sync
        self.perfect_csi = perfect_csi
        self.data_interference = data_interference
        self.cyclic_suffix_length = int(cyclic_suffix_length)
        self.cyclic_prefix_length = int(cyclic_prefix_length)
        if self.cyclic_prefix_length < 0:
            raise ValueError("cyclic_prefix_length must be non-negative.")
        if self.cyclic_suffix_length < 0:
            raise ValueError("cyclic_suffix_length must be non-negative.")
        if zc_length is None:
            if preamble_length is None:
                preamble_length = 20
            self.preamble_length = int(preamble_length)
            self.zc_length = (
                self.preamble_length
                - self.cyclic_prefix_length
                - self.cyclic_suffix_length
            )
        else:
            self.zc_length = int(zc_length)
            self.preamble_length = (
                self.cyclic_prefix_length + self.zc_length + self.cyclic_suffix_length
            )
            if (
                preamble_length is not None
                and int(preamble_length) != self.preamble_length
            ):
                raise ValueError(
                    "preamble_length must equal cyclic_prefix_length + zc_length "
                    "+ cyclic_suffix_length when zc_length is provided."
                )
        if self.preamble_length <= 0:
            raise ValueError("preamble_length must be positive.")
        if self.zc_length <= 0:
            raise ValueError(
                "zc_length must be positive and leave room after cyclic extensions."
            )
        if self.cyclic_prefix_length > self.zc_length:
            raise ValueError(
                "cyclic_prefix_length must not exceed the generated ZC length."
            )
        if self.cyclic_suffix_length > self.zc_length:
            raise ValueError(
                "cyclic_suffix_length must not exceed the generated ZC length."
            )
        self.payload_length = payload_length  # length of payload symbols
        self.num_taps = num_taps
        self.num_equalization_taps = int(num_effective_taps)
        if num_estimation_taps is None:
            num_estimation_taps = self.num_equalization_taps
        self.num_estimation_taps = max(
            self.num_equalization_taps, int(num_estimation_taps)
        )
        if data_interference and pilot_guard_length not in (None, 0):
            raise ValueError(
                "pilot_guard_length must be 0 when data_interference=True."
            )
        if not data_interference:
            self.pilot_guard_length = max(0, self.num_estimation_taps - 1)
        else:
            self.pilot_guard_length = 0

        # Backward-compatible alias used elsewhere in this file.
        self.num_effective_taps = self.num_equalization_taps
        self.max_delay = max_delay
        self.sync_detection_alpha = float(sync_detection_alpha)
        if self.sync_detection_alpha < 0.0:
            raise ValueError("sync_detection_alpha must be non-negative.")
        self.sync_onset_threshold = float(sync_onset_threshold)
        if self.sync_onset_threshold < 0.0:
            raise ValueError("sync_onset_threshold must be non-negative.")
        self.bp_iterations = bp_iterations
        self.turbo_iterations = turbo_iterations
        self.bp_damping = bp_damping
        self.feedback_weight = feedback_weight
        self.llr_max = float(llr_max)
        self.constellation = create_constellation(self.modulation)
        self.awgn_channel = sn.phy.channel.AWGN()
        self.binary_source = sn.phy.mapping.BinarySource()
        self.encoder = sn.phy.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.mapper = sn.phy.mapping.Mapper(constellation=self.constellation)

        self.demapper = sn.phy.mapping.Demapper("app", constellation=self.constellation)
        # Initialize the LDPC decoders
        self.decoder_soft = sn.phy.fec.ldpc.LDPC5GDecoder(
            self.encoder,
            hard_out=False,
            num_iter=self.bp_iterations,
            return_infobits=False,
            llr_max=self.llr_max,
        )
        self.decoder_hard = sn.phy.fec.ldpc.LDPC5GDecoder(
            self.encoder,
            hard_out=True,
            num_iter=self.bp_iterations,
            llr_max=self.llr_max,
        )

        # Generate the final preamble once:
        # cyclic prefix + Zadoff-Chu core + cyclic suffix.
        self._zc_core_base = self.generateZadoffChuPilot(
            self.zc_length, u=1
        )  # shape: [zc_length]
        self._zc_seq_base = self.addCyclicPrefixAndSuffix(
            self._zc_core_base,
            self.cyclic_prefix_length,
            self.cyclic_suffix_length,
        )  # shape: [preamble_length]
        tf.debugging.assert_equal(
            tf.shape(self._zc_seq_base)[0],
            self.preamble_length,
            message="Cyclic prefix + ZC + cyclic suffix must equal preamble_length.",
        )

        # Fixed bit interleaver (turbo loop)
        rng = np.random.RandomState(3)
        perm = rng.permutation(self.n)
        inv_perm = np.argsort(perm)
        self._interleaver_perm: Operation | _EagerTensorBase = tf.constant(
            perm, dtype=tf.int32
        )
        self._deinterleaver_perm = tf.constant(inv_perm, dtype=tf.int32)

        # Precompute constellation and trellis for BCJR
        self._build_constellation_table()
        self._build_bcjr_trellis()

        # Runtime diagnostics aggregated per Eb/N0 point during sim_ber sweeps.
        self.last_runtime_diagnostics = None
        self.runtime_diagnostics_history = []
        self._diag_current_ebno = None
        self._diag_accumulator = None

    def get_zc_seq(self, batch_size):
        """
        Get the cyclic-prefix/suffix Zadoff-Chu preamble tiled for the batch.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            zc_seq: CP + Zadoff-Chu + CS preamble
                [batch_size, preamble_length]
        """
        zc_seq = tf.expand_dims(
            self._zc_seq_base, axis=0
        )  # shape: [1, preamble_length]
        zc_seq = tf.tile(
            zc_seq, [batch_size, 1]
        )  # shape: [batch_size, preamble_length]
        return zc_seq

    @staticmethod
    def _scalar_float(value):
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        return float(np.asarray(value).reshape(()))

    def reset_runtime_diagnostics(self):
        self.last_runtime_diagnostics = None
        self.runtime_diagnostics_history = []
        self._diag_current_ebno = None
        self._diag_accumulator = None

    def _start_runtime_diagnostics(self, ebno_db):
        self._diag_current_ebno = self._scalar_float(ebno_db)
        self._diag_accumulator = {
            "num_batches": 0,
            "num_packets": 0,
            "residual_min_sum": 0.0,
            "timing_error_sum": 0.0,
            "timing_error_abs_sum": 0.0,
            "timing_hit_sum": 0.0,
            "timing_error_counts": {},
            "coarse_timing_error_counts": {},
            "channel_nmse_sum": 0.0,
            "channel_nmse_packets": 0,
        }

    def _record_runtime_diagnostics(
        self,
        ebno_db,
        residual_min,
        timing_error,
        timing_hit,
        h_hat,
        h_true,
        payload_shift=None,
        coarse_timing_error=None,
    ):
        ebno_scalar = self._scalar_float(ebno_db)
        if self._diag_current_ebno is None:
            self._start_runtime_diagnostics(ebno_scalar)
        elif abs(ebno_scalar - self._diag_current_ebno) > 1e-9:
            self.flush_runtime_diagnostics()
            self._start_runtime_diagnostics(ebno_scalar)

        packet_count = int(tf.size(residual_min).numpy())
        acc = self._diag_accumulator
        acc["num_batches"] += 1
        acc["num_packets"] += packet_count
        acc["residual_min_sum"] += float(tf.reduce_sum(residual_min).numpy())
        acc["timing_error_sum"] += float(tf.reduce_sum(timing_error).numpy())
        acc["timing_error_abs_sum"] += float(
            tf.reduce_sum(tf.abs(timing_error)).numpy()
        )
        acc["timing_hit_sum"] += float(
            tf.reduce_sum(tf.cast(timing_hit, tf.float32)).numpy()
        )
        timing_error_int = tf.cast(tf.round(timing_error), tf.int32)
        timing_error_values = tf.reshape(timing_error_int, [-1])
        values, _, counts = tf.unique_with_counts(timing_error_values)
        for value, count in zip(values.numpy(), counts.numpy()):
            value = int(value)
            acc["timing_error_counts"][value] = acc["timing_error_counts"].get(
                value, 0
            ) + int(count)

        if coarse_timing_error is not None:
            coarse_timing_error_int = tf.cast(tf.round(coarse_timing_error), tf.int32)
            coarse_timing_error_values = tf.reshape(coarse_timing_error_int, [-1])
            values, _, counts = tf.unique_with_counts(coarse_timing_error_values)
            for value, count in zip(values.numpy(), counts.numpy()):
                value = int(value)
                acc["coarse_timing_error_counts"][value] = acc[
                    "coarse_timing_error_counts"
                ].get(value, 0) + int(count)

        h_hat = tf.cast(h_hat, tf.complex64)
        h_true = tf.cast(h_true, tf.complex64)

        # Only record channel NMSE when the estimated channel and the true
        # channel have compatible dimensions. In particular, skip NMSE when
        # the runtime effective equalization taps exceed the number of model
        # taps to avoid shape-mismatch errors (see issue traceback).
        record_nmse = True
        if (
            getattr(self, "num_effective_taps", None) is not None
            and getattr(self, "num_taps", None) is not None
        ):
            if int(self.num_effective_taps) > int(self.num_taps):
                record_nmse = False

        if record_nmse:
            # compute NMSE only when shapes are compatible
            try:
                channel_nmse = (tf.reduce_sum(tf.abs(h_hat - h_true) ** 2, axis=1)) / (
                    tf.reduce_sum(tf.abs(h_true) ** 2, axis=1) + 1e-20
                )
                acc["channel_nmse_sum"] += float(tf.reduce_sum(channel_nmse).numpy())
                acc["channel_nmse_packets"] += int(tf.size(channel_nmse).numpy())
            except Exception:
                # If any TensorFlow shape/broadcast error occurs, skip NMSE
                # recording but continue collecting other diagnostics.
                pass

    def flush_runtime_diagnostics(self):
        if self._diag_accumulator is None or self._diag_accumulator["num_packets"] == 0:
            return None

        acc = self._diag_accumulator
        num_packets = acc["num_packets"]
        timing_error_counts = dict(sorted(acc["timing_error_counts"].items()))
        der_count = sum(
            count for error, count in timing_error_counts.items() if abs(error) > 5
        )
        coarse_timing_error_counts = dict(
            sorted(acc["coarse_timing_error_counts"].items())
        )
        diagnostics = {
            "ebno_db": self._diag_current_ebno,
            "num_batches": acc["num_batches"],
            "num_packets": num_packets,
            "residual_min_mean": acc["residual_min_sum"] / num_packets,
            "timing_error_mean": acc["timing_error_sum"] / num_packets,
            "timing_error_mae": acc["timing_error_abs_sum"] / num_packets,
            "timing_hit_rate": acc["timing_hit_sum"] / num_packets,
            "timing_error_counts": timing_error_counts,
            "der": der_count / num_packets,
            "coarse_timing_error_counts": coarse_timing_error_counts,
            "channel_nmse_mean": None,
        }
        if acc["channel_nmse_packets"] > 0:
            diagnostics["channel_nmse_mean"] = (
                acc["channel_nmse_sum"] / acc["channel_nmse_packets"]
            )

        self.last_runtime_diagnostics = diagnostics
        self.runtime_diagnostics_history.append(diagnostics)

        # Safely format channel NMSE which can be None when not recorded.
        channel_nmse_val = diagnostics.get("channel_nmse_mean")
        if channel_nmse_val is None:
            channel_nmse_str = "None"
        else:
            try:
                channel_nmse_str = f"{channel_nmse_val:.4e}"
            except Exception:
                channel_nmse_str = str(channel_nmse_val)

        print(
            "[rx-diag] "
            f"Eb/N0={diagnostics['ebno_db']:.2f} dB "
            f"batches={diagnostics['num_batches']} "
            f"packets={diagnostics['num_packets']} "
            f"residual_min={diagnostics['residual_min_mean']:.4e} "
            f"k_hat-offset_true mean={diagnostics['timing_error_mean']:.4f} "
            f"mae={diagnostics['timing_error_mae']:.4f} "
            f"hit_rate={diagnostics['timing_hit_rate']:.4f} "
            f"DER={diagnostics['der']:.4e} "
            f"k_hat_coarse-offset_true_freq={diagnostics['coarse_timing_error_counts']} "
            f"k_hat_raw-offset_true_freq={diagnostics['timing_error_counts']} "
            f"channel_nmse={channel_nmse_str}"
        )

        self._diag_current_ebno = None
        self._diag_accumulator = None
        return diagnostics

    @staticmethod
    def generateZadoffChuPilot(N, u=1):
        N = int(N)
        u = int(u)
        if N <= 0:
            raise ValueError("Zadoff-Chu length N must be positive.")

        # Ensure u is coprime to N
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        if gcd(u, N) != 1:
            # Find the smallest integer coprime to N
            for candidate in range(1, N):
                if gcd(candidate, N) == 1:
                    u = candidate
                    break

        n = tf.range(N, dtype=tf.float32)
        pi = tf.constant(np.pi, dtype=tf.float32)
        if N % 2 == 0:
            # Even N: x(n) = exp(-j*pi*u*n^2/N)
            phase = -pi * u * tf.square(n) / N
        else:
            # Odd N: x(n) = exp(-j*pi*u*n*(n+1)/N)
            phase = -pi * u * n * (n + 1) / N

        # Use tf.complex to create the exponent
        pilot = tf.exp(tf.complex(tf.zeros_like(phase), phase))

        # Normalize to unit average power
        power = tf.reduce_mean(tf.abs(pilot) ** 2)
        pilot = pilot / tf.cast(tf.sqrt(power), tf.complex64)
        return pilot

    @staticmethod
    def addCyclicPrefix(sequence, cyclic_prefix_length):
        """Prepend the last cyclic_prefix_length symbols of sequence."""
        return CodedSystemBlockFading.addCyclicPrefixAndSuffix(
            sequence, cyclic_prefix_length, 0
        )

    @staticmethod
    def addCyclicPrefixAndSuffix(sequence, cyclic_prefix_length, cyclic_suffix_length):
        """Add cyclic extension symbols before and after the ZC core."""
        cyclic_prefix_length = int(cyclic_prefix_length)
        cyclic_suffix_length = int(cyclic_suffix_length)
        parts = []
        if cyclic_prefix_length > 0:
            parts.append(sequence[-cyclic_prefix_length:])
        parts.append(sequence)
        if cyclic_suffix_length > 0:
            parts.append(sequence[:cyclic_suffix_length])
        return tf.concat(parts, axis=0)

    def _preamble_detection_and_payload_extraction(
        self,
        y_noisy,
        zc_seq,
        preamble_length,
        payload_length,
        batch_size,
        taps,
        no,
        known_timing_offset=None,
    ):
        """
        Two-stage timing + effective channel estimation.

        Stage 1 performs a coarse search with the current ZC-core correlation
        metric. Stage 2 refines only around that coarse estimate: it correlates
        the received signal with the full cyclic-prefix/core/suffix preamble,
        sums the correlation magnitude over a tap-length window, and chooses the
        best local packet start. The channel taps are then estimated with the
        ZC-core filter only:

            h_hat[q] = corr_h[k_hat + q] / E_zc.

        When data interference is enabled, the preamble reconstruction residual
        is evaluated on the ZC-core received window. The overlapping preamble
        tail is removed later in BCJR.

        The existing onset correction and channel truncation stages are kept so
        the downstream BCJR interface remains unchanged.

        Args:
            y_noisy: Received noisy signal [batch_size, signal_length]
            zc_seq: CP + Zadoff-Chu + CS preamble
                [batch_size, preamble_length]
            preamble_length: Length of preamble
            payload_length: Length of payload
            batch_size: Batch size, kept for interface compatibility
            taps: Number of effective channel taps to estimate
            known_timing_offset: Optional true timing offset [batch_size]. When
                provided, synchronization uses this offset directly while the
                matched-filter channel estimator remains unchanged.

        Returns:
            y_preamble: Extracted preamble [batch_size, preamble_length]
            y_message: Extracted message
                [batch_size, preamble_length+guard_length+payload_length+taps-1]
            h_hat: Truncated channel impulse response for BCJR
                [batch_size, num_equalization_taps]
            payload_shift: Starting tap index of the BCJR window inside the
                longer channel estimate [batch_size]
            k_hat_coarse: Coarse timing offset before step-2 refinement
                [batch_size]
            k_hat: Estimated timing offset (preamble start) [batch_size]
            residual_min: Normalized preamble reconstruction residual for the
                selected timing hypothesis [batch_size]
            sync_metric_peak: Combined sync score at the selected timing
                [batch_size]
        """
        y_noisy = tf.cast(y_noisy, tf.complex64)
        zc_seq = tf.cast(zc_seq, tf.complex64)
        zc_core = zc_seq[
            :, self.cyclic_prefix_length : self.cyclic_prefix_length + self.zc_length
        ]
        zc_core_length = self.zc_length
        zc_core_start = tf.cast(self.cyclic_prefix_length, tf.int32)

        # Prefer runtime batch dimension to stay compatible with graph execution.
        batch_size = tf.shape(y_noisy)[0]

        # Total message length (preamble + guard + payload + channel memory).
        total_len = (
            preamble_length + self.pilot_guard_length + payload_length + taps - 1
        )
        y_len = tf.shape(y_noisy)[1]

        # Timing hypothesis search window
        max_start = tf.maximum(0, y_len - total_len)
        max_start = tf.minimum(max_start, tf.cast(self.max_delay, tf.int32))
        num_hyp = max_start + 1  # K

        # Stage 1: coarse timing search. Use only the ZC core so cyclic
        # prefix/suffix aliases do not become correlation templates.
        corr_taps = taps
        obs_len = zc_core_length

        # Pad to make all hypothesis slices valid
        pad_right = max_start + total_len + 1
        y_padded = tf.pad(y_noisy, [[0, 0], [0, pad_right]])

        k_range = tf.range(num_hyp, dtype=tf.int32)[None, :, None]
        zc_energy = tf.reduce_sum(tf.abs(zc_core) ** 2, axis=1)  # [B]

        # Coarse synchronization metric: non-cyclic ZC-core tap-bank correlation
        # plus a small preamble-window energy term.
        preamble_range = tf.range(preamble_length, dtype=tf.int32)[None, None, :]
        idx_preamble_sync = tf.broadcast_to(
            k_range + preamble_range,
            tf.stack([batch_size, num_hyp, preamble_length]),
        )
        y_preamble_sync = tf.gather(y_padded, idx_preamble_sync, batch_dims=1)
        sync_energy = tf.reduce_mean(tf.abs(y_preamble_sync), axis=2)

        q_range = tf.range(corr_taps, dtype=tf.int32)[None, None, :, None]
        m_range_sync = tf.range(zc_core_length, dtype=tf.int32)[None, None, None, :]
        idx_sync_corr = tf.broadcast_to(
            k_range[:, :, None, :] + zc_core_start + q_range + m_range_sync,
            tf.stack([batch_size, num_hyp, corr_taps, zc_core_length]),
        )
        y_sync_corr = tf.gather(y_padded, idx_sync_corr, batch_dims=1)
        r_sync_hyp = tf.reduce_sum(
            y_sync_corr * tf.math.conj(zc_core)[:, None, None, :], axis=3
        )
        corr_power = tf.reduce_sum(tf.abs(r_sync_hyp), axis=2)
        sync_corr = corr_power / (zc_energy[:, None] + 1e-12)
        sync_alpha = tf.cast(self.sync_detection_alpha, tf.float32)
        sync_metric = sync_alpha * sync_corr + sync_energy  # [B, K]

        fallback_idx = tf.argmax(sync_metric, axis=1, output_type=tf.int32)
        # noise_power = tf.reshape(tf.cast(no, sync_metric.dtype), [-1])
        # fixed_onset_threshold = (
        #     tf.cast(self.sync_onset_threshold, sync_metric.dtype) * noise_power[:, None]
        # )
        packet_onset_threshold = 0.60 * tf.reduce_max(
            sync_metric, axis=1, keepdims=True
        )
        onset_threshold = packet_onset_threshold

        onset_candidate = sync_metric >= onset_threshold

        # add another condition to filter out the candidates: the sync_metric should be a local peak in a window of +-num_neighbours samples, where num_neighbours is determined by the expected timing error distribution (e.g., +-5 samples).
        num_neighbours = 3
        centre_idx = tf.range(num_hyp, dtype=tf.int32)[None, :]
        # neighbour_offsets = tf.range(-num_neighbours, 1, dtype=tf.int32)[:, None]
        neighbour_offsets = tf.range(
            -num_neighbours, num_neighbours + 1, dtype=tf.int32
        )[:, None]
        neighbour_idx = tf.clip_by_value(centre_idx + neighbour_offsets, 0, num_hyp - 1)
        local_peak = sync_metric >= tf.reduce_max(
            tf.gather(sync_metric, neighbour_idx, axis=1), axis=1
        )
        onset_candidate = tf.logical_and(onset_candidate, local_peak)

        onset_idx = tf.broadcast_to(
            tf.range(num_hyp, dtype=tf.int32)[None, :], tf.shape(sync_metric)
        )
        masked_onset_idx = tf.where(
            onset_candidate, onset_idx, tf.fill(tf.shape(onset_idx), num_hyp)
        )

        first_onset_idx = tf.reduce_min(masked_onset_idx, axis=1)

        k_hat_coarse = tf.where(
            first_onset_idx == num_hyp, fallback_idx, first_onset_idx
        )
        # k_hat_coarse = fallback_idx

        # Stage 2: refine around the coarse timing.
        cut_l = 7
        cut_l_half = 4
        cutout_base = k_hat_coarse - tf.constant(cut_l_half, dtype=tf.int32)
        cutout_start = tf.clip_by_value(cutout_base, 0, max_start)
        # cutout_len = int(payload_length) + int(corr_taps) + int(preamble_length) + cut_l
        cutout_len = 24

        cutout_range = (
            cutout_start[:, None] + tf.range(cutout_len, dtype=tf.int32)[None, :]
        )
        y_cutout = tf.gather(y_padded, cutout_range, batch_dims=1)

        local_offsets = tf.range(cut_l + 1, dtype=tf.int32)
        step2_offsets = tf.clip_by_value(
            cutout_base[:, None] + local_offsets[None, :], 0, max_start
        )
        local_starts = step2_offsets - cutout_start[:, None]

        q_range_step2 = tf.range(corr_taps, dtype=tf.int32)[None, None, :, None]
        preamble_range_step2 = tf.range(preamble_length, dtype=tf.int32)[
            None, None, None, :
        ]
        idx_step2_seq = (
            local_starts[:, :, None, None] + q_range_step2 + preamble_range_step2
        )
        y_step2_seq = tf.gather(y_cutout, idx_step2_seq, batch_dims=1)
        corr_step2 = tf.reduce_sum(
            y_step2_seq * tf.math.conj(zc_seq)[:, None, None, :], axis=3
        )
        step2_metric = tf.reduce_sum(tf.abs(corr_step2), axis=2)
        best_step2 = tf.argmax(step2_metric, axis=1, output_type=tf.int32)
        k_hat_raw = tf.gather(step2_offsets, best_step2[:, None], batch_dims=1)
        k_hat_raw = tf.squeeze(k_hat_raw, axis=1)

        if known_timing_offset is not None:
            known_timing_offset = tf.cast(known_timing_offset, tf.int32)
            k_hat_raw = tf.clip_by_value(known_timing_offset, 0, max_start)

        # Channel estimation: equivalent to correlating with a preamble-length
        # pilot filter that is zero on the cyclic prefix/suffix and equal to the
        # ZC core in the middle.
        q_range_h = tf.range(corr_taps, dtype=tf.int32)[None, :, None]
        m_range_h = tf.range(zc_core_length, dtype=tf.int32)[None, None, :]
        idx_h = k_hat_raw[:, None, None] + q_range_h + zc_core_start + m_range_h
        y_h = tf.gather(y_padded, idx_h, batch_dims=1)
        h_hat_raw = tf.reduce_sum(y_h * tf.math.conj(zc_core)[:, None, :], axis=2)
        h_hat_raw = h_hat_raw / tf.cast(zc_energy[:, None] + 1e-12, tf.complex64)

        tap_range = tf.range(corr_taps, dtype=tf.int32)[:, None]
        core_range = tf.range(zc_core_length, dtype=tf.int32)[None, :]
        cyclic_shift_idx = tf.math.floormod(core_range - tap_range, zc_core_length)
        zc_cyclic_shifts = tf.gather(zc_core, cyclic_shift_idx, axis=1)
        X = tf.transpose(zc_cyclic_shifts, [0, 2, 1])  # [B, obs_len, taps]

        idx_pre = (
            k_hat_raw[:, None]
            + zc_core_start
            + tf.range(obs_len, dtype=tf.int32)[None, :]
        )
        y_preamble_obs = tf.gather(y_padded, idx_pre, batch_dims=1)

        y_preamble_hat = tf.matmul(X, h_hat_raw[:, :, None])
        y_preamble_hat = tf.squeeze(y_preamble_hat, axis=-1)
        preamble_diff = y_preamble_obs - y_preamble_hat
        residual_min = tf.reduce_sum(
            tf.math.real(preamble_diff * tf.math.conj(preamble_diff)), axis=1
        ) / (tf.reduce_sum(tf.abs(y_preamble_obs) ** 2, axis=1) + 1e-12)

        sync_metric_peak = tf.gather(step2_metric, best_step2[:, None], batch_dims=1)
        sync_metric_peak = tf.squeeze(sync_metric_peak, axis=1)
        h_eq, payload_shift = self._truncate_channel_for_equalization(h_hat_raw)

        # Extract message window for each batch using selected k_hat
        idx_msg = k_hat_raw[:, None] + tf.range(total_len, dtype=tf.int32)[None, :]
        y_message = tf.gather(y_padded, idx_msg, batch_dims=1)
        y_preamble = y_message[:, :preamble_length]

        return (
            y_preamble,
            y_message,
            h_eq,
            payload_shift,
            k_hat_coarse,
            k_hat_raw,
            residual_min,
            sync_metric_peak,
        )

    def _truncate_channel_for_equalization(self, h_hat):
        """Keep the strongest contiguous tap window for BCJR."""
        batch_size = tf.shape(h_hat)[0]
        if self.num_estimation_taps == self.num_equalization_taps:
            return h_hat, tf.zeros([batch_size], dtype=tf.int32)

        tap_power = tf.abs(h_hat) ** 2  # [B, L_est]
        if self.num_equalization_taps == 1:
            window_power = tap_power
        else:
            cumsum = tf.cumsum(tap_power, axis=1)
            window_power = cumsum[:, self.num_equalization_taps - 1 :]
            prefix = tf.pad(cumsum[:, : -self.num_equalization_taps], [[0, 0], [1, 0]])
            window_power = window_power - prefix
        payload_shift = tf.argmax(window_power, axis=1, output_type=tf.int32)
        tap_idx = (
            payload_shift[:, None]
            + tf.range(self.num_equalization_taps, dtype=tf.int32)[None, :]
        )
        h_eq = tf.gather(h_hat, tap_idx, batch_dims=1)
        return h_eq, payload_shift

    def _build_constellation_table(self):
        m = self.num_bits_per_symbol
        bits = list(itertools.product([0, 1], repeat=m))
        bits_tensor = tf.constant(bits, dtype=tf.int32)
        symbols = self.mapper(bits_tensor)
        symbols = tf.squeeze(symbols, axis=-1)
        self._constellation_points = tf.cast(symbols, tf.complex64)
        self._constellation_bits = tf.cast(bits_tensor, tf.int32)
        self._constellation_bit_signs = tf.cast(
            2 * self._constellation_bits - 1, tf.float32
        )

        bit0_indices = []
        bit1_indices = []
        for b in range(m):
            idx0 = [i for i, bb in enumerate(bits) if bb[b] == 0]
            idx1 = [i for i, bb in enumerate(bits) if bb[b] == 1]
            bit0_indices.append(tf.constant(idx0, dtype=tf.int32))
            bit1_indices.append(tf.constant(idx1, dtype=tf.int32))

        self._bit0_indices = bit0_indices
        self._bit1_indices = bit1_indices

    def _build_bcjr_trellis(self):
        M = int(self._constellation_points.shape[0])
        L = int(self.num_equalization_taps)

        if L == 1:
            state_count = 1
            next_state = np.zeros((1, M), dtype=np.int32)
            x_vec_indices = np.zeros((1, M, 1), dtype=np.int32)
            for a in range(M):
                x_vec_indices[0, a, 0] = a
        else:
            states = list(itertools.product(range(M), repeat=L - 1))
            state_count = len(states)
            state_index = {tuple(s): i for i, s in enumerate(states)}

            next_state = np.zeros((state_count, M), dtype=np.int32)
            x_vec_indices = np.zeros((state_count, M, L), dtype=np.int32)

            for s_idx, s in enumerate(states):
                for a in range(M):
                    next_state_tuple = (a,) + tuple(s[:-1])
                    next_state[s_idx, a] = state_index[next_state_tuple]
                    x_vec_indices[s_idx, a] = np.array((a,) + tuple(s))

        self._bcjr_state_count = state_count
        self._bcjr_next_state = tf.constant(next_state, dtype=tf.int32)
        self._bcjr_x_vec = tf.gather(
            self._constellation_points, tf.constant(x_vec_indices, dtype=tf.int32)
        )

    @staticmethod
    def _segment_logsumexp(values, segment_ids, num_segments):
        max_val = tf.math.unsorted_segment_max(values, segment_ids, num_segments)
        max_val_gather = tf.gather(max_val, segment_ids)
        sum_exp = tf.math.unsorted_segment_sum(
            tf.exp(values - max_val_gather), segment_ids, num_segments
        )
        return max_val + tf.math.log(sum_exp + 1e-12)

    @staticmethod
    def _segment_max(values, segment_ids, num_segments):
        return tf.math.unsorted_segment_max(values, segment_ids, num_segments)

    def bcjr_equalization(
        self, y_message, h_hat, no, zc_seq, payload_shift=0, llr_a_priori=None
    ):
        """
        BCJR equalization for ISI channel with perfect CSI and synchronization.

        Args:
            y_message: Received signal
                [batch_size, preamble_length + guard_length + payload_length + num_estimation_taps - 1]
            h_hat: Truncated channel impulse response [batch_size, num_equalization_taps]
            no: Noise power (scalar or tensor)
            zc_seq: Known preamble [batch_size, preamble_length]
            payload_shift: Starting tap index of the BCJR window inside the
                longer LMMSE estimate
            llr_a_priori: A priori LLRs from decoder feedback
                [batch_size, payload_length, num_bits_per_symbol]

        Returns:
            llr: LLRs for payload bits [batch_size, payload_length, num_bits_per_symbol]
        """

        h_hat = tf.cast(h_hat, tf.complex64)
        no = tf.cast(no, tf.float32)
        channel_taps = h_hat.shape[-1]
        if channel_taps is None:
            channel_taps = self.num_equalization_taps
        batch_size = tf.shape(y_message)[0]
        if len(no.shape) == 0:
            no = tf.fill([batch_size], no)
        payload_shift = tf.cast(payload_shift, tf.int32)
        if len(payload_shift.shape) == 0:
            payload_shift = tf.fill([batch_size], payload_shift)

        # Extract payload region. When pilot/data interference is enabled, the
        # estimator above used only the clean preamble window, so the first
        # L-1 payload samples still contain the preamble tail and must be
        # corrected before BCJR branch metrics are formed.
        payload_span = self.payload_length + channel_taps - 1
        payload_start = self.preamble_length + self.pilot_guard_length + payload_shift
        payload_idx = (
            payload_start[:, None] + tf.range(payload_span, dtype=tf.int32)[None, :]
        )
        y_payload = tf.gather(y_message, payload_idx, batch_dims=1)

        if self.data_interference and channel_taps > 1:
            y_preamble_full = complex_convolution_full_tf(zc_seq, h_hat)
            y_preamble_tail = y_preamble_full[:, self.preamble_length :]
            y_payload = tf.concat(
                [
                    y_payload[:, : channel_taps - 1] - y_preamble_tail,
                    y_payload[:, channel_taps - 1 :],
                ],
                axis=1,
            )

        # Use payload samples for branch metrics; tail samples initialize beta
        y_obs = y_payload[:, : self.payload_length]
        y_tail = None
        if channel_taps > 1:
            y_tail = y_payload[
                :, self.payload_length : self.payload_length + channel_taps - 1
            ]

        # Precompute expected output for all transitions
        y_hat = tf.einsum("bl,sml->bsm", h_hat, self._bcjr_x_vec)

        S = self._bcjr_state_count

        # Build a priori symbol log-probabilities (up to a constant)
        if llr_a_priori is None:
            llr_a_priori = tf.zeros(
                [batch_size, self.payload_length, self.num_bits_per_symbol],
                dtype=tf.float32,
            )
        symbol_prior = 0.5 * tf.einsum(
            "btm,am->bta", llr_a_priori, self._constellation_bit_signs
        )

        # Initialize alpha (log-domain)
        alpha = tf.zeros([batch_size, S], dtype=tf.float32) - tf.math.log(
            tf.cast(S, tf.float32)
        )

        # Forward recursion
        alphas = [alpha]
        batch_offsets = tf.range(batch_size, dtype=tf.int32)[:, None] * S
        next_state = self._bcjr_next_state

        for t in range(self.payload_length):
            y_t = y_obs[:, t]
            diff = y_t[:, None, None] - y_hat
            metric = -tf.math.real(diff * tf.math.conj(diff)) / no[:, None, None]
            prior_t = symbol_prior[:, t, :][:, None, :]
            gamma = tf.cast(metric, tf.float32) + self.feedback_weight * prior_t

            values = alphas[-1][:, :, None] + gamma
            segment_ids = batch_offsets[:, None, None] + next_state[None, :, :]
            alpha_next_flat = self._segment_max(
                tf.reshape(values, [-1]), tf.reshape(segment_ids, [-1]), batch_size * S
            )
            alpha_next = tf.reshape(alpha_next_flat, [batch_size, S])
            alphas.append(alpha_next)

        # Backward recursion and LLR computation
        if channel_taps > 1:
            # Initialize beta using the tail observations (y_{N}..y_{N+L-2})
            # This uses the known zero-padding beyond the payload.
            state_symbols = self._bcjr_x_vec[
                :, 0, 1:
            ]  # [S, L-1], independent of current symbol
            tail_hats = []
            for k in range(1, channel_taps):
                h_slice = h_hat[:, k:]  # [B, L-k]
                s_slice = state_symbols[:, : channel_taps - k]  # [S, L-k]
                yk = tf.einsum("bl,sl->bs", h_slice, s_slice)
                tail_hats.append(yk)
            y_tail_hat = tf.stack(tail_hats, axis=2)  # [B, S, L-1]
            diff_tail = y_tail[:, None, :] - y_tail_hat
            metric_tail = (
                -tf.math.real(diff_tail * tf.math.conj(diff_tail)) / no[:, None, None]
            )
            beta = tf.reduce_sum(metric_tail, axis=2)
        else:
            beta = tf.zeros([batch_size, S], dtype=tf.float32)
        llr_list = []

        for t in range(self.payload_length - 1, -1, -1):
            y_t = y_obs[:, t]
            diff = y_t[:, None, None] - y_hat
            metric = -tf.math.real(diff * tf.math.conj(diff)) / no[:, None, None]
            prior_t = symbol_prior[:, t, :][:, None, :]
            gamma = tf.cast(metric, tf.float32) + self.feedback_weight * prior_t

            beta_next_gather = tf.gather(beta, next_state, axis=1)
            log_p_x = tf.reduce_max(
                alphas[t][:, :, None] + gamma + beta_next_gather, axis=1
            )

            llr_bits = []
            for b in range(self.num_bits_per_symbol):
                log_p1 = tf.reduce_max(
                    tf.gather(log_p_x, self._bit1_indices[b], axis=1), axis=1
                )
                log_p0 = tf.reduce_max(
                    tf.gather(log_p_x, self._bit0_indices[b], axis=1), axis=1
                )
                llr_bits.append(log_p1 - log_p0)

            llr_t = tf.stack(llr_bits, axis=1)
            llr_list.append(llr_t)

            beta = tf.reduce_max(gamma + beta_next_gather, axis=2)

        llr_list.reverse()
        llr = tf.stack(llr_list, axis=1)
        return llr

    # @tf.function # activate graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        logger.debug(f"Perfect CSI: {self.perfect_csi}")
        logger.debug(f"Perfect Sync: {self.perfect_sync}")
        logger.debug(f"Preamble Length: {self.preamble_length}")
        logger.debug(f"Cyclic Prefix Length: {self.cyclic_prefix_length}")
        logger.debug(f"Cyclic Suffix Length: {self.cyclic_suffix_length}")
        logger.debug(f"Zadoff-Chu Core Length: {self.zc_length}")
        logger.debug(f"Payload Length: {self.payload_length}")

        # TX
        bits = self.binary_source([batch_size, self.k])
        logger.debug(f"Shape of info bits: {bits.shape}")

        codewords = self.encoder(bits)
        logger.debug(f"Shape of codewords (bits): {codewords.shape}")
        codewords_int = tf.gather(codewords, self._interleaver_perm, axis=1)
        x_d = self.mapper(codewords_int)
        logger.debug(
            f"Shape of mapped symbols (bits) - before padding preamble: {x_d.shape}"
        )

        # TX: cyclic-prefix/suffix Zadoff-Chu preamble
        zc_seq = self.get_zc_seq(batch_size)  # shape: [batch_size, preamble_length]
        pilot_guard = tf.zeros(
            [batch_size, self.pilot_guard_length], dtype=zc_seq.dtype
        )

        # Concatenate along the last dimension (symbol dimension)
        x = tf.concat(
            [zc_seq, pilot_guard, x_d], axis=-1
        )  # shape: [batch_size, preamble_length + guard_length + payload_length]
        logger.debug(
            f"Shape of mapped symbols (bits) - after padding preamble: {x.shape}"
        )

        # overall rate
        overall_rate = self.k / (
            (self.preamble_length + self.payload_length) * self.num_bits_per_symbol
        )
        logger.debug(f"Overall rate: {overall_rate}")
        # use overall rate to calculate the noise power
        no = sn.phy.utils.ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=overall_rate,
        )

        # rx_output_length = (
        #     self.preamble_length
        #     + self.payload_length
        #     + self.num_taps
        #     - 1
        #     + self.max_delay
        # )

        if not self.perfect_csi:
            y_convolved, h_true, offset_true = proakis_c_channel_tf(
                x,
                max_offset=self.max_delay,
                taps=self.num_taps,
                is_data_interference=self.data_interference,
                pilot_length=self.preamble_length,
                guard_length=self.pilot_guard_length,
                debug=self.debug,
            )
            if self.debug:
                # assume no noise
                no = 1e-20

            y_noisy = self.awgn_channel(y_convolved, no)
            pulse_group_delay = FRACTIONAL_FILTER_SPAN // 2

            # The channel keeps the full RC-convolution tail by padding before
            # filtering, which also introduces `pulse_group_delay` leading
            # transient samples. Drop only that front transient for timing and
            # equalization; the useful tail remains available at the end.
            y_sync = y_noisy[
                :,
                pulse_group_delay : pulse_group_delay
                + self.preamble_length
                + self.pilot_guard_length
                + self.payload_length
                + self.num_estimation_taps
                - 1
                + self.max_delay,
            ]

            known_timing_offset = offset_true if self.perfect_sync else None

            # Preamble detection/known alignment and payload extraction
            (
                y_preamble,
                y_message,
                h_hat,
                payload_shift,
                k_hat_coarse,
                k_hat_raw,
                residual_min,
                sync_metric_peak,
            ) = self._preamble_detection_and_payload_extraction(
                y_noisy=y_sync,
                zc_seq=zc_seq,
                preamble_length=self.preamble_length,
                payload_length=self.payload_length,
                batch_size=batch_size,
                taps=self.num_estimation_taps,
                no=no,
                known_timing_offset=known_timing_offset,
            )

            self._record_runtime_diagnostics(
                ebno_db=ebno_db,
                residual_min=residual_min,
                timing_error=tf.cast(k_hat_raw - offset_true, tf.float32),
                timing_hit=tf.equal(k_hat_raw, offset_true),
                h_hat=h_hat,
                h_true=h_true,
                payload_shift=payload_shift,
                coarse_timing_error=tf.cast(k_hat_coarse - offset_true, tf.float32),
            )

        # RX: Iterative max-log BCJR equalization and damped min-sum BP decoding
        llr_ext = tf.zeros(
            [batch_size, self.payload_length, self.num_bits_per_symbol],
            dtype=tf.float32,
        )
        llr_bcjr = None
        llr_dec = None

        for _ in range(self.turbo_iterations):
            llr_bcjr = self.bcjr_equalization(
                y_message,
                h_hat,
                no,
                zc_seq,
                payload_shift=payload_shift,
                llr_a_priori=llr_ext,
            )
            logger.debug(f"Shape of LLRs from BCJR: {llr_bcjr.shape}")

            llr_bcjr_bits = tf.reshape(llr_bcjr, [batch_size, self.n])
            llr_bcjr_deint_raw = tf.gather(
                llr_bcjr_bits, self._deinterleaver_perm, axis=1
            )
            llr_bcjr_deint = tf.clip_by_value(
                llr_bcjr_deint_raw,
                -self.llr_max,
                self.llr_max,
            )

            llr_dec = self.decoder_soft(llr_bcjr_deint)
            logger.debug(f"Shape of decoder LLRs: {llr_dec.shape}")

            llr_ext_new_bits = llr_dec - llr_bcjr_deint
            llr_ext_new_int = tf.gather(
                llr_ext_new_bits, self._interleaver_perm, axis=1
            )
            llr_ext_new = tf.reshape(
                llr_ext_new_int,
                [batch_size, self.payload_length, self.num_bits_per_symbol],
            )
            llr_ext = self.bp_damping * llr_ext_new + (1.0 - self.bp_damping) * llr_ext
            llr_ext = tf.clip_by_value(llr_ext, -self.llr_max, self.llr_max)

        llr_bcjr_bits = tf.reshape(llr_bcjr, [batch_size, self.n])
        llr_bcjr_deint = tf.gather(llr_bcjr_bits, self._deinterleaver_perm, axis=1)
        llr_bcjr_deint = tf.clip_by_value(
            llr_bcjr_deint,
            -self.llr_max,
            self.llr_max,
        )
        bits_hat = self.decoder_hard(llr_bcjr_deint)
        logger.debug(f"Shape of decoded bits: {bits_hat.shape}")

        # check which message is error for debugging
        if self.debug:
            for i in range(batch_size):
                if not tf.reduce_all(tf.equal(bits[i], bits_hat[i])):
                    logger.debug(f"Message {i} has errors.")

        return bits, bits_hat


# tell wandb to get started
with wandb.init(
    project="Joint-detection-and-decoding-in-short-packet-communications",
):
    BATCH_SIZE = 500
    modulation = "4qam"  # Choose "4qam", "16qam", or "8psk" (Gray mapping).
    NUM_BITS_PER_SYMBOL = modulation_bits_per_symbol(modulation)
    cyclic_prefix_length = 2
    cyclic_suffix_length = 3
    zc_length = 15  # Zadoff-Chu core symbols before cyclic extensions
    preamble_length = cyclic_prefix_length + zc_length + cyclic_suffix_length
    payload_length = 44  # Length of the payload in symbols
    n = payload_length * NUM_BITS_PER_SYMBOL  # Number of bits in the codeword
    k = 64  # Number of information bits
    Channel_CODERATE = k / n

    # Create a single plot object for both BER and BLER
    ber_bler_plots = sn.phy.utils.PlotBER("BER and BLER for Block Fading Channel")

    model_perfect_csi = CodedSystemBlockFading(
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        modulation=modulation,
        n=n,
        Channel_CODERATE=Channel_CODERATE,
        perfect_sync=False,
        perfect_csi=False,
        data_interference=True,
        preamble_length=preamble_length,
        zc_length=zc_length,
        cyclic_prefix_length=cyclic_prefix_length,
        cyclic_suffix_length=cyclic_suffix_length,
        payload_length=payload_length,
        num_taps=5,
        # BCJR has M**(L-1) states: reduce batch size and/or this tap count
        # for higher modulation orders, especially 16QAM.
        num_effective_taps=6,
        num_estimation_taps=6,
        max_delay=59,  # Maximum delay
        sync_detection_alpha=1.8,
        sync_onset_threshold=35.0,
        debug=False,
    )

    snr_db = np.linspace(12, 13, num=1)  # SNR values in dB for simulation

    model_perfect_csi.reset_runtime_diagnostics()
    ber, bler = sim_ber(
        model_perfect_csi,
        ebno_dbs=snr_db,
        batch_size=BATCH_SIZE,
        num_target_block_errors=100,
        max_mc_iter=50,
    )
    model_perfect_csi.flush_runtime_diagnostics()

    # custom chart
    data_ber = []
    data_bler = []

    for i in range(len(snr_db)):
        data_ber.append([snr_db[i], ber[i]])

    for i in range(len(snr_db)):
        data_bler.append([snr_db[i], bler[i]])

    table_ber = wandb.Table(data=data_ber, columns=["SNR", "BER"])
    table_bler = wandb.Table(data=data_bler, columns=["SNR", "BLER"])

    # Map from the table's columns to the chart's fields
    fields_ber = {"x": "SNR", "y": "BER"}
    fields_bler = {"x": "SNR", "y": "BLER"}

    ber_plot = wandb.plot_table(
        vega_spec_name="BER_BlockFadingChannel",
        data_table=table_ber,
        fields=fields_ber,
    )
    bler_plot = wandb.plot_table(
        vega_spec_name="BLER_BlockFadingChannel",
        data_table=table_bler,
        fields=fields_bler,
    )

    # Log custom tables
    wandb.log(
        {"BER_BlockFadingChannel": ber_plot, "BLER_BlockFadingChannel": bler_plot}
    )

    print("Finished!")
