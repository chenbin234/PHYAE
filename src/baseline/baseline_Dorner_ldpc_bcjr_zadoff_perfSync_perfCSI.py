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
tf.random.set_seed(42)

# Set the log level to ERROR
logger.remove()  # Remove the default logger
logger.add(sys.stderr, level="ERROR")  # Set to "DEBUG" to see all messages
logger.add("./src/baseline/log_file.log", level="DEBUG", mode="w")


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

    # Sionna uses each point's array index as its binary bit label. Store
    # phase k*pi/4 at index gray(k) for cyclic Gray mapping, including wraparound.
    phase_indices = np.arange(8, dtype=np.int32)
    gray_labels = phase_indices ^ (phase_indices >> 1)
    points = np.empty(8, dtype=np.complex64)
    points[gray_labels] = np.exp(1j * np.pi * phase_indices / 4)
    return sn.phy.mapping.Constellation(
        "custom", num_bits, points=points, normalize=True, center=False
    )


# ---------------------------------------------------
# Proakis channel impulse response
# ---------------------------------------------------
def proakisImpulseResponse_tf(batch_size, taps=5, debug_gap: bool = False):
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

    # Generate real and imaginary parts
    if debug_gap:
        # Keep the first tap at magnitude 100 and all other taps below 1e-4.
        # Random phases and weak-tap magnitudes vary across channel samples.
        phase = tf.random.uniform(
            shape=(batch_size, taps),
            minval=0.0,
            maxval=2.0 * np.pi,
            dtype=tf.float32,
        )
        magnitude = tf.concat(
            [
                tf.fill((batch_size, 1), tf.constant(100.0, dtype=tf.float32)),
                tf.random.uniform(
                    shape=(batch_size, taps - 1),
                    minval=0.0,
                    maxval=1e-5,
                    dtype=tf.float32,
                ),
            ],
            axis=1,
        )
        h_real = magnitude * tf.cos(phase)
        h_imag = magnitude * tf.sin(phase)
    else:
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
    x, output_length, taps, perfect_sync=True, debug_gap: bool = False
):
    """
    x: shape (batch, n), complex
    """

    batch = tf.shape(x)[0]

    # 1. Channel impulse response
    h = proakisImpulseResponse_tf(
        batch_size=batch, taps=taps, debug_gap=debug_gap
    )  # shape [batch, h_len], complex

    h_len = tf.shape(h)[1]

    max_offset = (
        tf.maximum(output_length, tf.shape(x)[1] + h_len - 1)
        - tf.shape(x)[1]
        - h_len
        + 1
    )

    # 5. Integer timing offset (random)
    if perfect_sync:
        offset = tf.zeros(shape=(batch,), dtype=tf.int32)
    else:
        offset = tf.random.uniform(
            shape=(batch,), minval=0, maxval=max_offset + 1, dtype=tf.int32
        )

    offset_one_hot = tf.one_hot(offset, depth=max_offset + 1, dtype=tf.float32)

    # pad with 2 on both sides (same as PyTorch F.pad((2,2)))
    if taps == 5:
        offset_one_hot_padded = tf.pad(offset_one_hot, paddings=[[0, 0], [2, 2]])
    elif taps == 3:
        offset_one_hot_padded = tf.pad(offset_one_hot, paddings=[[0, 0], [1, 1]])
    elif taps == 1:
        offset_one_hot_padded = offset_one_hot  # No padding needed for AWGN channel
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

    return y, h, offset


class ClippedPolarBPDecoder(sn.phy.fec.polar.PolarBPDecoder):
    """Polar BP with bounded messages and optional codeword LLR outputs."""

    def __init__(
        self,
        frozen_pos,
        n,
        num_iter=20,
        hard_out=True,
        return_infobits=True,
        llr_max=20.0,
    ):
        super().__init__(frozen_pos, n, num_iter=num_iter, hard_out=hard_out)
        # Sionna 1.2.2 has no llr_max constructor argument for Polar BP.
        self._llr_max = float(llr_max)
        self._bp_iterations = num_iter
        self.return_infobits = return_infobits

        frozen_prior = np.zeros((1, self.n), dtype=np.float32)
        frozen_prior[:, self.frozen_pos] = self.llr_max
        self._frozen_prior = tf.constant(frozen_prior, dtype=self.rdtype)

        self._stage_indices = []
        for stage in range(self._n_stages):
            indices = np.arange(self.n // 2)
            ind_1 = 2 * indices - indices % (2**stage)
            ind_2 = ind_1 + 2**stage
            ind_inv = np.argsort(np.concatenate([ind_1, ind_2]))
            self._stage_indices.append(
                tuple(
                    tf.constant(ind, dtype=tf.int32) for ind in (ind_1, ind_2, ind_inv)
                )
            )

    def _boxplus_tf(self, x, y):
        # Stable check-node update, including for large user-selected limits.
        x = tf.clip_by_value(x, -self.llr_max, self.llr_max)
        y = tf.clip_by_value(y, -self.llr_max, self.llr_max)
        llr = tf.sign(x) * tf.sign(y) * tf.minimum(tf.abs(x), tf.abs(y))
        llr += tf.nn.softplus(-tf.abs(x + y)) - tf.nn.softplus(-tf.abs(x - y))
        return tf.clip_by_value(llr, -self.llr_max, self.llr_max)

    def call(self, llr_ch):
        input_shape = tf.shape(llr_ch)
        llr_ch = tf.cast(tf.reshape(llr_ch, [-1, self.n]), self.rdtype)
        # BP uses log(P(0)/P(1)); Sionna inputs and outputs use the opposite sign.
        llr_ch = -tf.clip_by_value(llr_ch, -self.llr_max, self.llr_max)
        batch_size = tf.shape(llr_ch)[0]
        msg_l = [tf.zeros_like(llr_ch) for _ in range(self._n_stages)] + [llr_ch]
        msg_r = [tf.tile(self._frozen_prior, [batch_size, 1])] + [
            tf.zeros_like(llr_ch) for _ in range(self._n_stages)
        ]

        for _ in range(self._bp_iterations):
            # Forward pass: source constraints towards the codeword bits.
            for stage, (ind_1, ind_2, ind_inv) in enumerate(self._stage_indices):
                l1 = tf.gather(msg_l[stage + 1], ind_1, axis=1)
                l2 = tf.gather(msg_l[stage + 1], ind_2, axis=1)
                r1 = tf.gather(msg_r[stage], ind_1, axis=1)
                r2 = tf.gather(msg_r[stage], ind_2, axis=1)
                r_out = tf.concat(
                    [self._boxplus_tf(r1, l2 + r2), self._boxplus_tf(r1, l1) + r2],
                    axis=1,
                )
                msg_r[stage + 1] = tf.clip_by_value(
                    tf.gather(r_out, ind_inv, axis=1), -self.llr_max, self.llr_max
                )

            # Backward pass: channel observations towards the source bits.
            for stage in range(self._n_stages - 1, -1, -1):
                ind_1, ind_2, ind_inv = self._stage_indices[stage]
                l1 = tf.gather(msg_l[stage + 1], ind_1, axis=1)
                l2 = tf.gather(msg_l[stage + 1], ind_2, axis=1)
                r1 = tf.gather(msg_r[stage], ind_1, axis=1)
                r2 = tf.gather(msg_r[stage], ind_2, axis=1)
                l_out = tf.concat(
                    [self._boxplus_tf(l1, l2 + r2), self._boxplus_tf(r1, l1) + l2],
                    axis=1,
                )
                msg_l[stage] = tf.clip_by_value(
                    tf.gather(l_out, ind_inv, axis=1), -self.llr_max, self.llr_max
                )

        if self.return_infobits:
            llr_out = -tf.gather(msg_l[0] + msg_r[0], self.info_pos, axis=1)
            output_length = self.k
        else:
            # Codeword posteriors are needed for extrinsic feedback to BCJR.
            llr_out = -(llr_ch + msg_r[-1])
            output_length = self.n
        llr_out = tf.clip_by_value(llr_out, -self.llr_max, self.llr_max)
        if self.hard_out:
            llr_out = tf.cast(llr_out >= 0.0, self.rdtype)
        output_shape = tf.concat([input_shape[:-1], [output_length]], axis=0)
        return tf.reshape(llr_out, output_shape)


class Polar5GBPDecoder(sn.phy.fec.polar.Polar5GDecoder):
    """5G Polar BP with soft codeword output or hard information-bit output."""

    def __init__(self, encoder, num_iter=20, hard_out=True, llr_max=20.0):
        super().__init__(encoder, dec_type="BP", num_iter=num_iter)
        self._llr_max = float(llr_max)
        self._hard_out = hard_out
        self._polar_dec = ClippedPolarBPDecoder(
            encoder.frozen_pos,
            encoder.n_polar,
            num_iter=num_iter,
            hard_out=hard_out,
            return_infobits=hard_out,
            llr_max=self.llr_max,
        )

        # This 5G encoder map includes subblock and channel interleaving.
        rate_matching_indices = encoder._ind_rate_matching
        self._rate_matching_indices = tf.constant(rate_matching_indices, dtype=tf.int32)
        counts = np.bincount(rate_matching_indices, minlength=encoder.n_polar)
        self._untransmitted_positions = tf.constant(counts == 0, dtype=tf.bool)
        self._use_shortening = (
            encoder.n_target < encoder.n_polar
            and encoder.k_polar / encoder.n_target > 7 / 16
        )

    def call(self, llr_ch):
        llr_ch = tf.clip_by_value(
            tf.cast(llr_ch, self.rdtype), -self.llr_max, self.llr_max
        )
        if self._hard_out:
            # Keep the standard 5G rate recovery, input deinterleaving, and CRC removal.
            return super().call(llr_ch)

        input_shape = tf.shape(llr_ch)
        llr_ch = tf.reshape(llr_ch, [-1, self.n_target])
        # Combine repeated observations and leave punctured positions at zero.
        llr_mother = tf.transpose(
            tf.math.unsorted_segment_sum(
                tf.transpose(llr_ch), self._rate_matching_indices, self.n_polar
            )
        )
        if self._use_shortening:
            llr_mother = tf.where(
                self._untransmitted_positions[None, :], -self.llr_max, llr_mother
            )
        llr_mother = tf.clip_by_value(llr_mother, -self.llr_max, self.llr_max)
        llr_dec = self._polar_dec(llr_mother)
        llr_dec = tf.gather(llr_dec, self._rate_matching_indices, axis=1)
        return tf.reshape(llr_dec, input_shape)


class CodedSystemBlockFading(Model):  # Inherits from Keras Model
    def __init__(
        self,
        num_bits_per_symbol,
        n,
        Channel_CODERATE,
        perfect_sync=True,
        perfect_csi=True,
        preamble_length=20,
        payload_length=44,
        num_taps=1,
        max_delay=12,  # Maximum delay
        bp_iterations=20,
        turbo_iterations=10,
        bp_damping=0.7,
        feedback_weight=0.1,
        modulation=None,
        debug_gap: bool = False,
        code_picked="ldpc",
        llr_max=20.0,
    ):
        super().__init__()  # Must call the Keras model initializer

        if code_picked not in ("polar_scl8", "ldpc", "polar_sc", "polar_bp"):
            raise ValueError(
                "code_picked must be 'polar_scl8', 'ldpc', 'polar_sc', or 'polar_bp'."
            )

        # Preserve the existing QAM default for callers that only pass the bit width.
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
        self.preamble_length = preamble_length  # length of Zadoff-Chu preamble symbols
        self.payload_length = payload_length  # length of payload symbols
        self.num_taps = num_taps
        self.max_delay = max_delay
        self.bp_iterations = bp_iterations
        self.turbo_iterations = turbo_iterations
        self.bp_damping = bp_damping
        self.feedback_weight = feedback_weight
        self.debug_gap = debug_gap
        self.code_picked = code_picked
        self.llr_max = float(llr_max)
        if not np.isfinite(self.llr_max) or self.llr_max <= 0.0:
            raise ValueError("llr_max must be finite and positive.")
        self.constellation = create_constellation(self.modulation)
        self.awgn_channel = sn.phy.channel.AWGN()
        self.binary_source = sn.phy.mapping.BinarySource()
        if self.code_picked == "ldpc":
            self.encoder = sn.phy.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        else:
            self.encoder = sn.phy.fec.polar.Polar5GEncoder(self.k, self.n)
        self.mapper = sn.phy.mapping.Mapper(constellation=self.constellation)

        self.demapper = sn.phy.mapping.Demapper("app", constellation=self.constellation)
        # Initialize the channel decoders
        if self.code_picked == "ldpc":
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
        elif self.code_picked == "polar_bp":
            self.decoder_soft = Polar5GBPDecoder(
                self.encoder,
                hard_out=False,
                num_iter=self.bp_iterations,
                llr_max=self.llr_max,
            )
            self.decoder_hard = Polar5GBPDecoder(
                self.encoder,
                hard_out=True,
                num_iter=self.bp_iterations,
                llr_max=self.llr_max,
            )
        elif self.code_picked == "polar_scl8":
            self.decoder_hard = sn.phy.fec.polar.Polar5GDecoder(
                self.encoder, dec_type="SCL", list_size=8
            )
        else:
            self.decoder_hard = sn.phy.fec.polar.Polar5GDecoder(
                self.encoder, dec_type="SC"
            )

        # Generate Zadoff-Chu sequence once during initialization
        self._zc_seq_base = self.generateZadoffChuPilot(
            self.preamble_length, u=1
        )  # shape: [preamble_length]

        # Fixed bit interleaver (turbo loop)
        rng = np.random.RandomState(3)
        perm = rng.permutation(self.n)
        inv_perm = np.argsort(perm)
        self._interleaver_perm = tf.constant(perm, dtype=tf.int32)
        self._deinterleaver_perm = tf.constant(inv_perm, dtype=tf.int32)

        # Precompute constellation and trellis for BCJR
        self._build_constellation_table()
        self._build_bcjr_trellis()

    def get_zc_seq(self, batch_size):
        """
        Get the Zadoff-Chu sequence tiled for the given batch size.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            zc_seq: Zadoff-Chu sequence [batch_size, preamble_length]
        """
        zc_seq = tf.expand_dims(
            self._zc_seq_base, axis=0
        )  # shape: [1, preamble_length]
        zc_seq = tf.tile(
            zc_seq, [batch_size, 1]
        )  # shape: [batch_size, preamble_length]
        return zc_seq

    @staticmethod
    def generateZadoffChuPilot(N, u=1):
        # Ensure u is coprime to N
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        if gcd(u, N) != 1:
            # Find the smallest integer coprime to N
            for candidate in range(2, N):
                if gcd(candidate, N) == 1:
                    u = candidate
                    break

        n = tf.range(N, dtype=tf.float32)
        pi = tf.constant(np.pi, dtype=tf.float32)
        if N % 2 == 0:
            # Even N: x(n) = exp(-j*pi*u*n*(n+1)/N)
            phase = -pi * u * n * (n + 1) / N
        else:
            # Odd N: x(n) = exp(-j*pi*u*n^2/N)
            phase = -pi * u * tf.square(n) / N

        # Use tf.complex to create the exponent
        pilot = tf.exp(tf.complex(tf.zeros_like(phase), phase))

        # Normalize to unit average power
        power = tf.reduce_mean(tf.abs(pilot) ** 2)
        pilot = pilot / tf.cast(tf.sqrt(power), tf.complex64)
        return pilot

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
        L = int(self.num_taps)

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

    def bcjr_equalization(self, y_message, h_hat, no, zc_seq, llr_a_priori=None):
        """
        BCJR equalization for ISI channel with perfect CSI and synchronization.

        Args:
            y_message: Received signal [batch_size, preamble_length + payload_length + num_taps - 1]
            h_hat: Channel impulse response [batch_size, num_taps]
            no: Noise power (scalar or tensor)
            zc_seq: Known preamble [batch_size, preamble_length]
            llr_a_priori: A priori LLRs from decoder feedback
                [batch_size, payload_length, num_bits_per_symbol]

        Returns:
            llr: LLRs for payload bits [batch_size, payload_length, num_bits_per_symbol]
        """

        h_hat = tf.cast(h_hat, tf.complex64)
        no = tf.cast(no, tf.float32)

        batch_size = tf.shape(y_message)[0]
        if len(no.shape) == 0:
            no = tf.fill([batch_size], no)

        # Extract payload region and subtract preamble contribution in the first L-1 samples
        y_payload = y_message[
            :,
            self.preamble_length : self.preamble_length
            + self.payload_length
            + self.num_taps
            - 1,
        ]
        y_preamble_full = complex_convolution_full_tf(zc_seq, h_hat)

        if self.num_taps > 1:
            y_preamble_tail = y_preamble_full[:, self.preamble_length :]
            y_payload = tf.concat(
                [
                    y_payload[:, : self.num_taps - 1] - y_preamble_tail,
                    y_payload[:, self.num_taps - 1 :],
                ],
                axis=1,
            )

        # Use payload samples for branch metrics; tail samples initialize beta
        y_obs = y_payload[:, : self.payload_length]
        y_tail = None
        if self.num_taps > 1:
            y_tail = y_payload[
                :, self.payload_length : self.payload_length + self.num_taps - 1
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
        if self.num_taps > 1:
            # Initialize beta using the tail observations (y_{N}..y_{N+L-2})
            # This uses the known zero-padding beyond the payload.
            state_symbols = self._bcjr_x_vec[
                :, 0, 1:
            ]  # [S, L-1], independent of current symbol
            tail_hats = []
            for k in range(1, self.num_taps):
                h_slice = h_hat[:, k:]  # [B, L-k]
                s_slice = state_symbols[:, : self.num_taps - k]  # [S, L-k]
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
        logger.debug(f"Payload Length: {self.payload_length}")

        if not (self.perfect_sync and self.perfect_csi):
            raise ValueError(
                "This BCJR baseline assumes perfect synchronization and CSI."
            )

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

        # TX: Zadoff-Chu sequence
        zc_seq = self.get_zc_seq(batch_size)  # shape: [batch_size, preamble_length]

        # Concatenate along the last dimension (symbol dimension)
        x = tf.concat(
            [zc_seq, x_d], axis=-1
        )  # shape: [batch_size, preamble_length + payload_length]
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

        # Perfect sync and CSI channel
        if self.num_taps == 5:
            y_convolved, h_true, _ = proakis_c_channel_tf(
                x,
                output_length=80,
                taps=5,
                perfect_sync=True,
                debug_gap=self.debug_gap,
            )
        elif self.num_taps == 3:
            y_convolved, h_true, _ = proakis_c_channel_tf(
                x,
                output_length=78,
                taps=3,
                perfect_sync=True,
                debug_gap=self.debug_gap,
            )
        elif self.num_taps == 1:
            y_convolved, h_true, _ = proakis_c_channel_tf(
                x,
                output_length=76,
                taps=1,
                perfect_sync=True,
                debug_gap=self.debug_gap,
            )
        else:
            raise ValueError("Unsupported number of taps. Must be 1, 3, or 5.")
        logger.debug(f"Shape of received signal (y_noisy): {y_convolved.shape}")
        y_noisy = self.awgn_channel(y_convolved, no)

        y_message = y_noisy[
            :, : self.preamble_length + self.payload_length + self.num_taps - 1
        ]
        assert y_message.shape[1] == (
            self.payload_length + self.preamble_length + self.num_taps - 1
        ), logger.error("Received message has incorrect shape.")

        h_hat = h_true

        # RX: Max-log BCJR equalization with iterative soft BP feedback
        llr_ext = tf.zeros(
            [batch_size, self.payload_length, self.num_bits_per_symbol],
            dtype=tf.float32,
        )
        llr_bcjr = None
        llr_dec = None

        turbo_iterations = (
            self.turbo_iterations if self.code_picked in ("ldpc", "polar_bp") else 1
        )
        for _ in range(turbo_iterations):
            llr_bcjr = self.bcjr_equalization(
                y_message, h_hat, no, zc_seq, llr_a_priori=llr_ext
            )
            logger.debug(f"Shape of LLRs from BCJR: {llr_bcjr.shape}")

            llr_bcjr_bits = tf.reshape(llr_bcjr, [batch_size, self.n])
            llr_bcjr_deint = tf.gather(llr_bcjr_bits, self._deinterleaver_perm, axis=1)
            llr_bcjr_deint = tf.clip_by_value(
                llr_bcjr_deint, -self.llr_max, self.llr_max
            )

            # SC/SCL return hard information bits; BP feeds back soft codeword LLRs.
            if self.code_picked in ("ldpc", "polar_bp"):
                llr_dec = self.decoder_soft(llr_bcjr_deint)
                llr_dec = tf.clip_by_value(llr_dec, -self.llr_max, self.llr_max)
                logger.debug(f"Shape of decoder LLRs: {llr_dec.shape}")

                llr_ext_new_bits = llr_dec - llr_bcjr_deint
                llr_ext_new_int = tf.gather(
                    llr_ext_new_bits, self._interleaver_perm, axis=1
                )
                llr_ext_new = tf.reshape(
                    llr_ext_new_int,
                    [batch_size, self.payload_length, self.num_bits_per_symbol],
                )
                llr_ext = (
                    self.bp_damping * llr_ext_new + (1.0 - self.bp_damping) * llr_ext
                )
                llr_ext = tf.clip_by_value(llr_ext, -self.llr_max, self.llr_max)

        llr_bcjr_bits = tf.reshape(llr_bcjr, [batch_size, self.n])
        llr_bcjr_deint = tf.gather(llr_bcjr_bits, self._deinterleaver_perm, axis=1)
        llr_bcjr_deint = tf.clip_by_value(llr_bcjr_deint, -self.llr_max, self.llr_max)
        bits_hat = self.decoder_hard(llr_bcjr_deint)
        logger.debug(f"Shape of decoded bits: {bits_hat.shape}")

        return bits, bits_hat


def main():
    debug_gap: bool = False  # Set True for a strong first tap and deep fading.
    modulation = "4qam"  # Choose "4qam", "16qam", or "8psk" (Gray mapping).
    code_picked = "ldpc"  # Choose "polar_scl8", "ldpc", "polar_sc", or "polar_bp".
    llr_max = 20.0  # Maximum magnitude of decoder and feedback LLRs.
    NUM_BITS_PER_SYMBOL = modulation_bits_per_symbol(modulation)
    # The five-tap BCJR trellis has M**4 states; limit batch memory for larger M.
    BATCH_SIZE = {"4qam": 4000, "8psk": 500, "16qam": 1}[modulation.lower()]
    preamble_length = 20  # Length of the preamble in symbols
    payload_length = 44  # Length of the payload in symbols
    n = payload_length * NUM_BITS_PER_SYMBOL  # Number of bits in the codeword 44*3=132
    k = 64  # Number of information bits
    Channel_CODERATE = k / n

    # Create a single plot object for both BER and BLER
    ber_bler_plots = sn.phy.utils.PlotBER("BER and BLER for Block Fading Channel")

    model_perfect_csi = CodedSystemBlockFading(
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        modulation=modulation,
        code_picked=code_picked,
        llr_max=llr_max,
        n=n,
        Channel_CODERATE=Channel_CODERATE,
        perfect_sync=True,
        perfect_csi=True,
        preamble_length=preamble_length,
        payload_length=payload_length,
        num_taps=5,
        max_delay=12,  # Maximum delay
        debug_gap=debug_gap,
    )

    snr_db = np.linspace(12, 13, num=1)  # SNR values in dB for simulation

    ber, bler = sim_ber(
        model_perfect_csi,
        ebno_dbs=snr_db,
        batch_size=BATCH_SIZE,
        num_target_block_errors=100,
        max_mc_iter=400,
    )

    # tell wandb to get started
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config={"code_picked": code_picked, "llr_max": llr_max},
    ):
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


if __name__ == "__main__":
    main()
