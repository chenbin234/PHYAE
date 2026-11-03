import unittest

import numpy as np
import tensorflow as tf

from src.channel.pulse_shaping import pulse_shaping, rcosdesign


class PulseShapingTests(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(1234)
        np.random.seed(1234)

    @staticmethod
    def _random_complex(shape):
        return tf.complex(
            tf.random.normal(shape, dtype=tf.float32),
            tf.random.normal(shape, dtype=tf.float32),
        )

    def test_output_shape_and_dtype(self):
        batch_size = 3
        n = 16
        sps = 4
        x = self._random_complex([batch_size, n])

        y = pulse_shaping(
            x, beta=0.22, span=16, sps=sps, shape="sqrt", offsets=np.array([0.0])
        )

        self.assertEqual(tuple(y.shape), (batch_size, n * sps))
        self.assertEqual(y.dtype, tf.complex64)

    def test_zero_input_gives_zero_output(self):
        x = tf.zeros([4, 12], dtype=tf.complex64)

        y = pulse_shaping(
            x, beta=0.22, span=16, sps=2, shape="sqrt", offsets=np.array([0.0])
        )

        max_abs = tf.reduce_max(tf.abs(y)).numpy()
        self.assertLess(max_abs, 1e-7)

    def test_linearity(self):
        x1 = self._random_complex([2, 20])
        x2 = self._random_complex([2, 20])
        a = tf.complex(tf.constant(0.7, tf.float32), tf.constant(-0.3, tf.float32))
        b = tf.complex(tf.constant(-0.2, tf.float32), tf.constant(0.5, tf.float32))

        lhs = pulse_shaping(
            a * x1 + b * x2,
            beta=0.22,
            span=16,
            sps=3,
            shape="sqrt",
            offsets=np.array([0.0]),
        )
        rhs = a * pulse_shaping(
            x1, beta=0.22, span=16, sps=3, shape="sqrt", offsets=np.array([0.0])
        ) + b * pulse_shaping(
            x2, beta=0.22, span=16, sps=3, shape="sqrt", offsets=np.array([0.0])
        )

        np.testing.assert_allclose(lhs.numpy(), rhs.numpy(), rtol=1e-4, atol=1e-4)

    def test_identical_inputs_produce_identical_outputs(self):
        row = self._random_complex([1, 18])
        x = tf.tile(row, [5, 1])

        y = pulse_shaping(
            x, beta=0.22, span=16, sps=2, shape="sqrt", offsets=np.array([0.0])
        ).numpy()

        for i in range(1, y.shape[0]):
            np.testing.assert_allclose(y[0], y[i], rtol=0.0, atol=1e-7)

    def test_invalid_sps_raises(self):
        x = self._random_complex([2, 8])

        with self.assertRaises((NameError, ZeroDivisionError)):
            pulse_shaping(
                x, beta=0.22, span=16, sps=0, shape="sqrt", offsets=np.array([0.0])
            )

    def test_rcosdesign_default_length(self):
        h = rcosdesign(0.22, 16, 1, offset=0.3, shape="normal")
        self.assertEqual(tuple(h.shape), (17,))

    def test_rcosdesign_num_taps_even_length(self):
        h = rcosdesign(0.22, 16, 1, offset=0.3, shape="normal", num_taps=16)
        self.assertEqual(tuple(h.shape), (16,))

    def test_rcosdesign_batch_offsets(self):
        offsets = tf.constant([0.0, 0.1, -0.2], dtype=tf.float32)
        h = rcosdesign(0.22, 16, 1, offset=offsets, shape="normal")
        self.assertEqual(tuple(h.shape), (3, 17))

    def test_pulse_shaping_batch_offsets(self):
        batch_size = 4
        n = 12
        sps = 2
        x = self._random_complex([batch_size, n])
        offsets = tf.linspace(-0.3, 0.3, batch_size)

        y = pulse_shaping(
            x, beta=0.22, span=16, sps=sps, shape="normal", offsets=offsets
        )

        self.assertEqual(tuple(y.shape), (batch_size, n * sps))


if __name__ == "__main__":
    unittest.main()
