import unittest

import numpy as np
import tensorflow as tf

from src.channel.pulse_shaping import complex_convolution_with_batch_filters


class ComplexConvolutionWithBatchFiltersTests(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(1234)
        np.random.seed(1234)

    @staticmethod
    def _random_complex(shape):
        return (
            np.random.randn(*shape).astype(np.float32)
            + 1j * np.random.randn(*shape).astype(np.float32)
        )

    @staticmethod
    def _reference_corr1d_batch(x, w, padding):
        """Reference for tf.nn.depthwise_conv2d 1D behavior (cross-correlation)."""
        batch_size, n = x.shape
        _, k = w.shape

        padding = padding.upper()
        if padding == "SAME":
            total_pad = max(k - 1, 0)
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            x_work = np.pad(x, ((0, 0), (pad_left, pad_right)), mode="constant")
            out_len = n
        elif padding == "VALID":
            x_work = x
            out_len = n - k + 1
        else:
            raise ValueError("padding must be 'SAME' or 'VALID'")

        y = np.zeros((batch_size, out_len), dtype=np.complex64)
        for b in range(batch_size):
            for i in range(out_len):
                y[b, i] = np.sum(x_work[b, i : i + k] * w[b])
        return y

    def test_same_padding_matches_reference_per_sample_filters(self):
        x_np = self._random_complex((4, 11))
        w_np = self._random_complex((4, 5))
        y_tf = complex_convolution_with_batch_filters(
            tf.constant(x_np), tf.constant(w_np), padding="SAME"
        ).numpy()
        y_ref = self._reference_corr1d_batch(x_np, w_np, padding="SAME")
        np.testing.assert_allclose(y_tf, y_ref, rtol=1e-4, atol=1e-4)

    def test_valid_padding_matches_reference_per_sample_filters(self):
        x_np = self._random_complex((3, 10))
        w_np = self._random_complex((3, 4))
        y_tf = complex_convolution_with_batch_filters(
            tf.constant(x_np), tf.constant(w_np), padding="VALID"
        ).numpy()
        y_ref = self._reference_corr1d_batch(x_np, w_np, padding="VALID")
        np.testing.assert_allclose(y_tf, y_ref, rtol=1e-4, atol=1e-4)

    def test_single_shared_filter_is_broadcast_to_batch(self):
        x_np = self._random_complex((5, 12))
        w_np = self._random_complex((1, 3))
        y_tf = complex_convolution_with_batch_filters(
            tf.constant(x_np), tf.constant(w_np), padding="SAME"
        ).numpy()
        y_ref = self._reference_corr1d_batch(
            x_np, np.repeat(w_np, repeats=x_np.shape[0], axis=0), padding="SAME"
        )
        np.testing.assert_allclose(y_tf, y_ref, rtol=1e-4, atol=1e-4)

    def test_invalid_padding_raises(self):
        x = tf.zeros([2, 8], dtype=tf.complex64)
        w = tf.zeros([2, 3], dtype=tf.complex64)
        with self.assertRaises(ValueError):
            complex_convolution_with_batch_filters(x, w, padding="FULL")

    def test_mismatched_batch_size_raises(self):
        x = tf.zeros([4, 8], dtype=tf.complex64)
        w = tf.zeros([3, 3], dtype=tf.complex64)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            complex_convolution_with_batch_filters(x, w, padding="SAME")


if __name__ == "__main__":
    unittest.main()
