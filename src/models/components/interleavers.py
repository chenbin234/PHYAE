# Authors: Veeresh Taranalli <veeresht@gmail.com>
# License: BSD 3-Clause

""" Interleavers and De-interleavers """

from numpy import array, arange, zeros
from numpy.random import mtrand
import torch

__all__ = ["RandInterlv"]


class _Interleaver:

    def interlv(self, in_array):
        """Interleave input array using the specific interleaver.

        Parameters
        ----------
        in_array : 1D ndarray of ints
            Input data to be interleaved.

        Returns
        -------
        out_array : 1D ndarray of ints
            Interleaved output data.

        """

        out_array = array(map(lambda x: in_array[x], self.p_array))
        return out_array

    def deinterlv(self, in_array):
        """De-interleave input array using the specific interleaver.

        Parameters
        ----------
        in_array : 1D ndarray of ints
            Input data to be de-interleaved.

        Returns
        -------
        out_array : 1D ndarray of ints
            De-interleaved output data.

        """
        out_array = zeros(len(in_array), in_array.dtype)
        for index, element in enumerate(self.p_array):
            out_array[element] = in_array[index]
        return out_array


class RandInterlv(_Interleaver):
    """Random Interleaver.

    Parameters
    ----------
    length : int
        Length of the interleaver.

    seed : int
        Seed to initialize the random number generator
        which generates the random permutation for
        interleaving.

    Returns
    -------
    random_interleaver : RandInterlv object
        A random interleaver object.

    Note
    ----
    The random number generator is the
    RandomState object from NumPy,
    which uses the Mersenne Twister algorithm.

    """

    def __init__(self, length, seed):
        rand_gen = mtrand.RandomState(seed)

        self.p_array = rand_gen.permutation(arange(length))


class Interleaver(torch.nn.Module):
    def __init__(self, args, p_array):
        super(Interleaver, self).__init__()
        self.args = args
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def set_parray(self, p_array):
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def forward(self, inputs):

        inputs = inputs.permute(1, 0, 2)
        res = inputs[self.p_array]
        res = res.permute(1, 0, 2)

        return res


class DeInterleaver(torch.nn.Module):
    def __init__(self, args, p_array):
        super(DeInterleaver, self).__init__()
        self.args = args

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def set_parray(self, p_array):

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)
        res = inputs[self.reverse_p_array]
        res = res.permute(1, 0, 2)

        return res
