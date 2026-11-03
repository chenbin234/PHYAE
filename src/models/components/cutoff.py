# This function is to cutoff the delay from the received signal.
# input size = (batch_size, 2, nb, n // nb * N_up + delay_max)
# output size = (batch_size, 2, n * N_up)
import torch


def cutoff_2d(x, starting_point_onehot, dmax, ts, N):
    """
    Cutoff the delay from the received signal.

    Parameters:
    - x (torch.Tensor): The received signal of shape (batch_size, nb, M), dtype:tensor.complex64.
    - starting_point_onehot (torch.Tensor): The predicted starting point of shape (batch_size, int(dmax/ts + 1)).
    - signal_length (int): The length of the signal to be extracted.

    Returns:
    - received_signal_cut (torch.Tensor): The output signal of shape (batch_size, nb, signal_length), dtype:tensor.complex64.
    """

    # get the batch size
    batch_size, nb, M = x.shape

    # get the signal length
    signal_length = M - int(dmax / ts) - 1

    # Convert one-hot to integer indices
    starting_indices = torch.argmax(
        starting_point_onehot, dim=1
    )  # Shape: (batch_size,)
    # Create a tensor of offsets [0, 1, 2, ..., L-1]
    offsets = torch.arange(signal_length, device=x.device).unsqueeze(
        0
    )  # Shape: (1, signal_length)
    # Compute the indices to gather
    # starting_indices: (batch_size, 1) + offsets: (1, L) => (batch_size, L)
    indices = starting_indices.unsqueeze(1) + offsets  # Shape: (batch_size, L)

    # Expand indices to match the number of blocks
    # (batch_size, L) -> (batch_size, 1, L) -> (batch_size, number_block, L)
    indices = indices.unsqueeze(1).expand(
        -1, nb, -1
    )  # Shape: (batch_size, number_block, L)

    # Gather the segments from X
    # X has shape (batch_size, number_block, M)
    # indices has shape (batch_size, number_block, L)
    # Gather along the last dimension (dim=2)
    segments = torch.gather(
        x, dim=2, index=indices
    )  # Shape: (batch_size, number_block, L)

    # output_signal = segments[:, :, ::N].view(
    #     batch_size, -1
    # )  # Shape: (batch_size, nb * ns), dtype:tensor.complex64
    output_signal = segments.view(
        batch_size, -1
    )  # Shape: (batch_size, nb * ns), dtype:tensor.complex64

    # convert the output signal to real number tensor, shape: (batch_size, nb * ns, 2), dtype:torch.float32
    output_signal = torch.stack((output_signal.real, output_signal.imag), dim=-1)

    return output_signal


def cutoff_proakis(x, starting_point_onehot, n_M):
    """
    Cutoff the delay from the received signal.

    Parameters:
    - x (torch.Tensor): The received signal of shape (batch_size, 2n, 2), dtype:tensor.float.
    - starting_point_onehot (torch.Tensor): The starting point of shape (batch_size, dmax).
    - signal_length (int): The length of the signal to be extracted.

    Returns:
    - received_signal_cut (torch.Tensor): The output signal of shape (batch_size, nb, signal_length), dtype:tensor.complex64.
    """
    # get the batch size
    batch_size, N, _ = x.shape
    n = int(N / 2)
    # n = int((N + n_M) / 2)  # AWGN+delay

    # Convert one-hot to integer indices
    starting_indices = torch.argmax(
        starting_point_onehot, dim=1
    )  # Shape: (batch_size,)

    # each element in starting_indices minus n_M
    starting_indices = starting_indices - n_M

    #! check if any value in starting_indices is smaller than 0, if so, set the specific value of the element to 0
    starting_indices = torch.where(
        starting_indices < 0,
        torch.zeros_like(starting_indices),
        starting_indices,
    )

    # get the signal length
    # signal_length = n + n_M * 2
    signal_length = n  #! Debug

    # Create a tensor of offsets [0, 1, 2, ..., L-1]
    offsets = torch.arange(signal_length, device=x.device).unsqueeze(
        0
    )  # Shape: (1, signal_length)

    # Compute the indices to gather
    # starting_indices: (batch_size, 1) + offsets: (1, L) => (batch_size, L)
    indices = starting_indices.unsqueeze(1) + offsets  # Shape: (batch_size, L)

    # Gather the segments from X
    # X has shape (batch_size, 2n, 2)
    # indices has shape (batch_size, n+2n_M, 2)
    # Gather along the dimension (dim=1)
    segments = torch.gather(
        x, dim=1, index=indices.unsqueeze(2).expand(-1, -1, 2)
    )  # Shape: (batch_size, n+2n_M, 2)

    return segments
