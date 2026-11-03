# This file is to pulse shape the signal.
# The pulse shaping is done by convolving the signal with a pulse shape.
# ! Using a square pulse with normalized energy.


import torch


def pulse_shaping(x, tp, N_up, device):
    """
    Perform pulse shaping (and upsampling) on the input signal.

    Args:
        x (torch.Tensor, complex): Input tensor of shape (batch_size, num_symbols_per_block), in this case, num_symbols = n/nb (since the transmitted signal was splitted).
        tp (float): Pulse duration.
        N_up (int): Upsampling factor.

    Returns:
        torch.Tensor: Pulse-shaped tensor of shape (batch_size, num_symbols * N_up).
    """

    # get the batch size and the number of symbols per block
    batch_size, num_symbols_per_block = x.shape

    # create the square pulse, normalize it and convert it to a complex tensor
    norm = torch.sqrt(torch.tensor(1 / tp, dtype=torch.complex64, device=device))

    # repeat the input tensor N_up times along the second dimension
    x_squared = torch.repeat_interleave(x, N_up, dim=1) * norm

    x_squared = torch.reshape(x_squared, (batch_size, num_symbols_per_block * N_up))

    return x_squared
