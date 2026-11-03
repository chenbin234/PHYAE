# This file is to build a Black fadding channel
# The channel is defined as:
#     output_signal = transmitter_signal * h + noise

# The input signal and output signal are both in the form of torch tensor


import torch
from models.components.Pluse_shaping import pulse_shaping


def Block_fading_channel(
    transmitted_signal, tp, N_up, nb, delay, SNR_db, delay_max, device
):
    """
    Function to simulate a block fading channel (include upsampling + pulse shaping).

    Args:
        transmitted_signal (torch.Tensor): The input signal of size (batch_size, 2, n). where n is the number of symbols per information message of length k bits.
        N_up (int): The upsampling factor.
        nb (int): The number of splitted blocks per message. in this case we set it to 4.
        delay (torch.Tensor): The delay of the block fading channel (same for each message of k bits) of size (batch_size, 1).
        SNR_db (float): The signal-to-noise ratio in dB.
        delay_max (int): The maximum delay of the block fading channel.

    Returns:
        received_signal (torch.Tensor): The output signal of shape (batch_size, 2, nb, n // nb * N_up + delay_max).
    """

    # get the batch size and the number of symbols per message
    batch_size, _, n = transmitted_signal.shape

    # split the transmitted signal into nb blocks,
    # each block has n // nb symbols
    # transmitted_signal_blocks is a turple of nb tensors, each tensor has the shape (batch_size, 2, n // nb)
    transmitted_signal_blocks = torch.split(transmitted_signal, n // nb, dim=2)

    # create a list to store the received signals
    received_signal_blocks = []

    # loop over each block
    for block in transmitted_signal_blocks:

        # block is a tensor of shape (batch_size, 2, n // nb), each elemet is a real number
        # conver the block to a complex tensor.
        block_real = block[:, 0, :]
        block_imag = block[:, 1, :]
        block_complex = torch.complex(block_real, block_imag)

        # upsampling & pulse shape the block, the output is a complex tensor of shape (batch_size, n // nb * N_up)
        block_pulse_shaped = pulse_shaping(block_complex, tp, N_up, device)

        # simulate the block fading channel
        # the output is a complex tensor of shape (batch_size, n // nb * N_up + delay_max)
        block_fading_delay_noise = Block_fading(
            block_pulse_shaped, delay, delay_max, SNR_db, device
        )

        # store the received block
        received_signal_blocks.append(block_fading_delay_noise)

    # concatenate the nb received blocks to get the received signal, each has the shape (batch_size, n // nb * N_up + delay_max)
    # the output is a complex tensor of shape (batch_size, nb, n // nb * N_up + delay_max)
    received_signal = torch.stack(received_signal_blocks, dim=1)

    # stack the real and imaginary parts of the received signal
    # the output is a tensor of shape (batch_size, 2, nb, n // nb * N_up + delay_max)
    received_signal_real = received_signal.real
    received_signal_imag = received_signal.imag
    received_signal_stack = torch.stack(
        [received_signal_real, received_signal_imag], dim=1
    )

    return received_signal_stack.to(device)


def Block_fading(x_pulse_shaped, delay, delay_max, SNR_db, device):
    """
    Simulate a block fading channel with the given SNR for a signal x.

    Parameters:
    - x (torch.Tensor): Input tensor of shape (batch_size, num_symbols_per_block * N_up), each element is a complex number.
    - snr_db: Signal-to-Noise Ratio in dB.

    Returns:
    - received_signal: Tensor of the signal after experiencing the block fading channel, it has the same shape with x.
    """

    #! x_length = num_symbols_per_block * N_up + delay_max
    batch_size, x_length = x_pulse_shaped.shape

    #! step 1: normalise the transmitted signal to have unit power
    # Convert SNR from dB to linear scale
    SNR = 10 ** (SNR_db / 10).to(device)

    # Calculate the signal power
    signal_power = calculate_average_power_complex(x_pulse_shaped).to(device)

    #! normalise the transmitted signal to have unit power
    signal_normalized = x_pulse_shaped / torch.sqrt(signal_power.clone().detach())

    #! step 2: add block fading channel gain
    # Generate the block gains which is complex Gaussian with unit variance
    H_l = torch.randn((batch_size, 1), dtype=torch.complex64, device=device)

    # Repeat each block gain for x_length times and align with the signal shape
    H_l = torch.repeat_interleave(H_l, x_length, dim=1)

    # reshape the block gains to the same shape as the signal
    H_l_reshaped = torch.reshape(H_l, (batch_size, -1))

    # Apply the block fading channel to the transmitted signal (element-wise multiplication)
    faded_signal = signal_normalized * H_l_reshaped

    # ! step 3 : add delay
    # ! add delay, we consider a simple case that each sub block of one message experiences the same delay
    # ! and the delay is uniform distributed in [0, delay_max]
    # the input size is (batch_size, n // nb * N_up), the output size is (batch_size, n // nb * N_up + delay_max), each element is a complex number
    block_fading_delayed = add_delay(faded_signal, delay, delay_max, device)

    #! step 4: add noise
    # generate the noise
    noise = torch.randn(block_fading_delayed.shape, dtype=torch.complex64).to(
        device
    ) * torch.sqrt(1 / SNR)

    received_signal = block_fading_delayed + noise

    return received_signal


def calculate_average_power_complex(signal):
    """
    Calculate the average power of a complex signal.

    Parameters:
    signal (torch.Tensor): Input signal of size (batch_size, n)

    Returns:
    torch.Tensor: Average power
    """
    # Compute power of the real part
    real_power = signal.real**2

    # Compute power of the imaginary part
    imag_power = signal.imag**2

    # Compute average power, which is a value
    avg_power = torch.mean(real_power + imag_power)

    return avg_power


def add_delay(block_pulse_shaped, delay, delay_max, device):
    """
    Add delay to the signal.

    Parameters:
    - block_pulse_shaped (torch.Tensor): The input signal of shape (batch_size, n // nb * N_up).
    - delay (torch.Tensor): The delay of the block fading channel (same for each message of k bits) of shape (batch_size, 1).
    - delay_max (int): The maximum delay of the block fading channel.

    Returns:
    - block_pulse_shaped_delayed (torch.Tensor): The output signal of shape (batch_size, n // nb * N_up + delay_max).
    """
    # get the batch size and block length
    batch_size, block_length = block_pulse_shaped.shape

    # create a tensor of zeros with the shape (batch_size, delay_max)
    block_pulse_shaped_delayed = torch.zeros(
        (batch_size, block_length + delay_max), dtype=torch.complex64, device=device
    )

    # loop over each message
    for i in range(batch_size):
        # get the delay of the current message
        d = delay[i, 0]

        # add the delay to the message
        block_pulse_shaped_delayed[i, d : d + block_length] = block_pulse_shaped[i, :]

    return block_pulse_shaped_delayed
