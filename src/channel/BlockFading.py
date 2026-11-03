import torch
import torch.nn.functional as F


class RayleighBlockFadingChannel:
    def __init__(self, nb, tp, ts, N, dmax, is_delay, h_constant=False):
        self.nb = nb
        self.tp = tp
        self.ts = ts
        self.N = N
        self.dmax = dmax
        self.is_delay = is_delay
        self.h_constant = h_constant

    def forward(self, X, SNR_db_low, SNR_db_high):

        # X shape: (batch_size, n, 2)
        # The third dimension represents real and imaginary parts
        (
            batch_size,
            n,
            _,
        ) = X.shape
        device = X.device
        # Combine real and imaginary parts to form complex tensor
        X_real = X[:, :, 0]  # Shape: (batch_size, n)
        X_imag = X[:, :, 1]  # Shape: (batch_size, n)
        X_complex = X_real.type(torch.complex64) + 1j * X_imag.type(torch.complex64)
        # Now X_complex has shape: (batch_size, n), dtype: torch.complex64

        if self.is_delay:
            # Generate a random delay d for each batch, uniformly between 0 and dmax
            delay_true = torch.rand((batch_size, 1), device=device) * self.dmax
        else:
            # If no delay is present, set delay to zero
            delay_true = torch.zeros((batch_size, 1), device=device)

        # calculate the eps and q values
        eps = delay_true % self.ts  # Shape: (batch_size,)
        q = torch.floor((delay_true + 1e-10) / self.ts).long()  # Shape: (batch_size,)
        true_starting_point_onehot = starting_point_onehot(q, self.dmax, self.ts)

        # Calculate the block length - ns
        ns = n // self.nb
        # if n % self.nb != 0:
        #     # Pad X_complex with zeros to make it divisible by nb
        #     pad_width = self.nb * ns - n
        #     X_complex = F.pad(X_complex, (0, pad_width))
        #     n = X_complex.shape[1]

        # total length of the received sequence (per block)
        M = int(self.dmax / self.ts + (ns * self.tp) / (self.ts) + 1)
        # Initialize an tensor array to store received signals for each block (upsampled)
        received_signal = torch.zeros(
            (batch_size, self.nb, M), dtype=torch.complex64, device=device
        )

        for l in range(self.nb):
            # Extract the block of transmitted signal for all batches
            start_idx = l * ns
            end_idx = (l + 1) * ns
            X_block = X_complex[:, start_idx:end_idx]  # Shape: (batch_size, block_len)

            # Normalize transmitted signal per block for each batch
            energies = torch.mean(torch.abs(X_block) ** 2, dim=1, keepdim=True)
            # X_block_normalized = X_block
            X_block_normalized = (
                X_block / torch.sqrt(energies) * torch.sqrt(torch.tensor(2.0))
            )

            # Upsample the signal by factor N
            upsampled_length = ns * self.N
            X_upsampled = torch.zeros(
                (batch_size, upsampled_length), dtype=torch.complex64, device=device
            )
            indices = torch.arange(0, upsampled_length, self.N, device=device)
            # X_upsampled is a tensor of shape (batch_size, upsampled_length),
            # It repeats the normalized block signal N times
            for j in range(self.N):
                X_upsampled[:, indices + j] = X_block_normalized

            # add zeros at the beginning of the upsampled signal and in the end
            X_N_Q_l = torch.zeros((batch_size, M), dtype=torch.complex64, device=device)
            X_N_Q_l = insert_tensor_in_place(X_N_Q_l, q, X_upsampled, self.N, device)

            X_N_Q_l_plus_1 = torch.zeros(
                (batch_size, M), dtype=torch.complex64, device=device
            )
            X_N_Q_l_plus_1 = insert_tensor_in_place(
                X_N_Q_l_plus_1, q + 1, X_upsampled, self.N, device
            )

            X_interference = X_N_Q_l * (1 - eps / self.ts) + X_N_Q_l_plus_1 * (
                eps / self.ts
            )

            received_signal[:, l, :] = X_interference

        if not self.h_constant:
            # Generate Rayleigh fading coefficients of shape (batch_size, nb, 1)
            h = torch.randn(
                (batch_size, self.nb, 1), dtype=torch.complex64, device=device
            )
            h_expended = h.expand(-1, -1, M)
            # Apply fading, faded_signal of shape (batch_size, self.nb, M)
            faded_signal = h_expended * received_signal
        else:
            # generate a tensor array of shape (batch_size, self.nb, M), each element is a complex number (1+1j)/4
            value = 1
            h_expended = torch.full(
                (batch_size, self.nb, M), value, dtype=torch.complex64, device=device
            )
            # Apply fading, faded_signal of shape (batch_size, self.nb, M)
            faded_signal = h_expended * received_signal

        # Convert SNR from dB to linear scale
        sigma_low = torch.tensor(10 ** (-SNR_db_low * 1.0 / 20)).to(device)
        sigma_high = torch.tensor(10 ** (-SNR_db_high * 1.0 / 20)).to(device)

        # mix the SNR
        this_sigma = (
            torch.rand(faded_signal.shape).to(device) * (sigma_low - sigma_high)
            + sigma_high
        )

        noise_real = this_sigma * torch.randn(
            faded_signal.shape, dtype=torch.float32
        ).to(device)
        noise_imag = this_sigma * torch.randn(
            faded_signal.shape, dtype=torch.float32
        ).to(device)
        noise = noise_real + 1j * noise_imag

        output_signal = faded_signal + noise

        return output_signal, true_starting_point_onehot


def starting_point_onehot(q, dmax, ts):
    """
    Function to generate the one-hot encoded starting point for each batch.

    Args:
        q (torch.Tensor): (batch_size, ).
        dmax (float): The maximum delay.
        ts (float): The sampling period.

    Returns:
        torch.Tensor: The one-hot encoded starting point for each batch.
    """
    batch_size = q.shape[0]
    device = q.device

    # Number of classes must be specified for one-hot encoding
    num_classes = int(dmax / ts) + 1
    # Perform one-hot encoding, one hot tensor of shape (batch_size, num_classes)
    one_hot = F.one_hot(q, num_classes=num_classes)

    return one_hot.squeeze(1).to(device)


def insert_tensor_in_place(X_N_Q_l, q, X_upsampled, N, device):
    """
    Function to insert the upsampled signal into the correct position of the tensor.

    Args:
        X_N_Q_l (torch.Tensor): The tensor to insert the upsampled signal into.
        q (int): The delay value.
        X_upsampled (torch.Tensor): The upsampled signal.

    Returns:
        torch.Tensor: The tensor with the upsampled signal inserted.
    """
    batch_size, insert_length = X_upsampled.size()
    # Create a batch index
    batch_indices = torch.arange(batch_size).unsqueeze(1).to(device)
    # Create column indices using q
    column_indices = q + torch.arange(insert_length).to(device)
    # Insert X into X_N_Q_l using advanced indexing
    X_N_Q_l[batch_indices, column_indices] = X_upsampled * torch.sqrt(
        torch.tensor(1 / N)
    )

    return X_N_Q_l
