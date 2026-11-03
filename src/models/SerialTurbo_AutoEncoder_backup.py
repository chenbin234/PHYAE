# This file contains the implementation of the CNN AutoEncoder model using PyTorch (Xi Zhang's Master Thesis)
# The model consists of an Transmitter and a Receiver
#! Note this is An AE-based Joint Synchronization, Equalization, and Decoding System
#! The transmitter is the same as before, and the receiver was designed to have a iterative architecture


import sys

sys.path.append("src/")

from numpy import arange
from numpy.random import mtrand
import torch
import torch.nn as nn
import torch.nn.functional as F
from channel.Proakis_type_C_channel import proakis_c_channel
from channel.AWGN_Channel import AWGN_Channel_new
from models.components.cutoff import cutoff_proakis
from models.components.STE import STEQuantize
from train.trainer import unit_power_normalization
from models.components.CNN_block import CNN_block1


def add_parity_bit(x):
    """
    Append a parity bit to each sample in the batch.

    The input tensor x must be of shape (batch_size, k, 1) with binary values (0 and 1).
    The parity bit is computed such that the total number of 1s in the new (k+1)-long vector is even.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, k, 1)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, k+1, 1)
    """
    # Reshape x to (batch_size, k) to ease sum computation along each sample.
    batch_size, k, _ = x.shape
    # Squeeze the last dimension if needed for summing
    x_squeezed = x.view(batch_size, k)

    # Compute the parity bit (XOR of all bits = sum % 2)
    # The parity is 1 if the number of ones is odd (#! to make total even),
    # and 0 when it is even.
    parity = torch.remainder(torch.sum(x_squeezed, dim=1), 2)

    # Reshape parity to (batch_size, 1, 1)
    parity = parity.view(batch_size, 1, 1)

    # Concatenate the parity as the last element along dimension=1, resulting in (batch_size, k+1, 1)
    y = torch.cat([x, parity], dim=1)
    return y


def generate_permutation_array(k, seed=0):

    rand_gen = mtrand.RandomState(seed=seed)
    p_array = rand_gen.permutation(arange(k))

    p_array = torch.LongTensor(p_array).view(len(p_array))

    return p_array


class CNN_block(nn.Module):
    def __init__(self, in_channels, k, n, F, padding="same"):
        super(CNN_block, self).__init__()

        # Define the sequence of convolutional layers
        # input size = (batch_size, in_channels, k), output size = (batch_size, n, k)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=100,
                kernel_size=5,
                padding=padding,
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=100, out_channels=100, kernel_size=5, padding=padding
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=100, out_channels=100, kernel_size=5, padding=padding
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=100, out_channels=100, kernel_size=5, padding=padding
            ),
            nn.ELU(),
            nn.Conv1d(
                in_channels=100, out_channels=100, kernel_size=5, padding=padding
            ),
            nn.ELU(),
        )

        # Define the output layer
        # input size = (batch_size, n, k), output size = (batch_size, F, k)
        # self.output_layer = nn.Conv1d(in_channels=100, out_channels=F, kernel_size=1)
        self.output_layer = nn.Linear(in_features=100, out_features=F)

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = self.conv_layers(x)  # output size = (batch_size, n, k)

        # convert the second and third dimension of the tensor
        # x = x.permute(0, 2, 1)  # output size = (batch_size, k, n)

        # Pass the result through the output layer
        x = self.output_layer(x)  # output size = (batch_size, F, k)

        return x


def normalize_signal(signal, eps=1e-8):
    """
    Normalize a batch of complex signals so that each has an average power of 1.
    Args:
        signal (torch.Tensor): A tensor of shape (batch_size, n, 2),
                            where the last dimension contains the real and
                            imaginary parts of the signal.
        eps (float): A small value to avoid division by zero.

    Returns:
        torch.Tensor: The normalized signal with the same shape as the input.
    """
    # Compute the power (squared magnitude) for each sample and each time point:
    # signal[..., 0] is the real part and signal[..., 1] is the imaginary part.
    power = signal[..., 0] ** 2 + signal[..., 1] ** 2

    # Compute the mean power over the signal length (n) for each sample.
    # We keep the dimensions so that the normalization factor can be broadcast easily.
    mean_power = power.mean(dim=1, keepdim=True)

    # Compute the normalization factor (square root of the mean power) for each sample.
    norm_factor = torch.sqrt(mean_power + eps)
    sqrt2 = torch.full(signal.shape, torch.sqrt(torch.tensor(2.0)))
    # Divide the signal by the normalization factor so that, for each sample,
    # the average (over time) power becomes 2.
    normalized_signal = signal / norm_factor

    return normalized_signal


def calculate_average_power(signal, eps=1e-8):
    """
    Calculate the average power of a batch of complex signals.

    Args:
        signal (torch.Tensor): A tensor of shape (batch_size, n, 2),
                               where the last dimension contains the real and
                               imaginary parts of the signal.
        eps (float, optional): A small value to ensure numerical stability
                               when taking square roots. Defaults to 1e-8.

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the average
                      power for each sample in the batch.
    """
    # Compute the power (squared magnitude) for each time point:
    # power = real^2 + imag^2
    power = signal[..., 0] ** 2 + signal[..., 1] ** 2

    # Compute the mean power over the signal length (n) for each sample
    mean_power = power.mean(dim=1)

    return mean_power


class Encoder(nn.Module):
    def __init__(self, in_channels, k, n, F):

        super(Encoder, self).__init__()

        # Outer Encoder
        self.outer_encoder = CNN_block(
            in_channels=1, k=k, n=n, F=F
        )  # output size = (batch_size, F, k)

        # Inner Encoder
        self.inner_encoder = CNN_block(
            in_channels=F, k=n, n=n, F=2
        )  # output size = (batch_size, 2, n)

    def set_interleaver(self, inputs):
        # inputs has the shape (batch_size, k, 1)
        k = inputs.shape[1]

        rand_gen = mtrand.RandomState(seed=0)
        p_array = rand_gen.permutation(arange(k))
        # print(p_array)

        # convert p_array from numpy array to torch tensor
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

        inputs = inputs.permute(1, 0, 2)
        res = inputs[self.p_array]
        res = res.permute(1, 0, 2)

        return res

    # def power_constraint(self, inputs):
    #     """
    #     Apply power constraint to the encoded tensor.

    #     Args:
    #         inputs (torch.Tensor): Encoded tensor after applying the Turbo AutoEncoder, with shape (batch_size, k, 3).

    #     Returns:
    #         torch.Tensor: Encoded tensor after applying power constraint, with shape (batch_size, k, 3).

    #     """

    #     # x has the shape (batch_size, k, 2)
    #     # Apply power constraint to the encoded tensor
    #     x_input_norm = (inputs - torch.mean(inputs)) * 1.0 / torch.std(inputs)

    #     return x_input_norm

    def power_constraint(self, x, eps=1e-20):
        # x has shape (batch_size, k, 2)
        # First, subtract the mean of each batch along the k dimension
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_centered = x_centered - x_centered.mean(
            dim=1, keepdim=True
        )  # Ensure exact zero mean

        # Compute the mean squared magnitude for each batch.
        # For each element, the (squared) magnitude is: real^2 + imag^2.
        # We average those over k elements.
        power = torch.mean(
            x_centered[..., 0] ** 2 + x_centered[..., 1] ** 2, dim=1, keepdim=True
        )
        # Compute the root mean square per batch. The shape is (batch_size, 1).
        scale = torch.sqrt(power + eps)

        # Reshape scale to be broadcastable to x_centered; adding a dimension for k.
        scale = scale.unsqueeze(-1)  # Now shape is (batch_size, 1, 1)

        # Normalize the signal so that its power becomes 1 (scaled by sqrt(2) when training).
        # x_normalized = x_centered / scale * torch.sqrt(torch.tensor(2.0))
        x_normalized = x_centered / scale

        return x_normalized

    def forward(self, x):

        # add parity bit
        # x = add_parity_bit(x)  # output size = (batch_size, k+1, 1)

        # switch the second and third dimension of the tensor
        x = torch.transpose(x, 1, 2)  # output size = (batch_size, 1, k)

        # outer encoder input size = (batch_size, 1, k), output size = (batch_size, F, k)
        x = self.outer_encoder(x)

        # quantization to 2 levels
        x = STEQuantize.apply(x, 1.0, 2)

        # interleaver
        x = self.set_interleaver(x)  # output size = (batch_size, F, n)

        # inner encoder
        x = self.inner_encoder(x)  # output size = (batch_size, 2, k)

        # switch the second and third dimension of the tensor
        x = torch.transpose(x, 1, 2)  # output size = (batch_size, n, 2)

        # power constraint
        x = self.power_constraint(x)  # output size = (batch_size, n, 2)

        return x


class EQ_CNN_block(nn.Module):
    def __init__(self, in_channels, k, n, F):
        super(EQ_CNN_block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=in_channels,
            k=n,
            n=n,
            F=F,
        )  # output size = (batch_size, F, n)
        # self.cnn_block2 = nn.Conv1d(
        #     in_channels=F, out_channels=F, kernel_size=1, padding="same"
        # )

    def forward(self, x):
        # input size = (batch_size, n, n), output size = (batch_size, F, n)
        x = self.cnn_block1(x)

        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        # x = self.cnn_block2(x)

        return x


class DEC_CNN_block(nn.Module):
    def __init__(self, in_channels, k, n, F):
        super(DEC_CNN_block, self).__init__()

        self.cnn_block1 = CNN_block(
            in_channels=in_channels,
            k=n,
            n=n,
            F=F,
        )  # output size = (batch_size, F, n)
        self.cnn_block2 = nn.Conv1d(
            in_channels=F, out_channels=F, kernel_size=1, padding="same"
        )

    def forward(self, x):
        # input size = (batch_size, F, n), output size = (batch_size, F, n)
        x = self.cnn_block1(x)

        # input size = (batch_size, n, n), output size = (batch_size, F, n)
        x = self.cnn_block2(x)

        return x


class DEC_CNN_last_block(nn.Module):
    def __init__(self, in_channels, k, n, F, output_size):
        super(DEC_CNN_last_block, self).__init__()

        self.in_channels = in_channels
        self.k = k
        self.n = n
        self.F = F
        self.output_size = output_size

        self.cnn_block1 = CNN_block(
            in_channels=in_channels,
            k=n,
            n=n,
            F=F,
        )  # output size = (batch_size, F, n)

        # input size = (batch_size, F, n), output size = (batch_size, F*n)
        self.flatten = nn.Flatten()

        self.decoder_ouput = nn.Linear(in_features=F * k, out_features=output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input size = (batch_size, F, n), output size = (batch_size, M1, n)
        x = self.cnn_block1(x)

        # flattten layer
        # input size = (batch_size, M1, n), output size = (batch_size, M1*n)
        x = self.flatten(x)

        # final decoder output
        # input size = (batch_size, M1*n), output size = (batch_size, k)
        x = self.decoder_ouput(x)

        # add sigmoid activation function
        x = self.sigmoid(x)

        # add one dimension to the output (second dimension) to make sure the output has the shape (batch_size, 1, k)
        x = x.unsqueeze(1)

        return x


class Sync_Block(nn.Module):
    def __init__(self, n, n_M):
        super(Sync_Block, self).__init__()
        self.n = n
        self.n_M = n_M
        # The first layer input shape: (batch_size, 2, 2n), output shape: (batch_size, 100, 2n)
        # self.cnn_block1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.cnn_block1 = CNN_block1(
            in_channels=2,
            out_channels=100,
            kernel_size=5,
            num_blocks=5,
        )

        # The second layer, input shape: (batch_size, 100, 2n), output shape: (batch_size, 1, 2n)
        # self.cnn_block2 = nn.Conv1d(
        #     in_channels=100, out_channels=1, kernel_size=1, padding="same"
        # )
        self.cnn_block2 = nn.Linear(in_features=100, out_features=1)

        # The third layer
        self.flatten = nn.Flatten()
        # The forth layer
        # self.linear = nn.Linear(
        #     in_features=int(self.n * 2), out_features=int(self.n - self.n_M + 1)
        # )
        self.linear = nn.Linear(
            in_features=int(self.n * 2 - 1), out_features=int(self.n - self.n_M)
        )
        self.softmax = nn.Softmax(dim=1)

    def normalize_unit_power(self, X):
        """
        Normalize the input tensor to have average unit power per block per.

        Args:
            X (torch.Tensor): Input tensor with shape (batch_size, nb, M), dtype: torch.complex64.

        Returns:
            torch.Tensor: Normalized input tensor with shape (batch_size, nb, M), dtype: torch.complex64.

        """
        # Calculate the power of the input tensor
        power = torch.mean(torch.abs(X) ** 2, dim=2, keepdim=True)
        # Avoid division by zero by clamping the minimum mean power
        power = torch.clamp(power, min=1e-12)
        # Normalize the input tensor to have unit power per block
        X = X / torch.sqrt(power)
        return X

    def starting_point_onehot(self, X):
        """
        Function to generate the one-hot encoded starting point for each batch.

        Args:
            X (torch.Tensor): (batch_size, 1).

        Returns:
            torch.Tensor: The one-hot encoded starting point for each batch.
        """
        batch_size = X.shape[0]
        device = X.device

        # Perform one-hot encoding, one hot tensor of shape (batch_size, num_classes)
        one_hot = (
            F.one_hot(X, num_classes=(self.n - self.n_M + 1)).squeeze(1).to(device)
        )

        return one_hot

    def forward(self, x):

        # input shape: (batch_size, 2n, 2), dtype:torch.float32

        # step 1: convert the input tensor to shape (batch_size, 2, 2n), dtype: torch.float32
        x = x.permute(0, 2, 1)  # output shape: (batch_size, 2, 2n)
        # step 2: apply the first layer
        x = self.cnn_block1(x)  # output shape: (batch_size, 100, 2n)

        # transpose the tensor to shape (batch_size, 2n, 100)
        x = x.permute(0, 2, 1)  # output shape: (batch_size, 2n, 100)

        # step 3: apply the second layer
        x = self.cnn_block2(x)  # output shape: (batch_size, 2n, 1)
        # step 4: apply the second layer
        x = self.flatten(x)  # shape: (batch_size, 2n)
        # step 5: apply the third layer
        x = self.linear(x)
        x_predicted_probability = self.softmax(x)  # shape: (batch_size, dmax / ts + 1)

        x_max_index = torch.argmax(
            x_predicted_probability, dim=1, keepdim=True
        )  # shape: (batch_size, 1)
        x_predicted_onehot = self.starting_point_onehot(
            x_max_index
        )  # shape: (batch_size, int(dmax/ts) + 1)

        return x_predicted_probability, x_predicted_onehot


def add_padding(input_tensor: torch.Tensor, n_M: int) -> torch.Tensor:
    """
    Pads the input tensor with zeros on the left and right along the last dimension.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, F, n) with dtype torch.float32.
        n_M (int): Number of zeros to pad on both the left and the right sides of the last dimension.

    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, F, n + 2*n_M).
    """
    # F.pad expects the padding dimensions in the order: (pad_left, pad_right, pad_top, pad_bottom, ...)
    # Since we need to pad the last dimension (n), we set pad=(n_M, n_M)
    padded_tensor = F.pad(input_tensor, pad=(n_M, n_M), mode="constant", value=0)
    return padded_tensor


def remove_padding(padded_tensor: torch.Tensor, n_M: int) -> torch.Tensor:
    """
    Removes the padding (n_M zeros) from the left and right along the last dimension.

    Args:
        padded_tensor (torch.Tensor): Padded tensor of shape (batch_size, F, n + 2*n_M).
        n_M (int): Number of padded zeros on each end of the last dimension.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, F, n) after removing padding.
    """
    # We remove the first n_M and last n_M elements along the last dimension.
    # Assumes padded_tensor.shape[-1] >= 2*n_M.

    if n_M > 0:
        return padded_tensor[..., n_M:-n_M]
    else:
        return padded_tensor


class Decoder(nn.Module):
    def __init__(
        self,
        k,
        n,
        F,
        n_M,
        num_iteration,
    ):
        super(Decoder, self).__init__()

        self.k = k
        self.n = n
        self.F = F
        self.n_M = n_M
        self.num_iteration = num_iteration

        # input size = (batch_size, 2, n*N_up), output size = (batch_size, 2, n)
        self.dec_input = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=n, kernel_size=1, stride=1),
            nn.BatchNorm1d(n),
            nn.ELU(),
        )

        # input size = (batch_size, n+F, n), output size = (batch_size, F, n)
        self.eq_block = EQ_CNN_block(in_channels=n + F, k=n, n=n, F=F)

        # input size = (batch_size, F, n), output size = (batch_size, F, n)
        self.dec_block = DEC_CNN_block(in_channels=F, k=n, n=n, F=F)

        # input size = (batch_size, F, n), output size = (batch_size, 1, k)
        self.dec_last_block = DEC_CNN_last_block(
            in_channels=F, k=k, n=n, F=F, output_size=k
        )

    def interleaver(self, inputs, p_array):
        # inputs has the shape (batch_size, F, n)

        inputs = inputs.permute(2, 0, 1)  # output size = (n, batch_size, F)
        res = inputs[p_array]
        res = res.permute(1, 2, 0)  # output size = (batch_size, F, n)

        return res

    def deinterleaver(self, inputs, p_array):
        # inputs has the shape (batch_size, F, n)

        reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            reverse_p_array[p_array[idx]] = idx

        reverse_p_array = torch.LongTensor(reverse_p_array).view(len(p_array))

        inputs = inputs.permute(2, 0, 1)  # input size = (n, batch_size, F)
        res = inputs[reverse_p_array]
        res = res.permute(1, 2, 0)  # output size = (batch_size, F, n)

        return res

    def forward(self, y_delay_removed):

        device = y_delay_removed.device

        # y_delay_removed has the shape (batch_size, n+2n_M, 2)
        # switch the second and third dimension of the tensor
        y_delay_removed = y_delay_removed.permute(0, 2, 1)

        # input size = (batch_size, 2, n+2n_M), output size = (batch_size, n, n+2n_M)
        y = self.dec_input(y_delay_removed)

        block_length = self.n
        p_array = generate_permutation_array(block_length, seed=0)

        # initiate i_c, i_c_pri is all zeros with the shape (batch_size, F, n)
        i_c_pri = torch.zeros(
            (y.shape[0], self.F, self.n), dtype=torch.float32, device=device
        )

        # iterative EQ-DEC block
        for i in range(self.num_iteration):

            # pad i_c_pri with zeros from the left and the right to make it of shape (batch_size, F, n+2n_M)
            i_c_pri_padding = add_padding(
                i_c_pri, self.n_M
            )  # output size = (batch_size, F, n+2n_M)

            # concatenate y with i_c_pri
            # y is of shape (batch_size, n, n+2n_M), i_c_pri shape (batch_size, F, k+2n_M)
            y_concatenate = torch.cat(
                (y, i_c_pri_padding), dim=1
            )  # output size = (batch_size, n+F, n+2n_M)

            # EQ-CNN block
            # input size = (batch_size, n+F, n+2n_M), output size = (batch_size, F, n+2n_M)
            i_c = self.eq_block(y_concatenate)

            # remove padding
            # input size = (batch_size, F, n+2n_M), output size = (batch_size, F, n)
            i_c = remove_padding(i_c, self.n_M)

            # update i_c
            i_c = i_c - i_c_pri

            # deinterleaver
            i_c = self.deinterleaver(i_c, p_array)

            # DEC-CNN block
            # input size = (batch_size, F, n), output size = (batch_size, F, n)
            i_b = self.dec_block(i_c)

            # update i_c_pri
            i_c_pri = i_b - i_c

            # interleaver
            i_c_pri = self.interleaver(i_c_pri, p_array)

        #! last iteration
        # EQ-CNN last block

        # pad i_c_pri with zeros from the left and the right to make it of shape (batch_size, F, n+2n_M)
        i_c_pri_padding = add_padding(
            i_c_pri, self.n_M
        )  # output size = (batch_size, F, n+2n_M)

        # concatenate y with i_c_pri
        # y is of shape (batch_size, n, n+2n_M), i_c_pri shape (batch_size, F, k+2n_M)
        y_concatenate = torch.cat(
            (y, i_c_pri_padding), dim=1
        )  # output size = (batch_size, n+F, n+2n_M)

        # input size = (batch_size, M2, n), output size = (batch_size, F, n)
        i_c = self.eq_block(y_concatenate)

        # remove padding
        # input size = (batch_size, F, n+2n_M), output size = (batch_size, F, n)
        i_c = remove_padding(i_c, self.n_M)

        # update i_c
        i_c = i_c - i_c_pri

        # deinterleaver
        i_c = self.deinterleaver(i_c, p_array)

        # DEC-CNN last block
        # input size = (batch_size, F, k), output size = (batch_size, 1, k)
        y_decoded = self.dec_last_block(i_c)

        # switch the second and third dimension of the tensor, output size = (batch_size, 1, k)
        y_decoded = y_decoded.permute(0, 2, 1)

        return y_decoded


class Turbo_AutoEncoder(nn.Module):
    def __init__(
        self,
        k,
        n,
        n_M,
        F,
        channel,
        is_delay,
        num_iteration,
    ):
        super(Turbo_AutoEncoder, self).__init__()

        self.k = k
        self.n = n
        self.n_M = n_M
        self.F = F
        self.is_delay = is_delay
        self.channel = channel
        self.num_iteration = num_iteration

        self.encoder = Encoder(in_channels=1, k=self.k, n=self.n, F=self.F)
        if self.is_delay:
            self.sync_block = Sync_Block(n=self.n, n_M=n_M)
        self.decoder = Decoder(
            k=self.k,
            n=self.n,
            F=self.F,
            n_M=self.n_M,
            num_iteration=self.num_iteration,
        )

    def forward(self, x, SNR_db_low, SNR_db_high, is_training_encoder=False):

        # Transmitter part
        # input size = (batch_size, 1, k), output size = (batch_size, n, 2)
        x = self.encoder(x)

        if self.channel == "AWGN":
            x = AWGN_Channel_new(x, SNR_db_low, SNR_db_high)
            x = self.decoder(x)
            return x

        elif self.channel == "proakis_c_channel":
            x, true_starting_point_onehot = proakis_c_channel(
                x, SNR_db_low, SNR_db_high
            )
            # x, true_starting_point_onehot = proakis_c_channel(
            #     x, 2 * self.n, "awgn", SNR_db_low, SNR_db_high
            # )
            # x, true_starting_point_onehot = proakis_c_channel(
            #     x, self.n, "awgn", SNR_db_low, SNR_db_high
            # )
            # case 1: delay is generated randomly (0, dmax) and synchronization block is added
            if self.is_delay:
                # normalize the input tensor to have average unit power per block
                x = unit_power_normalization(x)
                (
                    predicted_starting_point_probability,
                    predicted_starting_point_onehot,
                ) = self.sync_block(x)
                if is_training_encoder:
                    # When training the encoder, the true starting point is used
                    x = cutoff_proakis(x, true_starting_point_onehot, self.n_M)
                    x = self.decoder(x)
                else:
                    # When training the decoder (or inference), the predicted starting point is used
                    x = cutoff_proakis(x, predicted_starting_point_onehot, self.n_M)
                    x = self.decoder(x)
                return (
                    x,
                    predicted_starting_point_probability,
                    true_starting_point_onehot,
                )
            # case 2: delay is generated with all zeros and synchronization block is added
            elif not self.is_delay:
                # normalize the input tensor to have average unit power per block
                x = unit_power_normalization(x)
                x = cutoff_proakis(x, true_starting_point_onehot, self.n_M)
                x = self.decoder(x)
                return x
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
