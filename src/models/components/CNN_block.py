# This file is to build CNN block as a component of the model
# The CNN block is a sequence of convolutional layers followed by batch normalization and ELU activation
# The block is used in the transmitter and receiver parts of the model

import torch.nn as nn


class CNN_block1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_blocks,
        padding="same",
    ):
        super(CNN_block1, self).__init__()

        layers = []
        for _ in range(num_blocks):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            layers.append(nn.ELU())
            in_channels = out_channels  # Update in_channels for the next block

        self.cnn_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn_block(x)
        return x
