"""
This file is to constract the CNN module for the model Turbo AutoEncoder
"""

import torch
import torch.nn as nn


class SameShapeConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, num_layers, stride, padding, bias
    ):
        super(SameShapeConv1d, self).__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
            layers.append(nn.ELU())
            in_channels = out_channels  # Update in_channels for the next block

        self.cnn_block = nn.Sequential(*layers)

    def forward(self, x):
        # x is originally of shape (250, 100, 1), after transpoose it becomes (250, 1, 100)
        x = torch.transpose(x, 1, 2)

        x = self.cnn_block(x)

        output = torch.transpose(x, 1, 2)

        return output
