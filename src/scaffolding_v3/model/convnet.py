from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        sidelength: int,
        conv_channels: Sequence[int],
        linear_channels: Sequence[int],
        use_dropout: bool,
    ):
        super(ConvNet, self).__init__()

        self.convs = self._init_conv_layers(in_channels, conv_channels)

        # Calculate the number of features after the convolutional layers
        conv_out_dim = self._get_conv_out_dim(in_channels, sidelength, self.convs)

        self.linears = self._init_linear_layers(
            conv_out_dim, linear_channels, num_classes, use_dropout
        )

    def _get_conv_out_dim(
        self, in_channels: int, sidelength: int, convs: nn.Sequential
    ) -> int:
        # Create a random tensor to get the output size of the convolutional layers
        x = torch.rand((1, in_channels, sidelength, sidelength))
        x = convs(x)
        return x.numel()

    def _init_conv_layers(
        self, in_channels: int, conv_channels: Sequence[int]
    ) -> nn.Sequential:
        conv_blocks = [
            nn.Conv2d(in_channels, conv_channels[0], 3, 1),
            nn.ReLU(),
        ]
        for conv_channel_in, conv_channel_out in zip(conv_channels, conv_channels[1:]):
            conv_blocks.extend(
                [
                    nn.Conv2d(conv_channel_in, conv_channel_out, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
        return nn.Sequential(*conv_blocks)

    def _init_linear_layers(
        self,
        conv_out_dim: int,
        linear_channels: Sequence[int],
        num_classes: int,
        use_dropout: bool,
    ) -> nn.Sequential:
        linear_blocks = [
            nn.Linear(conv_out_dim, linear_channels[0]),
            nn.ReLU(),
        ]
        for linear_channel_in, linear_channel_out in zip(
            linear_channels, linear_channels[1:]
        ):
            linear_blocks.extend(
                [
                    nn.Linear(linear_channel_in, linear_channel_out),
                    nn.ReLU(),
                ]
            )
            if use_dropout:
                linear_blocks.append(nn.Dropout(0.5))

        linear_blocks.append(nn.Linear(linear_channels[-1], num_classes))
        return nn.Sequential(*linear_blocks)

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.linears(x)
        output = F.log_softmax(x, dim=1)
        return output
