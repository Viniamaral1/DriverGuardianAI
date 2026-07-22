"""
Residual Neural Network for driver fatigue classification.

This module defines a fully connected residual neural network
designed for tabular driver monitoring features.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Fully connected residual block.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    output_dim : int
        Number of output features.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout=0.3
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            output_dim
        )

        self.batch_norm = nn.BatchNorm1d(
            output_dim
        )

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )

        if input_dim != output_dim:

            self.residual = nn.Linear(
                input_dim,
                output_dim
            )

        else:

            self.residual = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the residual block.
        """

        identity = self.residual(x)

        out = self.linear(x)

        out = self.batch_norm(out)

        out = self.activation(out)

        out = self.dropout(out)

        out += identity

        return out


class FatigueResidualNN(nn.Module):
    """
    Residual neural network for
    driver fatigue classification.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    hidden_dims : list
        Hidden layer dimensions.

    num_classes : int
        Number of output classes.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=None,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()

        if hidden_dims is None:

            hidden_dims = [
                256,
                128,
                64
            ]

        self.blocks = nn.ModuleList()

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            self.blocks.append(

                ResidualBlock(

                    previous_dim,

                    hidden_dim,

                    dropout

                )

            )

            previous_dim = hidden_dim

        self.output_layer = nn.Linear(

            previous_dim,

            num_classes

        )

    def forward(self, x):
        """
        Forward pass through the network.
        """

        for block in self.blocks:

            x = block(x)

        return self.output_layer(x)