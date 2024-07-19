from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from .conv import SpectralConv
from .mlp import MLP


class FNO(nn.Module):
    """
    FNO model consisting of a sequence of a lifting, `n` fourier 
    interal operator layers, and a projection layer
    """

    def __init__(self, 
                 n_modes,
                 hidden_channels,
                 in_channels=3,
                 out_channels=1,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 non_linearity=F.gelu,
                 skip_fno_bias=False,
                 fft_norm="forward",
                 rank=0.5,
                 max_n_modes=None):
        
        super(FNO, self).__init__()
        self.n_dim = len(n_modes)

        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )


        # Spectral convolution layers
        self.convs = SpectralConv(
            self.hidden_channels,
            self.hidden_channels,
            self.n_modes,
            max_n_modes=max_n_modes,
            n_layers=n_layers,
            fft_norm=fft_norm,
            rank=rank
        )

        self.fno_skips = nn.ModuleList(
            [
                getattr(nn, f"Conv{self.n_dim}d")(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=1,
                    bias=skip_fno_bias,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, output_shape=None, **kwargs):
        """TFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        # if output_shape is None:
        #     output_shape = [None]*self.n_layers
        # elif isinstance(output_shape, tuple):
        #     output_shape = [None]*(self.n_layers - 1) + [output_shape]
        

        x = self.lifting(x)

        for layer_idx in range(self.n_layers):
            x_skip_fno = self.fno_skips[layer_idx](x)
            x_fno = self.convs(x, layer_idx, output_shape=output_shape)

            # if self.norm is not None:
            #     x_fno = self.norm[self.n_norms * layer_idx](x_fno)

            x = x_fno + x_skip_fno

            if layer_idx < (self.n_layers - 1):
                x = self.non_linearity(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self._n_modes = n_modes