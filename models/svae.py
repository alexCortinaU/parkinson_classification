from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import Convolution

__all__ = ["spatialVAE"]

class spatialVAE(AutoEncoder):
    def __init__(
            self,
            spatial_dims: int,
            in_shape: Sequence[int],
            out_channels: int,
            latent_size: int,
            channels: Sequence[int],
            strides: Sequence[int],
            kernel_size: Union[Sequence[int], int] = 3,
            up_kernel_size: Union[Sequence[int], int] = 3,
            num_res_units: int = 0,
            inter_channels: Optional[list] = None,
            inter_dilations: Optional[list] = None,
            num_inter_units: int = 2,
            act: Optional[Union[Tuple, str]] = Act.PRELU,
            norm: Union[Tuple, str] = Norm.INSTANCE,
            dropout: Optional[Union[Tuple, str, float]] = None,
            bias: bool = True,
            use_sigmoid: bool = True,
            ) -> None:
        
        self.in_channels, *self.in_shape = in_shape
        self.use_sigmoid = use_sigmoid

        self.latent_size = latent_size
        self.final_size = np.asarray(self.in_shape, dtype=int)

        super().__init__(
            spatial_dims,
            self.in_channels,
            out_channels,
            channels,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            inter_channels,
            inter_dilations,
            num_inter_units,
            act,
            norm,
            dropout,
            bias,
        )

        padding = same_padding(self.kernel_size)

        for s in strides:
            self.final_size = calculate_out_shape(self.final_size, self.kernel_size, s, padding)  # type: ignore

        # linear_size = int(np.product(self.final_size)) * self.encoded_channels
        self.mu = Convolution(spatial_dims=self.dimensions,
                              in_channels=self.encoded_channels,
                              out_channels=self.latent_size,
                              strides=1,
                              kernel_size=self.kernel_size,
                              act=None,
                              norm=None,
                              dropout=None,
                              bias=True)
        self.logvar = Convolution(spatial_dims=self.dimensions,
                              in_channels=self.encoded_channels,
                              out_channels=self.latent_size,
                              strides=1,
                              kernel_size=self.kernel_size,
                              act=None,
                              norm=None,
                              dropout=None,
                              bias=True)
        self.decodeL = Convolution(spatial_dims=self.dimensions,
                              in_channels=self.latent_size,
                              out_channels=self.encoded_channels,
                              strides=1,
                              kernel_size=self.kernel_size,
                              act=None,
                              norm=None,
                              dropout=None,
                              bias=self.bias)
        
    def encode_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encode(x)
        x = self.intermediate(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def decode_forward(self, z: torch.Tensor, use_sigmoid: bool = True) -> torch.Tensor:
        x = nn.functional.relu(self.decodeL(z))
        x = self.decode(x)
        if use_sigmoid:
            x = torch.sigmoid(x)
        return x
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        
        std = torch.exp(0.5 * logvar)

        # if self.training:  # multiply random noise with std only during training
        #     std = torch.randn_like(std).mul(std)
        
        return std.add(mu)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_forward(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_forward(z, self.use_sigmoid), mu, logvar, z