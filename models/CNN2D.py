from typing import Any, Tuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F


# default hyperparameters
dropout = 0.2
n_pure_layers = 3
n_mix_layers = 3
n_features = 5
n_channels = n_features * 5
kernel_size = 3
# ---------------

class ConvBlock(nn.Module):
    def __init__(self, 
                 is_pure: bool,
                 n_features: int,
                 n_channels: int,
                 kernel_size: int|Tuple[int, int],
                 n_layers: int,
                 dropout: float):
        super().__init__()
        self.n_layers = n_layers
        n_groups = n_features if is_pure else 1
        
        self.init_conv = nn.Conv2d(in_channels=n_features,
                                   out_channels=n_channels,
                                   kernel_size=kernel_size,
                                   padding="same",
                                   groups=n_groups)
        self.init_BN = nn.BatchNorm2d(n_channels)
        self.init_dpout = nn.Dropout(dropout)
        if self.n_layers > 1:
            self.layers = nn.Sequential(
                *[
                    nn.Sequential(nn.Conv2d(in_channels=n_channels,
                              out_channels=n_channels,
                              kernel_size=kernel_size,
                              padding="same",
                              groups=n_groups),
                    nn.BatchNorm2d(n_channels),
                    nn.GELU(),
                    nn.Dropout(dropout)) for _ in range(n_layers-1)
                ]
            )

    def forward(self, x) -> Any:
        x = self.init_conv(x)
        x = self.init_BN(x)
        x = F.gelu(x)
        x = self.init_dpout(x)
        if self.n_layers > 1: x = self.layers(x)
        return x


class CNN2D(nn.Module):
    def __init__(self,
                 n_pure_layers: int,
                 n_mix_layers: int,
                 n_features: int,
                 n_channels: int,
                 kernel_size: int|Tuple[int, int],
                 dropout: float):
        super().__init__()

        self.use_pure = n_pure_layers > 0
        self.use_mix = n_mix_layers > 0
        c = 0
        if self.use_pure:
            c += 1
            if n_channels % n_features != 0: raise ValueError("n_channels must be divisible by n_features to use pure blocks.")
            self.pure_block = ConvBlock(True, n_features, n_channels, kernel_size, n_pure_layers, dropout)
        if self.use_mix:
            c += 1
            self.mix_block = ConvBlock(False, n_features, n_channels, kernel_size, n_mix_layers, dropout)
        if c == 0: raise Exception("No layers! n_pure_layers and/or n_mix_layers must be > 0.")
        self.fc = nn.Linear(c*n_channels, 1)

    def forward(self, x) -> Any:
        # apply pure and mixed bocks in parallel and concat.
        streams = []
        if self.use_pure: out_pure = self.pure_block(x); streams.append(out_pure)
        if self.use_mix: out_mix = self.mix_block(x); streams.append(out_mix)
        if len(streams) == 2: out = t.cat((streams[0], streams[1]), dim=1)
        else: out = streams[0]
        # global average pooling
        out = F.avg_pool2d(out, out.shape[-2:]).squeeze()
        # final prediction
        out = self.fc(out)
        return out