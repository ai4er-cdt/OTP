from typing import Any

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, 
                 n_features: int,
                 n_hidden_layers: int=1,
                 hidden_dim_width: int=32,
                 dropout: float=0):
        super().__init__()
        self.dropout = dropout
        self.ln_start = nn.Linear(in_features=n_features, out_features=hidden_dim_width)
        self.n_hidden_layers = n_hidden_layers
        if n_hidden_layers > 1:
            self.layers = nn.Sequential(
                *[
                    nn.Linear(in_features=hidden_dim_width, out_features=hidden_dim_width),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ]*(n_hidden_layers-1)
            )
        self.ln_end = nn.Linear(in_features=hidden_dim_width, out_features=1)

    def forward(self, x) -> Any:
        out = F.gelu(self.ln_start(x))
        out = F.dropout(out, self.dropout)
        if self.n_hidden_layers > 1:
            out = self.layers(out)
        out = self.ln_end(out)
        return out
    
print('hi from MLP.py!2')