from typing import Any, List

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    

    def __init__(self, 
                 n_features: int,
                 n_hidden_layers: int=1,
                 hidden_dim_width: List[int]|int=32,
                 dropout: float=0):
        super().__init__()
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if type(hidden_dim_width) == list: 
            assert len(hidden_dim_width) == n_hidden_layers
        else:
            hidden_dim_width = [hidden_dim_width]*n_hidden_layers
        self.dropout = dropout
        self.ln_start = nn.Linear(in_features=n_features, out_features=hidden_dim_width[0])
        self.ln_start.apply(weights_init)
        layers = []
        for i in range(n_hidden_layers-1):
            layer = [
                nn.Linear(in_features=hidden_dim_width[i], out_features=hidden_dim_width[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            layers += layer
        if n_hidden_layers >= 1: 
            self.layers = nn.Sequential(*layers)
            self.layers.apply(weights_init)
        else: 
            self.layers = None
        self.ln_end = nn.Linear(in_features=hidden_dim_width[-1], out_features=1)
        self.ln_end.apply(weights_init)

    def forward(self, x) -> Any:
        out = F.gelu(self.ln_start(x))
        out = F.dropout(out, self.dropout)
        if self.layers: out = self.layers(out)
        out = self.ln_end(out)
        return out