import torch as t
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, 
                 n_features: int,
                 hidden_dim_width: int=1):
        super().__init__()
        self.ln_start = nn.Linear(in_features=n_features, out_features=hidden_dim_width)
        self.ln_end = nn.Linear(in_features=hidden_dim_width, out_features=1)

    def forward(self, x):
        out = F.relu(self.ln_start(x))
        out = self.ln_end(out)
        return out