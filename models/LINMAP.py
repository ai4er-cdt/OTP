import torch as t
import torch.nn as nn
import torch.nn.functional as F


class LINMAP(nn.Module):
    def __init__(self, 
                 n_features: int):
        super().__init__()
        self.map = nn.Linear(in_features=n_features, out_features=1, dtype=t.float32)

    def forward(self, x):
        return self.map(x)