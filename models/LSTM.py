from typing import Any, Optional

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class LSTMcell(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 d_hidden: int,
                 dropout: float=0.2,
                 device: str="cpu"):
        super().__init__()
        self.device = device
        self.dpout = nn.Dropout(dropout)
        self.d_hidden = d_hidden
        self.n_outputs = n_outputs
        # gates
        self.i2fg = nn.Linear(n_inputs+d_hidden, d_hidden)
        self.i2ig = nn.Linear(n_inputs+d_hidden, d_hidden)
        self.i2og = nn.Linear(n_inputs+d_hidden, d_hidden)
        self.fgbn = nn.BatchNorm1d(d_hidden)
        self.igbn = nn.BatchNorm1d(d_hidden)
        self.ogbn = nn.BatchNorm1d(d_hidden)
        # cell state
        self.i2c = nn.Linear(n_inputs+d_hidden, d_hidden)
        self.cbn = nn.BatchNorm1d(d_hidden)
        # output
        self.h2o = nn.Linear(d_hidden, n_outputs)

    def forward(self, X) -> Any:
        # X: (batch_size, n_seq, n_inputs)
        X = self.dpout(X)
        hidden = t.zeros(X.shape[0], self.d_hidden, requires_grad=False).to(self.device)
        cell = t.zeros_like(hidden)
        # reshape X to (n_seq, batch_size, n_inputs)
        X = X.transpose(0, 1)
        out = t.zeros(X.shape[0], X.shape[1], self.n_outputs, requires_grad=False).to(self.device)
        for i in range(X.shape[0]):
            combined = t.cat([X[i], hidden], dim=-1)
            # gates
            forget_gate = F.sigmoid(self.fgbn(self.i2fg(combined)))
            input_gate = F.sigmoid(self.igbn(self.i2ig(combined)))
            output_gate = F.sigmoid(self.ogbn(self.i2og(combined)))
            # cell state
            candidate_cell = F.tanh(self.cbn(self.i2c(combined)))
            cell = forget_gate*cell + input_gate*candidate_cell
            # hidden state
            hidden = output_gate*F.tanh(cell)
            out[i] = F.tanh(self.h2o(hidden))
        return out.transpose(0, 1)

class LSTM(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 n_layers: int,
                 d_hidden: int,
                 d_model: Optional[int],
                 dropout: float=0.2,
                 device: str="cpu"):
        super().__init__()
        assert n_layers > 0
        d_model = d_hidden if d_model is None else d_model
        layers = []
        for _ in range(n_layers):
            block = LSTMcell(n_inputs, d_model, d_hidden, dropout, device)
            n_inputs = block.n_outputs
            layers.append(block)
        self.layers = nn.Sequential(*layers)     
        self.ln_out = nn.Linear(n_inputs, n_outputs)   

    def forward(self, X) -> Any:
        out = self.layers(X)
        out = self.ln_out(out[:, -1, :])
        return out