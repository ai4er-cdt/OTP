import os
import SimDataset
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import trange
import numpy as np


parent_dir = os.path.dirname(os.path.abspath("train.py"))
print(f"parent_dir: {parent_dir}")
t.manual_seed(123456)

# default hyperparameters
batch_size = 32
max_iters = 5000 
lr = 1e-3
# ---------------

def train_model(model: nn.Module, 
                name: str, 
                X: t.Tensor, 
                y: t.Tensor, 
                save_dir: str,
                device: str|None=None):
    
    if device == None:
        device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # print number of parameters
    print(f"{sum([p.numel() for p in model.parameters()])} parameters.")
        
    # get training data
    train_dataset = SimDataset.SimDataset(X, y, device)
    train_DL = DataLoader(train_dataset, batch_size, shuffle=True)
    data_iterator = cycle(train_DL)

    # training
    model.train()
    criterion = nn.MSELoss()
    opt = t.optim.AdamW(model.parameters(), lr=lr)
    full_loss = []
    for iter in trange(max_iters):
        # use dataloader to sample a batch
        x, y = next(data_iterator)
        # update model
        out = model(x)
        loss = criterion(out.squeeze(-1), y); full_loss.append(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print(os.path.join(parent_dir, f"{save_dir}/saved_models/{name}.pt"))
    t.save(model.state_dict(), os.path.join(parent_dir, f"{save_dir}/saved_models/{name}.pt"))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.sqrt(full_loss)*1e-6, linestyle="--", color="red", alpha=0.5)
    ax.set_ylabel("RMSE / Sv"); ax.set_xlabel("Training Step")
    ax.set_title(f"{name}: Loss")
    plt.savefig(os.path.join(parent_dir, f"{save_dir}/loss_curves/{name}.png"), dpi=400)
    plt.close()

    print(f"final RMSE loss: {np.sqrt(loss.item())*1e-6} Sv")
    print(f"model saved to {save_dir}/saved_models/{name}.pt")
    print(f"loss curve saved to {save_dir}/loss_curves/{name}.png")