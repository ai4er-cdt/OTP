import os
from typing import Optional
import SimDataset
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import trange


parent_dir = os.path.dirname(os.path.abspath("train.py"))
t.manual_seed(123456)

# default hyperparameters
batch_size = 32
max_iters = 5000 
lr = 1e-3
weight_decay = 1e-5
# ---------------

class EarlyStopping:
    def __init__(self, patience: int=5, min_delta: int=0, threshold: float=float("inf")):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.threshold = threshold
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss > self.best_loss + self.min_delta and self.best_loss < self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model: nn.Module,  
                X_train: t.Tensor, 
                y_train: t.Tensor, 
                name: str="model",
                X_val: Optional[t.Tensor]=None,
                y_val: Optional[t.Tensor]=None,
                early_stopping: Optional[bool]=False,
                eval_iter: Optional[int]=None,
                device: Optional[str]=None):
    
    if device == None: device = "cuda" if t.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"device: {device}")
    # print number of parameters
    print(f"{sum([p.numel() for p in model.parameters()])} parameters.")
        
    # get training data
    train_dataset = SimDataset.SimDataset(X_train, y_train, device)
    train_DL = DataLoader(train_dataset, batch_size, shuffle=True)
    data_iterator = cycle(train_DL)

    validate = X_val is not None

    # training
    model.train()
    criterion = nn.MSELoss()
    opt = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss = []; val_loss = []
    if early_stopping: es = EarlyStopping(patience=100, min_delta = 0.01, threshold=2.25)
    if eval_iter is None:
        for iter in trange(max_iters):
            # use dataloader to sample a batch
            x, y = next(data_iterator)
            # update model
            out = model(x)
            loss = criterion(out.squeeze(-1), y); train_loss.append(loss.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if validate:
                out = model(X_val.to(device))
                loss = criterion(out.squeeze(-1), y_val.to(device)); val_loss.append(loss.item())
                es(loss.item())
                if es.early_stop: 
                    print("early stopping")
                    break
    else:
        for iter in range(max_iters):
            # use dataloader to sample a batch
            x, y = next(data_iterator)
            # update model
            out = model(x)
            loss = criterion(out.squeeze(-1), y); train_loss.append(loss.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if iter % eval_iter == 0:
                print("----------")
                print(f"Training Loss: {loss.item()}")

            if validate:
                out = model(X_val.to(device))
                loss = criterion(out.squeeze(-1), y_val.to(device)); val_loss.append(loss.item())
                if iter % eval_iter == 0:
                    print(f"Validation Loss: {loss.item()}")
                es(loss.item())
                if es.early_stop: 
                    print("early stopping")
                    break
    if validate:
        return model, train_loss, val_loss
    else:
        return model. train_loss