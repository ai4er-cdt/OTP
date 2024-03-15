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
                loss_function: str="mse",
                X_val: Optional[t.Tensor]=None,
                y_val: Optional[t.Tensor]=None,
                early_stopping: Optional[bool]=False,
                eval_iter: Optional[int]=None,
                device: Optional[str]=None):
    
    loss_L1 = nn.L1Loss()
    loss_MSE = nn.MSELoss()
    def composite_loss(y_pred, y_test, model, lambda_l1=1e-4, lambda_l2=1e-4, device="cpu"):
        _loss_L1 = loss_L1(y_pred, y_test)
        _loss_MSE = loss_MSE(y_pred, y_test)

        # weight sparsity
        sparsity = t.tensor(0.).to(device)
        for param in model.parameters(): sparsity += t.norm(param, 1)
        # weight decay
        decay = t.tensor(0.).to(device)
        for param in model.parameters(): decay += t.norm(param, 2)

        return _loss_L1 + lambda_l1*sparsity + _loss_MSE + lambda_l2*decay
    
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
    if loss_function == "mse":
        criterion = nn.MSELoss()
    elif loss_function == "l1":
        criterion = nn.L1Loss()
    elif loss_function == "composite":
        criterion = composite_loss
    opt = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss = []; val_loss = []
    if early_stopping: es = EarlyStopping(patience=100, min_delta = 0.1, threshold=50)
    if eval_iter is None:
        for iter in trange(max_iters):
            # use dataloader to sample a batch
            x, y = next(data_iterator)
            x = x.float(); y = y.float()
            # update model
            out = model(x)
            loss = criterion(out.squeeze(-1), y); train_loss.append(loss.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if validate:
                out = model(X_val.to(device))
                loss = criterion(out.squeeze(-1), y_val.to(device)); val_loss.append(loss.item())
                mape = t.mean(t.abs((y_val.to(device) - out.squeeze(-1))/y_val.to(device)))*100
                if early_stopping:
                    es(mape.item())
                    if es.early_stop: 
                        print("early stopping")
                        break
    else:
        for iter in range(max_iters):
            # use dataloader to sample a batch
            x, y = next(data_iterator)
            x = x.float(); y = y.float()
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
                mape = t.mean(t.abs((y_val.to(device) - out.squeeze(-1))/y_val.to(device)))*100
                if iter % eval_iter == 0:
                    print(f"Validation Loss: {loss.item()}")
                if early_stopping:
                    es(mape.item())
                    if es.early_stop: 
                        print("early stopping")
                        break
    if validate:
        return model, train_loss, val_loss
    else:
        return model, train_loss