import warnings
warnings.filterwarnings("ignore")
import pickle
import sys; sys.path.append("../models")
import utils
import train
from SOLODOCH import MLP

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import xarray as xr

from typing import List, Tuple


# globals
gdrive = "/mnt/g/My Drive/GTC"
data_home = f"{gdrive}/ecco_data_minimal"
sections = ["26N", "30S", "55S", "60S"]
coordinates = ["time", "latitude", "longitude"]
vars = ["SSH", "SST", "SSS", "OBP", "ZWS"]

# training settings
train.batch_size = 128
train.max_iters = 50000
train.lr = 1e-3
train.weight_decay = 1e-5

def get_data(section: str, 
             remove_tr_ssn: bool, 
             standardise: str="ours", 
             splits: List[float]=[0.6, 0.8]) -> Tuple[t.Tensor, t.Tensor]:
    # load input data
    data = xr.open_dataset(f"{data_home}/{section}.nc").transpose(*coordinates)
    # choose the single latitude we are modelling (middle)
    data = data.isel(latitude=[1])
    # remove empty coords
    all_nan = data.isnull().all(dim=["time", "latitude"])
    data = data.where(~all_nan, drop=True)
    # first 60% of data is train
    # next 20% is validation
    # last 20% is test
    split_trainval = int(splits[0]*data.sizes["time"])
    split_valtest = int(splits[1]*data.sizes["time"])    

    if remove_tr_ssn:
        # de-trend inputs
        data_train = data.isel(time=slice(0, split_trainval))
        for v in range(len(vars)):
            for l in range(data.sizes["longitude"]):
                series = data_train[vars[v]].isel(latitude=0, longitude=l).to_numpy().flatten()
                y = series[~np.isnan(series)]
                # fit linear trend to training data
                if len(y) > 1: slope, intercept = np.polyfit(np.arange(len(y)), y, deg=1)
                else: slope = np.nan; intercept = np.nan
                # subtract trend from full data
                trend = slope*np.arange(data.sizes["time"]) + intercept
                data[vars[v]].loc[:, data.latitude.item(), data.longitude[l].item()] -= trend
        # de-season inputs
        data_train = data.isel(time=slice(0, split_trainval))
        monthly_means = data_train.groupby("time.month").mean()
        data = data.groupby("time.month") - monthly_means

    # standardise following Solodoch et al. (remove mean, and temporal std)
    if standardise == "solodoch":
        data_train = data.isel(time=slice(0, split_trainval))
        for var in vars:
            m = data_train[var].mean(skipna=True).item()
            s = data_train[var].std(dim=["time"], skipna=True)
            data[var] = (data[var] - m) / s
    # grab only OBP - return a tensor
    pp_data = utils.reshape_inputs(data,
                                keep_coords=["time", "longitude"],
                                data_vars=["OBP"],
                                return_pt=True, verbose=False)

    # load output data
    with open(f"{gdrive}/moc/single_lats/{section}_moc_density.pickle", "rb") as f: moc = t.Tensor(pickle.load(f))
    if remove_tr_ssn:
        # calculate and remove linear trend and seasonal (monthly) components
        slope, intercept = np.polyfit(np.arange(split_trainval), moc[:split_trainval], deg=1)
        trend = slope*np.arange(len(moc)) + intercept
        moc -= trend
        ix = (split_trainval // 12) * 12
        monthly_means = moc[:ix].reshape(-1, 12).mean(axis=0)
        moc -= monthly_means.repeat(len(moc)//12)

    return pp_data, moc, split_trainval, split_valtest

def train_solodoch(section: str, 
                   remove_tr_ssn: bool, 
                   standardise: str, 
                   splits: List[float]=[0.6, 0.8], 
                   device="cpu"):
    """
    Train an MLP following the best architecture reported in Solodoch et al., 2023
    (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003370)
    Optionally standardise locally and keep trend/seasonal components.

    section: latitude to study [26N, 30S, 55S, 60S]
    remove_tr_ssn: remove trend and monthly seasonal components from both inputs and moc strength
    standardise: "solodoch" - global mean and temporal std
                 "ours" - local mean (standardise across each feature in the final dataset)
    splits: array of two proportions - end of training data and end of validation data
    device: for potential gpu acceleration
    """
    pp_data, moc, split_trainval, split_valtest = get_data(section, remove_tr_ssn, standardise, splits)
    # split into train/val/test
    X_train = pp_data[:split_trainval].squeeze(-1)
    X_val = pp_data[split_trainval:split_valtest].squeeze(-1)
    X_test = pp_data[split_valtest:].squeeze(-1)
    y_train = moc[:split_trainval]
    y_val = moc[split_trainval:split_valtest]
    y_test = moc[split_valtest:]

    if standardise == "ours":
        # local standardisation
        m = t.Tensor(np.nanmean(X_train, axis=0))
        s = t.Tensor(np.nanstd(X_train, axis=0))
        X_train = (X_train - m[..., :]) / s[..., :]
        X_val = (X_val - m[..., :]) / s[..., :]
        X_test = (X_test - m[..., :]) / s[..., :]

    # train model
    model = MLP(n_features=X_train.shape[-1],
                hidden_dim_width=1).to(device)
    model, train_loss, val_loss = train.train_model(model, X_train, y_train, X_val=X_val, y_val=y_val, early_stopping=False)
    
    # grab metrics and predictions
    y_pred = model(X_test.to(device))
    test_loss = F.mse_loss(y_pred.squeeze(-1), y_test.to(device)).item()
    rmse = np.sqrt(test_loss)
    mape = (t.mean(t.abs((y_test.to(device) - y_pred.squeeze(-1))/y_test.to(device)))*100).item()

    ixs = (t.abs(y_test) > 0.5).nonzero().flatten()
    y_test_ = y_test[ixs]
    y_pred_ = y_pred.squeeze(-1)[ixs].cpu()
    mape_ = t.mean(t.abs((y_test_ - y_pred_)/y_test_)*100).item()

    X = t.cat([X_train, X_val, X_test], dim=0)
    Y = t.cat([y_train, y_val, y_test], dim=0)
    full_pred = model(X.to(device)).squeeze(-1)
    full_pred = full_pred.detach().cpu().numpy()
    
    return rmse, mape, mape_, full_pred, moc