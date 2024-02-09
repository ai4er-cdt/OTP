from typing import Callable, Any, List, Literal, Tuple, Optional
import numpy as np
from numpy.lib.stride_tricks import as_strided
import xarray as xr
from scipy.signal import butter, filtfilt 
from statsmodels.tsa.seasonal import seasonal_decompose 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def reshape_inputs(data: xr.core.dataset.Dataset, 
                   keep_coords: List=["time", "latitude", "longitude"],
                   lag: Optional[int]=None, 
                   data_vars: List=["SSH", "SST", "SSS", "OBP", "ZWS"]) -> np.ndarray:
    """
    Read in the original input dataset, with coordinates "time", "latitude", "longitude",
    and data variables "SSH", "SST", "SSS", "OBP", "ZWS".

    Return a numpy array of any subset of the data variables, optionally averaged over any coordinates.

    data: original xarray dataset - see solodoch_data_minimal in google drive.
    keep_coords: coordinate axes to be kept. others will be averaged over and collapsed.
    lag: if time is not included in keep_coords, optionally choose a lag over which to average.
    data_vars: data variables to be kept.

    TODO: add support for tensors.
    """

    def moving_average(data: np.ndarray,
                       lag: int) -> np.ndarray:
        """
        Calculate a moving average over the time dimension.

        data: subset of values from original dataset. intermediate step of reshape_inputs.
        lag: lag over which to average.
        """
        # axis order is guaranteed due to reshape_inputs
        n_features, n_times, n_lats, n_lons = data.shape
        D = n_times - lag + 1
        view_shape = (n_features, D, lag, n_lats, n_lons)
        s = data.strides; strides = (s[0], s[1], s[1], s[2], s[3])
        data_ = as_strided(data, shape=view_shape, strides=strides)
        return data_.mean(axis=2).squeeze(axis=2)
    
    coords = ["time", "latitude", "longitude"]
    data = data[coords + data_vars].to_array().values
    for ax in coords:
        if ax not in keep_coords: 
            if ax == "time" and lag != None: 
                data = moving_average(data, lag)
            else:
                data = data.mean(axis=coords.index(ax)+1)
            coords = [c for c in coords if c != ax]
    data = data.transpose((*(np.arange(len(coords))+1), 0))
    print(f"axes: {coords + ['feature']}")
    print(f"variables: {data_vars}")
    print(f"shape: {data.shape}")    
    return data

def apply_preprocessing(dataset, mode = 'inputs', remove_trend = True, remove_season = True, standardize = True, lowpass = False):

    """
    Preprocessing function for covariates, including de-trending, de-seasonalizing, standardization,
    and applying a low-pass filter to remove effect of short timescale variations. All operations 
    act on the longitudinal timeseries and for each covariate independently.

    Parameters
    ----------
    dataset : xarray.dataset
        xarray dataset of covariates with time, latitude, and longitude as coordinates (in that order)
    mode : string
        either 'inputs' for covariates or 'outputs' for streamfunction
    remove_trend : boolean
        should the long-term trend be removed?
    remove_season : boolean
        should the seasonal trend be removed?
    standardize : boolean
        should we apply standardization, i.e., subtract time-mean and divide by standard 
        deviation?
    lowpass : boolean
        should we apply a low-pass filter to the covariate timeseries?

    Returns
    -------
    preprocessed_array : xarray.dataset
        xarray dataset with the same format as input, but with preprocessed covariates 
    """

    avail_modes = ['inputs', 'outputs']
    assert mode in avail_modes, f'mode argument must be one of {avail_modes}'

    # Making sure the dataset has the expected ordering or coordinates
    if mode == 'inputs':
        dataset = dataset.transpose('time', 'latitude', 'longitude')
    elif mode == 'outputs':
        dataset = dataset.transpose('time', 'latitude')

    # Instantiating a new array like the original to populate with preprocessed values
    preprocessed_array = xr.full_like(dataset, 0)

    if mode == 'inputs':
        dims = ('time', 'latitude', 'longitude')
    elif mode == 'outputs':
        dims = ('time', 'latitude')

    for k in dataset.keys():
        var = dataset[k].values.squeeze()

        var_deseason = seasonal_decompose(var, model = 'additive', period = 12, extrapolate_trend = 6)
        new_var = var_deseason.resid # extract residual - the variation not captured by seasonality or long-term trend

        if not remove_season:
            new_var = new_var + var_deseason.seasonal # add back in seasonal component
        if not remove_trend:
            new_var = new_var + var_deseason.trend # add back in trend component

        if lowpass:
            cutoff = 2.0 # cutoff is 2.0 for 6-month LPF
            fs = 12.0 # freq of sampling is 12.0 times in a year
            order = 6 # order of polynomial component of filter

            b, a = butter(order, cutoff, fs = fs, btype = 'low', analog = False)
            new_var = filtfilt(b, a, new_var, axis = 0) # apply on each lon timeseries separately

        # Making sure to apply standardization last to ensure covariates have the right time-wise stats
        if standardize and mode == 'inputs':
            scaler = StandardScaler()
            new_var = scaler.fit_transform(new_var)

        # Adding back in latitude dimension that got squeezed out
        if mode == 'inputs':
            new_var = new_var.reshape(new_var.shape[0], 1, new_var.shape[1])
        elif mode == 'outputs':
            new_var = new_var.reshape(new_var.shape[0], 1)

        preprocessed_array[k] = (dims, new_var)

    return preprocessed_array