from scipy.signal import butter, filtfilt # low-pass filter
from statsmodels.tsa.seasonal import seasonal_decompose # deseason timeseries
from sklearn.preprocessing import StandardScaler

import xarray as xr

import sys

def apply_preprocessing(dataset, remove_trend = True, remove_season = True, standardize = True, lowpass = False):

    """
    Preprocessing function for covariates, including de-trending, de-seasonalizing, standardization,
    and applying a low-pass filter to remove effect of short timescale variations. All operations 
    act on the longitudinal timeseries and for each covariate independently.

    Parameters
    ----------
    dataset : xarray.dataset
        xarray dataset of covariates with time, latitude, and longitude as coordinates (in that order)
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

    # Instantiating a new array like the original to populate with preprocessed values
    preprocessed_array = xr.full_like(dataset, 0)
    dims = ('time', 'latitude', 'longitude')

    for k in dataset.keys():
        var = dataset[k].values.squeeze()

        var_deseason = seasonal_decompose(var, model = 'additive', period = 12, extrapolate_trend = 6)

        new_var = var_deseason.resid # extract residual - the variation not captured by seasonality or long-term trend

        if not remove_season:
            new_var = new_var + var_deseason.seasonal # add back in seasonal component
        if not remove_trend:
            new_var = new_var + var_deseason.trend # add back in trend component

        if standardize:
            scaler = StandardScaler()
            new_var = scaler.fit_transform(new_var)

        if lowpass:
            cutoff = 2.0 # cutoff is 2.0 for 6-month LPF
            fs = 12.0 # freq of sampling is 12.0 times in a year
            order = 6 # order of polynomial component of filter

            b, a = butter(order, cutoff, fs = fs, btype = 'low', analog = False)
            new_var = filtfilt(b, a, new_var, axis = 0) # apply on each lon timeseries separately

        new_var = new_var.reshape(new_var.shape[0], 1, new_var.shape[1])
        preprocessed_array[k] = (dims, new_var)

    return preprocessed_array
        

if __name__ == '__main__':
    data_fp = 'solodoch_data_minimal/26N.nc'

    ecco_data = xr.open_dataset(data_fp)
    
    print(apply_preprocessing(ecco_data, remove_trend = True, remove_season = True, standardize = True, lowpass = True))