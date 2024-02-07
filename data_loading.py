from scipy.signal import butter # low-pass filter
from statsmodels.tsa.seasonal import seasonal_decompose # deseason timeseries
from sklearn.preprocessing import StandardScaler

import xarray as xr

import sys

def apply_preprocessing(dataset, remove_trend = True, remove_season = True, standardize = True, lowpass_mode = 'off'):

    valid_lowpass = ['off', 'before', 'after']
    assert lowpass_mode in valid_lowpass, f'Input lowpass_mode must be one of {valid_lowpass}'

    # Instantiating a new array like the original to populate with preprocessed values
    preprocessed_array = xr.full_like(dataset, 0)
    dims = ('time', 'latitude', 'longitude')

    for k in dataset.keys():
        var = dataset[k].values.squeeze()

        var_deseason = seasonal_decompose(var, model = 'additive', period = 12, extrapolate_trend = 6)

        new_var = var_deseason.resid

        if not remove_season:
            new_var = new_var + var_deseason.seasonal
        if not remove_trend:
            new_var = new_var + var_deseason.trend

        if standardize:
            scaler = StandardScaler()
            new_var = scaler.fit_transform(new_var)

        if lowpass_mode == 'before':
            pass
        elif lowpass_mode == 'after':
            pass

        new_var = new_var.reshape(new_var.shape[0], 1, new_var.shape[1])
        preprocessed_array[k] = (dims, new_var)

    return preprocessed_array
        

if __name__ == '__main__':
    data_fp = 'solodoch_data_minimal/26N.nc'

    ecco_data = xr.open_dataset(data_fp)
    
    print(apply_preprocessing(ecco_data, remove_trend = False, remove_season = False, standardize = True))