from typing import Callable, Any, List, Literal, Tuple, Optional
import numpy as np
from numpy.lib.stride_tricks import as_strided
import xarray as xr
from scipy.signal import butter, filtfilt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import torch as t
import sys

def reshape_inputs(data: xr.core.dataset.Dataset,
                   keep_coords: List=["time", "latitude", "longitude"],
                   avg_time_window: Optional[int]=None,
                   history: Optional[int]=None,
                   data_vars: List=["SSH", "SST", "SSS", "OBP", "ZWS"],
                   return_pt: bool=False,
                   verbose: bool=True) -> np.ndarray | t.Tensor:
    """
    Read in the original input dataset, with coordinates "time", "latitude", "longitude",
    and data variables "SSH", "SST", "SSS", "OBP", "ZWS".

    Return a numpy array of any subset of the data variables, optionally averaged over any coordinates or including history.
    Can also return a pytorch tensor.

    data: original xarray dataset - see solodoch_data_minimal in google drive.
    keep_coords: coordinate axes to be kept. others will be averaged over and collapsed.
    avg_time_window: if time is not included in keep_coords, optionally choose a lag over which to average.
    history: include a new axis for history (useful if we want to convolve over past values for example)
    data_vars: data variables to be kept.
    return_pt: if true, returns a pytorch tensor (cpu!) instead of a numpy array.
    """

    def moving_average(data: np.ndarray,
                       lag: int) -> np.ndarray:
        """
        Calculate a moving average over the time dimension.

        data: subset of values from original dataset. intermediate step of reshape_inputs.
        lag: lag over which to average.
        """
        # axis order is guaranteed due to reshape_inputs
        n_times, n_lats, n_lons, n_features = data.shape
        D = n_times - lag + 1
        view_shape = (D, lag, n_lats, n_lons, n_features)
        s = data.strides; strides = (s[0], s[0], s[1], s[2], s[3])
        data_ = as_strided(data, shape=view_shape, strides=strides)
        return data_.mean(axis=1).squeeze(axis=1)

    coords = ["time", "latitude", "longitude"]
    data = data[coords + data_vars].to_array().values
    data = data.transpose(1, 2, 3, 0)
    for ax in coords:
        if ax not in keep_coords:
            if ax == "time" and avg_time_window != None:
                data = moving_average(data, avg_time_window)
            else:
                data = np.nanmean(data, axis=coords.index(ax))
            coords = [c for c in coords if c != ax]

    if history != None:
        if "time" not in keep_coords: raise Exception("Error. 'time' must be in keep_coords in order to use history.")
        coords = ["time", "history"] + coords[1:]
        n_times = data.shape[0]
        if history > n_times: raise ValueError("Desired history is longer than the full time series.")
        view_shape = (n_times-history+1, history, *data.shape[1:])
        s = data.strides[0]
        data = as_strided(data, shape=view_shape, strides=(s, s, *data.strides[1:]))

    if verbose:
        print(f"axes: {coords + ['feature']}")
        print(f"variables: {data_vars}")
        print(f"shape: {data.shape}")
    return t.Tensor(data) if return_pt else data

def custom_seasonal_decompose(mode, data, dims = ('time', 'latitude', 'longitude'), train_pct = 0.7):

    """
    A wrapper function over code from Sharan's model training notebook. Allows for additive
    timeseries decomposition using only info from the training portion of the timeseries for fitting
    the trend line and seasonal components.
    """

    # Extract important variables
    vars = ['SSH', 'SST', 'SSS', 'OBP', 'ZWS']
    times = data.time.values
    latitudes = data.latitude.values
    train_split = int(train_pct * data.sizes['time'])

    if mode == 'inputs':
        longitudes = data.longitude.values

        train_data = data.isel(time = slice(0, train_split)).copy(deep = True)
        test_data = data.isel(time = slice(train_split, data.sizes['time'])).copy(deep = True)

        # Calculate linear trend
        slopes = np.empty((len(vars), len(longitudes)))
        intercepts = np.empty_like(slopes)
        for j in range(len(vars)):
            var = vars[j]
            for i in range(len(longitudes)):
                series = train_data[var].isel(latitude = 0, longitude = i)
                y = series[series.notnull()]
                x = np.arange(len(series))[series.notnull()]
                if len(y) > 1:
                    slope, intercept = np.polyfit(x, y, deg = 1)
                    slopes[j, i] = slope
                    intercepts[j, i] = intercept
                else:
                    slopes[j, i] = np.nan
                    intercepts[j, i] = np.nan

        # Subtract linear trend from train/test data
        for j in range(len(vars)):
            for i in range(len(longitudes)):
                trend = (slopes[j, i] * np.arange(len(times))) + intercepts[j, i]

                train_data[vars[j]].loc[ : , latitudes[0], longitudes[i]] -= trend[ : train_split]
                test_data[vars[j]].loc[ : , latitudes[0], longitudes[i]] -= trend[train_split : ]

        # Calculate monthly seasonal components from training data - remove from train/test seperately
        monthly_means = train_data.groupby('time.month').mean()
        train_data = train_data.groupby('time.month') - monthly_means
        test_data = test_data.groupby('time.month') - monthly_means

        inputs = xr.concat((train_data, test_data), dim = 'time')

        return inputs
    elif mode == 'outputs':
        # Load output data
        moc_train = data.moc.squeeze().values[ : train_split]
        moc_test = data.moc.squeeze().values[train_split : ]

        # Calculate and remove linear trend
        y = moc_train
        x = np.arange(train_split)

        slope, intercept = np.polyfit(x, y, deg = 1)
        trend = (slope * np.arange(len(data.moc))) + intercept

        moc_train -= trend[ : train_split]
        moc_test -= trend[train_split : ]

        # Calculate and remove monthly seasonal component
        moc_reshaped = moc_train[ : (moc_train.shape[0] // 12) * 12].reshape(-1, 12)
        monthly_means = moc_reshaped.mean(axis = 0)

        full_moc = np.concatenate([moc_train, moc_test])
        monthly_means = monthly_means[np.newaxis, : ].repeat(full_moc.shape[0] // 12, axis = 0)
        monthly_means = monthly_means.reshape(-1)
        full_moc -= monthly_means

        outputs = xr.full_like(data, 0)
        outputs['moc'] = ((dims[0], dims[1]), full_moc.reshape(-1, 1))

        return outputs
    else:
        raise ValueError('"mode" must be one of ["inputs", "outputs"]')


def apply_preprocessing(dataset, mode = 'inputs', remove_trend = True, remove_season = True,
                        standardize = True, lowpass = False, train_pct = None):

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
    train_pct : float or None
        if not None, only the first "train_pct" percent of the timeseries is used
        for fitting the preprocessing methods

    Returns
    -------
    preprocessed_array : xarray.dataset
        xarray dataset with the same format as input, but with preprocessed covariates
    """

    avail_modes = ['inputs', 'outputs']
    assert mode in avail_modes, f'mode argument must be one of {avail_modes}'

    if mode == 'inputs':
        dims = ('time', 'latitude', 'longitude')

        if 'latitude' not in dataset.dims:
            dims = (dims[0], dims[2])
    elif mode == 'outputs':
        dims = ('time', 'latitude')

        if 'latitude' not in dataset.dims:
            dims = (dims[0], )

    # Making sure the dataset has the expected ordering or coordinates
    if mode == 'inputs':
        dataset = dataset.transpose(*dims)
    elif mode == 'outputs':
        dataset = dataset.transpose(*dims)

    # Instantiating a new array like the original to populate with preprocessed values
    preprocessed_array = xr.full_like(dataset, 0)

    # New behavior: fit standardization/trend + seasonality ONLY on train set to
    #  ensure no data leakage
    if train_pct is not None:
        if remove_trend and remove_season:
            new_dataset = custom_seasonal_decompose(mode, dataset, dims = dims, train_pct = train_pct)
        if standardize and mode == 'inputs':
            if (not remove_trend) or (not remove_season):
                new_dataset = dataset.copy(deep = True)

            #  loop through variables and standardize them independently for each longitude
            for var in new_dataset:
                dataset_np = new_dataset[var].squeeze().to_numpy()
                train_split = int(train_pct * dataset.sizes['time'])

                scaler = StandardScaler()
                train_vals = scaler.fit_transform(dataset_np[ : train_split])
                test_vals = scaler.transform(dataset_np[train_split : ])

                all_vals = np.concatenate((train_vals, test_vals), axis = 0)[ : , np.newaxis, : ]

                new_dataset[var] = (dims, all_vals)

        preprocessed_array = new_dataset.copy(deep = True) if (remove_trend or remove_season or standardize) else dataset.copy()

    # Old behavior: fit standardization/trend + seasonality on ALL data indiscriminately
    else:
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
                if new_var.ndim == 1:
                    new_var = new_var.reshape(-1, 1)
                new_var = scaler.fit_transform(new_var)

            # Adding back in latitude dimension that got squeezed out
            if mode == 'inputs' and 'latitude' in dataset.dims:
                if new_var.ndim == 1:
                    new_var = new_var.reshape(-1, 1)
                new_var = new_var.reshape(new_var.shape[0], 1, new_var.shape[1])
            elif mode == 'outputs' and 'latitude' in dataset.dims:
                new_var = new_var.reshape(new_var.shape[0], 1)

            preprocessed_array[k] = (dims, new_var)

    return preprocessed_array

def align_inputs_outputs(inputs, outputs, date_range = ('1992-01-16', '2015-12-16'), ecco = True):

    """
    Align input and output dataset date ranges and latitudes to prepare for preprocessing.
    If working with ECCO, this will also extract the MOC strength variable.

    Parameters
    ----------
    inputs : xarray.dataset
        dataset of surface variables
    outputs : xarray.dataset
        dataset of MOC strength
    date_range : tuple
        the start and end date for extraction
    ecco : boolean
        is this data from ecco?

    Returns
    -------
    inputs, outputs : xarray.dataset, xarray.dataset
        aligned input and output datasets
    """

    inputs = inputs.sel(time = slice(*date_range))
    outputs = outputs.sel(time = slice(*date_range))

    if ecco:
        outputs = outputs.moc.to_dataset()
        outputs = outputs.rename({'lat' : 'latitude'})

    return inputs, outputs

def reg_results_txt(grid_search, fp, zonal_avgs, no_zonal_avgs, test_metrics, intercept_first = True, save_weights = True):

    """
    Helper function to write linear regression results to a text file.

    Parameters
    ----------
    grid_search : sklearn.CVGridSearch
        sklearn grid search object
    fp : string
        the filepath to write to
    zonal_avgs : list
        the list of variables for which zonal averages WERE taken
    no_zonal_avgs : list
        the list of variables for which zonal averages WERE NOT taken
    test_metrics : dictionary
        a dictionary of all test metrics to save
    intercept_first : boolean
        is the intercept the first weight or the last weight of the fitted model?
    save_weights : boolean
        should we save the model weights?

    Returns
    -------
    None
    """

    with open(fp, 'w') as f:
        f.write(f'Best hyperparameter values: {grid_search.best_params_}\n\n')

        if save_weights:
            model_weights = grid_search.best_estimator_.model.params
            data_vars = ['Intercept'] + data_vars if intercept_first else data_vars + ['Intercept']
            named_weights = {name : weight_val for name, weight_val in zip(data_vars, model_weights)}

            for k, v in named_weights.items():
                f.write(f'{k} weight: {round(v, 3)}\n')
        else:
            f.write(f'All longitudes used for: {" ".join(no_zonal_avgs)}\n' if len(no_zonal_avgs) > 0 else '')
            f.write(f'Zonal averages used for: {" ".join(zonal_avgs)}\n' if len(zonal_avgs) > 0 else '')

        f.write('\n')

        for k, v in test_metrics.items():
            f.write(f'{k}: {v}\n')

def custom_MAPE(y_test, y_pred, threshold = 0, return_num_discarded = False):

    """
    A custom mean absolute percentage error metric that ignores very small values.

    Parameters
    ----------
    y_test : numpy.array
        observed values on the test set (ground truth)
    y_pred : numpy.array
        predicted values for the test set
    threshold : float
        only keep observations that are more extreme than +/- threshold
    return_num_discarded : boolean
        should we return the number of samples that were discarded using
        this threshold?

    Returns
    -------
    mape, initial_len : float, integer
    OR
    mape : float
        the MAPE and, optionally, the number of samples discarded
    """

    # Get starting length of test set
    initial_len = len(y_test)

    # Mask out values less extreme than threshold
    mask = np.abs(y_test) > threshold
    y_test, y_pred = y_test[mask], y_pred[mask]
    new_len = len(y_test)

    # Calculate the MAPE on this subset of the test set - a small number is added
    #  to the denominator to avoid dividing by zero
    mape = np.mean(np.abs((y_pred - y_test)) / (np.abs(y_test) + 1e-4))

    if return_num_discarded:
        return mape, initial_len - new_len

    return mape
