import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import sys

def sliding_window_min_max_mix(ds_xr, window_size = 5, ignore_first_last = 10, depth_coord_name = 'depth', remove_above_depth = -2000):

    """
    Assumes input dataset has shape [# depth levels x time] and each element is the streamfunction
    using the depth level as the upper bound for the integral.
    """

    # Downsample data to monthly means
    ds_xr_monthly = ds_xr.resample({'time' : 'M'}).mean()
    ds_xr_monthly = ds_xr_monthly.transpose(depth_coord_name, 'time')

    # Take a 5-month window mean for streamfunction values at each depth & time
    ds_window_mean = np.nanmean(sliding_window_view(ds_xr_monthly.values, window_shape = window_size, axis = 1), axis = 2)

    # Extract the minimum, maximum, and (intermediate) zero streamfunction depths
    idx_min = np.nanargmin(ds_window_mean, axis = 0)
    idx_max = np.nanargmax(ds_window_mean, axis = 0)

    #  ignore the bottom/top depths, which are always 0
    ds_window_mean[-1 * ignore_first_last : ] = np.nan
    ds_window_mean[ : ignore_first_last] = np.nan
    idx_zero = np.nanargmin(np.abs(ds_window_mean), axis = 0)

    # Grab actual depth values according to the dataset's depth layers
    depths = ds_xr[depth_coord_name].values.astype(float)

    min_depths = depths[idx_min]
    max_depths = depths[idx_max]
    zero_depths = depths[idx_zero]

    # Removing values that are too high to be lowest cell
    if remove_above_depth is not None:
        min_depths[min_depths > remove_above_depth] = np.nan

    return min_depths, max_depths, zero_depths

if __name__ == '__main__':
    import pickle
    import xarray as xr
    import os

    data_home = '/Users/emiliolr/Google Drive/My Drive/GTC'

    # Grabbing the time values for ECCO data
    time = xr.open_dataset(os.path.join(data_home, 'ecco_data_minimal', '26N.nc')).time.values

    # Loading density-space ECCO data
    ecco_streamfunction_density_fp = os.path.join(data_home, 'ecco_data_minimal', '26N_streamfunction_density.pickle')

    with open(ecco_streamfunction_density_fp, 'rb') as f:
        ecco_streamfunction_density = pickle.load(f)

    # Put into an xarray dataset for convenience
    ecco_strf = xr.Dataset(data_vars = {'streamfunction' : (['density_layer', 'time'], ecco_streamfunction_density)},
                         coords = {'time' : time,
                                   'density_layer' : np.arange(0, ecco_streamfunction_density.shape[0], 1)})
    ecco_strf = ecco_strf.streamfunction

    print(sliding_window_min_max_mix(ecco_strf, depth_coord_name = 'density_layer')[0].shape)
