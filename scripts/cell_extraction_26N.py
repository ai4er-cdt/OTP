import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def sliding_window_min_max_mix(ds_xr, ignore_first_last = 10, depth_coord_name = 'depth'):

    """
    Assumes input dataset has shape [# depth levels x time] and each element is the streamfunction
    using the depth level as the upper bound for the integral.
    """

    # Downsample data to monthly means
    ds_xr_monthly = ds_xr.resample({'time' : 'M'}).mean()

    # Take a 5-month window mean for streamfunction values at each depth & time
    ds_window_mean = np.nanmean(sliding_window_view(ds_xr_monthly.values, window_shape = 5, axis = 1), axis = 2)

    # Extract the minimum, maximum, and (intermediate) zero streamfunction depths
    idx_min = np.nanargmin(ds_window_mean, axis = 0)
    idx_max = np.nanargmax(ds_window_mean, axis = 0)

    #  ignore the bottom/top depths, which are always 0
    ds_window_mean[-1 * ignore_first_last : ] = np.nan
    ds_window_mean[ : ignore_first_last] = np.nan
    idx_zero = np.nanargmin(np.abs(ds_window_mean), axis = 0)

    # Grab actual depth values according to the dataset's depth layers
    depths = ds_xr[depth_coord_name].values

    min_depths = depths[idx_min]
    min_depths[min_depths > -2000] = np.nan # removing values that are too high to be lowest cell
    max_depths = depths[idx_max]
    zero_depths = depths[idx_zero]

    return min_depths, max_depths, zero_depths
