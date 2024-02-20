%matplotlib inline
%config InlineBackend.figure_format='retina'

import matplotlib.pyplot as plt
import cmocean as cm
import xarray as xr
import numpy as np
import IPython.display

import cosima_cookbook as cc

import cosima_cookbook as cc

experiment = "01deg_jra55v140_iaf_cycle4"


variable = 'sea_level'

sea_level = cc.querying.getvar(experiment, variable, session, start_time="1990-01-01", end_time="2019-01-01", frequency="1 monthly")
sea_level_SO_monthly = sea_level.sel(yt_ocean = slice(-82, -30))

sea_level_SO_monthly.to_netcdf("./sea_level_SO_monthly.nc")
