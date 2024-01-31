import sys

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("ECCOv4-py")
import ecco_v4_py as ecco
from ecco_download import *

def get_longitudes_at_latitude(lat, basin):

    """
    Return the relevant longitudes for a given latitude and basin.

    Parameters
    ----------
    lat : float 
        latitude of interest
    basin : string
        basin name, corresponding to the available basins in ecco.get_available_basin_names()

    Returns
    -------

    """

    pass

if __name__ == '__main__':
    print('hello')