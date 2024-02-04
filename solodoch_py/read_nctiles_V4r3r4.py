import os
import numpy as np
import xarray as xr


def read_nctiles_V4r3r4(file_name, field_name, sampnum, zlev, ECCOv):
    if ECCOv == "V4r4":
        # Assuming you have a function read_nctiles_daily_V2 for V4r4
        addpath = "./ECCOv4r4Andrew"
        field_tiles_out = read_nctiles_daily_V2(file_name, field_name)
    elif ECCOv == "V4r3":
        # Read netCDF files for V4r3
        field_tiles_out = read_nctiles(file_name, field_name, sampnum, zlev)
        # Plug in zlev=1 for surface depth if it's a 4D field
        if zlev == 1:
            field_tiles_out = field_tiles_out.isel(Z=0)
    else:
        raise ValueError("Unfamiliar ECCO version descriptor")

    return field_tiles_out
