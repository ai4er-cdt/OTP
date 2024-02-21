import os
import xarray as xr
import numpy as np

from scipy.interpolate import griddata


'''
Load the grid from v4r3's 'nctiles_grid files'
- Other file formats are 'straight', 'cube', 'compact'
'''
def grid_load(dirGrid, fileFormat="nctiles", omitNativeGrid=False):

    if omitNativeGrid:
        print("Native grid loading is omitted.")
        return None

    # Construct file path pattern based on fileFormat
    if fileFormat == "nctiles":
        filePathPattern = os.path.join(dirGrid, "*.nc")
    else:
        print("Unsupported file format:", fileFormat)
        return None

    # Use xarray to open a dataset; assuming NetCDF files for this example
    # TO CHECK -- 'nested' instead of 'by_coords', with specifically defined 'concat_dim'?
    ds = xr.open_mfdataset(filePathPattern, concat_dim='chunk', combine="nested")
    return ds


def convert2pcol(X, Y, A):
    # Define your original grid points and values
    points = np.array([X.values.ravel(), Y.values.ravel()]).T  # Original grid points
    values = A.values.ravel() # Original data values

    num_points = 100
    lon = np.linspace(-180, 180, num_points)
    lat = np.linspace(-90, 90, num_points)

    # Define the grid points where you want to interpolate
    grid_lon, grid_lat = np.meshgrid(lon, lat)

    # Interpolate onto the new grid
    return griddata(points, values, (grid_lon, grid_lat), method="linear")


def calc_UEVNfromUXVY(ds, grid, UVEL, VVEL):
    # first interpolate velocity to cell centers
    vel_c = grid.interp_2d_vectors({"X": UVEL, "Y": VVEL}, boundary="fill")

    # Compute East and North components using cos() and sin()
    u_east = vel_c["X"] * ds["CS"] - vel_c["Y"] * ds["SN"]
    v_north = vel_c["X"] * ds["SN"] + vel_c["Y"] * ds["CS"]
    return u_east, v_north