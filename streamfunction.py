import os
import sys
import logging
import xarray as xr
import numpy as np
import basin_masks
import matplotlib.pyplot as plt

user_home_dir = os.getcwd()
sys.path.append(os.path.join(user_home_dir, "ECCOv4-py"))
import ecco_v4_py as ecco

import warnings

warnings.filterwarnings("ignore")

from functools import reduce

from basin_masks import *

def _format_lat_lon(value):
    """Format latitude or longitude with N/S or E/W suffix."""
    if value < 0:
        return f"{abs(value)}S"
    else:
        return f"{value}N"


def calc_streamfunction(
    latitude,
    longitudes,
    geom_fp,
    ecco_grid_fp,
    output_path,
    zonal_average=False,
    recalculate=False,
):

    if zonal_average is True:
        add = "zonal_average_"
    else:
        add = ""

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{_format_lat_lon(latitude)}",
        f"PSI_{add}{_format_lat_lon(latitude)}.nc",
    )
    if os.path.exists(output_file) and not recalculate:
        logging.info(
            f"Streamfunction calculation skipped; {output_path} already exists."
        )
        return False  # Indicate that the function did not perform the calculation

    # Load the Ecco Variables
    nc_files = os.path.join(ecco_grid_fp, "*.nc")
    grid_var = xr.open_mfdataset(
        nc_files, data_vars="minimal", coords="minimal", compat="override"
    )
    grid = xr.open_dataset(geom_fp)
    ds = xr.merge((grid_var, grid))

    # Load the longitude section
    longitude_sections = np.split(longitudes, np.where(np.diff(longitudes) > 1)[0] + 1)

    masks_W = []
    masks_S = []

    for index, section in enumerate(longitude_sections):
        maskC, maskW, maskS = ecco.get_section_line_masks(
            [section[0], latitudes[0]], [section[-1], latitudes[0]], ds
        )
        masks_W.append(maskW)
        masks_S.append(maskS)

        if len(longitude_sections) > 1:
            PSI = ecco.calc_section_stf(
                ds,
                maskW=maskW,
                maskS=maskS,
                section_name=f"PSI at {latitudes[0]} latitude",
                zonal_average=zonal_average,
            ).compute()
            PSI.to_netcdf(
                os.path.join(
                    OUTPUT_DIR,
                    f"{_format_lat_lon(latitude)}",
                    f"PSI_{add}{_format_lat_lon(latitude)}_section_{index}.nc",
                )
            )

    maskS_tot = reduce(lambda x, y: x | y, masks_S)
    maskW_tot = reduce(lambda x, y: x | y, masks_W)

    PSI = ecco.calc_section_stf(
        ds,
        maskW=maskW_tot,
        maskS=maskS_tot,
        section_name=f"PSI at {latitudes[0]} latitude",
        zonal_average=zonal_average,
    ).compute()
    PSI.to_netcdf(output_file)
    logging.info(f"Streamfunction calculation completed and saved to {output_path}.")

    return True


def get_PSI_at_max_density_level(PSI, max=True):
    PSI_mean = np.abs(PSI["psi_moc"].mean("time"))
    if max is True:
        max_index = PSI_mean.argmax(dim="k")
    else:
        max_index = PSI_mean.argmin(dim="k")
    return PSI.isel(k=max_index)


def plot_depth_stf_vs_time(stf_ds, label, param):
    fig = plt.figure(figsize=(18, 6))

    # Time evolving
    plt.subplot(1, 4, (1, 3))
    time_edge_extrap = np.hstack(
        (
            stf_ds["time"].values[0] - (0.5 * np.diff(stf_ds["time"].values[0:2])),
            stf_ds["time"].values[:-1] + (0.5 * np.diff(stf_ds["time"].values)),
            stf_ds["time"].values[-1] + (0.5 * np.diff(stf_ds["time"].values[-2:])),
        )
    )
    Z_edge_extrap = np.hstack(
        (
            np.array([0]),
            stf_ds["Z"].values[:-1] + (0.5 * np.diff(stf_ds["Z"].values)),
            np.array([-6134.5]),
        )
    )
    plt.pcolormesh(time_edge_extrap, Z_edge_extrap, stf_ds[param].T)
    plt.title("ECCOv4r4\nOverturning streamfunction across latitude %s [Sv]" % label)
    plt.ylabel("Depth [m]")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    cb = plt.colorbar()
    cb.set_label("[Sv]")

    plt.subplot(1, 4, 4)
    plt.plot(stf_ds[param].mean("time"), stf_ds["Z"])
    plt.title("ECCOv4r4\nTime mean streamfunction %s" % label)
    plt.ylabel("Depth [m]")
    plt.xlabel("[Sv]")
    plt.grid()
    plt.show()


def plot_2D_streamfunction(stf_ds, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(stf_ds["time"], stf_ds["psi_moc"])
    plt.xlabel("Time")
    plt.ylabel("PSI in layer with maximal density level")
    if title is None:
        title = "PSI Streamfunction"
    plt.title(title)
    plt.grid(True)
    plt.show()

def get_gridllc0090_mask(target_latitude, longitudes, ds):

    FLIPPED_TILES = [7,8,9,10,11,12,13]
    
    # Split up the separate sections for the longitude list
    longitude_sections = np.split(longitudes, np.where(np.diff(longitudes) > 1)[0] + 1)
    
    # Create an empty mask
    mask = xr.DataArray(data=0, dims=ds['YC'].dims, coords=ds['YC'].coords)

    # Iterate over each "tile" of the grid 
    for tile_num in ds['tile'].values:

        if tile_num+1 in FLIPPED_TILES:
            latitude_dim = 'j'
        else:
            latitude_dim = 'i'

        tile_lats = ds['YC'].sel(tile=tile_num)
        tile_lons = ds['XC'].sel(tile=tile_num)
    
        tile_mask = abs(tile_lats - target_latitude) 
        zeros = xr.zeros_like(tile_mask)
    
        first_col = tile_lats.isel(**{latitude_dim: 0})
        lowest_lat = first_col.values.max()
        highest_lat = first_col.values.min()
    
        if max(lowest_lat, highest_lat) > target_latitude and min(lowest_lat, highest_lat) < target_latitude:
        
            row_to_mask = int(tile_mask.isel(**{latitude_dim: 0}).argmin().values)
        
            for longitude in longitude_sections:
                if tile_num+1 in FLIPPED_TILES:
                    column_mask = (tile_lons[:, row_to_mask] >= longitude[0]) & (tile_lons[:, row_to_mask] <= longitude[-1])
                    zeros.loc[column_mask, row_to_mask]= 1
                else:
                    column_mask = (tile_lons[row_to_mask, :] >= longitude[0]) & (tile_lons[row_to_mask,:] <= longitude[-1])
                    zeros.loc[row_to_mask, column_mask]= 1
    
        # Update the mask DataArray for the current face
        mask.loc[dict(tile=tile_num)] = zeros.astype(int)
    return mask

def calculate_streamfunction_JANKY(ds, basin, latitude, time, geom_fp):
    longitudes = basin_masks.get_longitudes_at_latitude(latitude, basin, geom_fp)

    mask = get_gridllc0090_mask(latitude, longitudes, ds)

    velocity = ds['VVELMASS'].isel(time = time)
    delta_x = ds['dxG']
    delta_z = ds['drF']

    weighted_velocity = velocity * delta_x * delta_z

    mask = mask.rename({'j' : 'j_g'}) 
    weighted_velocity_at_lat = mask * weighted_velocity

    lon_by_depth = []
    for i in range(13):    
        arr = weighted_velocity_at_lat.isel(tile = i).fillna(0).values
        nonzero_ind = np.nonzero(arr)

        unique_vals = np.unique(nonzero_ind[0]), np.unique(nonzero_ind[1])
        if len(unique_vals[0]) == 1:
            nonzero_vals = arr[unique_vals[0][0], : , : ]
        elif len(unique_vals[1]) == 1:
            nonzero_vals = arr[ : , unique_vals[1][0], : ]
        else:
            print('No nonzero vals')
            continue

        print(nonzero_vals.shape)
        lon_by_depth.append(nonzero_vals)

    lon_by_depth = np.concatenate(lon_by_depth, axis = 0)
    streamfunction = -1 * lon_by_depth.sum(axis = 0)[ : : -1].cumsum()
    depth_level = np.abs(streamfunction).argmax()

    strength_sv = round(streamfunction[depth_level] / (10 ** 6))

    return strength_sv

if __name__ == "__main__":

    # Replace with local connection to Google Drive
    ROOT = "H:/.shortcut-targets-by-id/1wvJjD0RMTujKYaXQapEiGk-Mx03_KSin"
    VOLUME_DIR = os.path.join(
        ROOT,
        "GTC/solodoch_data_full/ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4",
    )
    OUTPUT_DIR = os.path.join(ROOT, "GTC/streamfunctions_ecco")

    geom_fp = (
        "ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"
    )

    basins = basin_masks.get_available_basin_names_solodoch()
    for basin in basins:
        latitudes = basin_masks.get_lats_of_interest_solodoch(basin)
        for latitude in latitudes:
            print("getting the latitudes")
            longitudes = basin_masks.get_longitudes_at_latitude(
                latitude, basin, geom_fp
            )
            calc_streamfunction(
                latitude,
                longitudes,
                geom_fp,
                VOLUME_DIR,
                OUTPUT_DIR,
                zonal_average=True,
                recalculate=False,
            )
