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
