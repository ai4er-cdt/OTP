import os
import paths
import xarray as xr

from scripts.archive import streamfunction_LLC0090_utils as strf_ecco_utils, LLC0090_masking as strf_custom_utils


def calculate_streamfunction(
    latitude, basin_names, velocity_field="residual", save=True
):

    ecco_grid = xr.open_dataset(paths.geom_fp)

    if velocity_field == "residual":
        nc_files = os.path.join(paths.VELOCITY_NATIVE_GRID, "*.nc")
    elif velocity_field == "bolus":
        nc_files = os.path.join(paths.BOLUS_NATIVE_GRID, "*.nc")

    nc_ds = xr.open_mfdataset(
        nc_files, data_vars="minimal", coords="minimal", compat="override"
    )

    ds = xr.merge((nc_ds, ecco_grid))

    PSI = strf_ecco_utils.calc_meridional_stf(
        ds, [latitude], velocity_field, doFlip=True, basin_name=basin_names
    )

    if save is True:
        output_file = os.path.join(
            paths.STREAMFUNCTIONS_ECCO_OUTPUT,
            f"{strf_custom_utils.format_lat_lon(latitude)}",
            f"PSI_{velocity_field}{strf_custom_utils.format_lat_lon(latitude)}.nc",
        )
        PSI.to_netcdf(output_file)

    return PSI


if __name__ == "__main__":
    latitude = -60
    basin = None

    calculate_streamfunction(latitude, basin, velocity_field="bolus")
