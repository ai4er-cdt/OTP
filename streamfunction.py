import os
import logging
import xarray as xr
import numpy as np
import basin_masks
import ecco_v4_py as ecco

import warnings
warnings.filterwarnings('ignore')

from functools import reduce


def calc_streamfunction(latitude, longitudes, geom_fp, ecco_grid_fp, output_path, recalculate=False):

    output_file = os.path.join(OUTPUT_DIR, f"PSI_{latitude}.nc")
    if os.path.exists(output_file) and not recalculate:
        logging.info(f"Streamfunction calculation skipped; {output_path} already exists.")
        return False  # Indicate that the function did not perform the calculation

    # Load the Ecco Variables
    nc_files = (os.path.join(ecco_grid_fp, '*.nc'))
    grid_var = xr.open_mfdataset(nc_files, data_vars='minimal', coords='minimal', compat='override')
    grid = xr.open_dataset(geom_fp)
    ds = xr.merge((grid_var, grid))

    # Load the longitude section
    longitude_sections = np.split(longitudes, np.where(np.diff(longitudes) > 1)[0] + 1)

    masks_W = []
    masks_S = []

    for section in longitude_sections:
        maskC, maskW, maskS = ecco.get_section_line_masks([section[0], latitudes[0]], [section[-1], latitudes[0]], ds)
        masks_W.append(maskW)
        masks_S.append(maskS)

    maskS_tot = reduce(lambda x, y: x | y, masks_S)
    maskW_tot = reduce(lambda x, y: x | y, masks_W)

    PSI = ecco.calc_section_stf(
        ds, maskW=maskW_tot, maskS=maskS_tot, section_name=f"PSI at {latitudes[0]} latitude").compute()
    PSI.to_netcdf(output_file)
    logging.info(f"Streamfunction calculation completed and saved to {output_path}.")

    return True


if __name__ == '__main__':

    # Replace with local connection to Google Drive
    ROOT = "H:/.shortcut-targets-by-id/1wvJjD0RMTujKYaXQapEiGk-Mx03_KSin"
    VOLUME_DIR = os.path.join(ROOT, "GTC/solodoch_data_full/ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4")
    OUTPUT_DIR = os.path.join(ROOT, "GTC/streamfunctions_ecco")

    geom_fp = 'ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc'

    basins = basin_masks.get_available_basin_names_solodoch()
    for basin in basins:
        latitudes = basin_masks.get_lats_of_interest_solodoch(basin)
        for latitude in latitudes:
            print('getting the latitudes')
            longitudes = basin_masks.get_longitudes_at_latitude(latitude, basin, geom_fp)
            calc_streamfunction(latitude, longitudes, geom_fp, VOLUME_DIR, OUTPUT_DIR)

