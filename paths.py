import os
import config

geom_fp = os.path.join(
    config.LOCAL_DIR,
    "solodoch_data_full",
    "ECCO_L4_GEOMETRY_LLC0090GRID_V4r4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc",
)

VELOCITY_NATIVE_GRID = DATA_DIR = os.path.join(
    config.LOCAL_DIR,
    "solodoch_data_full",
    "ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4",
)

STREAMFUNCTIONS_ECCO_OUTPUT = os.path.join(config.LOCAL_DIR, "streamfunctions_ecco")
