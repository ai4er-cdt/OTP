import os
import config

LOCAL_DIR = config.LOCAL_DIR

SOLODOCH_DATA = os.path.join(LOCAL_DIR, "ecco_data_full")

geom_fp = os.path.join(
    SOLODOCH_DATA,
    "ECCO_L4_GEOMETRY_LLC0090GRID_V4r4",
    "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc",
)

VELOCITY_NATIVE_GRID = os.path.join(
    SOLODOCH_DATA, "ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4"
)
BOLUS_NATIVE_GRID = os.path.join(
    SOLODOCH_DATA, "ECCO_L4_BOLUS_LLC0090GRID_MONTHLY_V4R4"
)

STREAMFUNCTIONS_ECCO_OUTPUT = os.path.join(LOCAL_DIR, "[OLD] streamfunctions_ecco")

RAPID_ARRAY = os.path.join(LOCAL_DIR, "rapid_26N")

MODELS_OUTPUT = os.path.join(LOCAL_DIR, "ecco_models")
