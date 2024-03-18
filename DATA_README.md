# Authors
- [Emilio Luz-Ricca](mailto:el590@cam.ac.uk)
- [Sharan Maiya](mailto:sm2783@cam.ac.uk)
- [Aline Van Driessche](mailto:av656@cam.ac.uk)
- [Nina Baranduin](mailto:ngb34@cam.ac.uk)
- [Thomas Cowperthwaite](mailto:tc656@cam.ac.uk)

# Overview

This repository contains all [NASA ECCO](https://ecco-group.org/) data used for the AI4ER Guided Team Challenge oceans project. Data was downloaded from [PO.DAAC](https://podaac.jpl.nasa.gov/) using the entries in the table below. Download and preprocessing scripts can be found in the [corresponding GitHub repository](https://github.com/ai4er-cdt/OTP/tree/main).

|       Data Type       |               PO.DAAC Entry              |                 DOI                 |
|:---------------------:|:----------------------------------------:|:-----------------------------------:|
|          SSH          |      ECCO_L4_SSH_05DEG_MONTHLY_V4R4      | https://doi.org/10.5067/ECG5M-SSH44 |
|          SSS          | ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4 | https://doi.org/10.5067/ECG5M-OTS44 |
|          SST          | ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4 | https://doi.org/10.5067/ECG5M-OTS44 |
|          ZWS          |     ECCO_L4_STRESS_05DEG_MONTHLY_V4R4    | https://doi.org/10.5067/ECG5M-STR44 |
|          OBP          |      ECCO_L4_OBP_05DEG_MONTHLY_V4R4      | https://doi.org/10.5067/ECG5M-OBP44 |
| Monthly-Mean Velocity |   ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4   | https://doi.org/10.5067/ECG5M-OVE44 |
|     Bolus Velocity    |     ECCO_L4_BOLUS_05DEG_MONTHLY_V4R4     | https://doi.org/10.5067/ECG5M-BOL44 |
|  Model Grid Geometry  |     ECCO_L4_GEOMETRY_LLC0090GRID_V4R4    | https://doi.org/10.5067/ECL5A-GRD44 |

Use and redistribution of this data is in line with [NASA EarthData's Data and Information Policy](https://www.earthdata.nasa.gov/engage/open-data-services-and-software/data-and-information-policy) and [NASA PO.DAAC's Data Use and Citation Policy](https://podaac.jpl.nasa.gov/CitingPODAAC).

# Data Structure

## Latitudes of Interest

We primarily focus on abyssal circulation at four latitudes: 26N, 30S, 55S, and 60S. Data is monthly and derived from the 0.5 degree latitude-longitude interpolated product on PO.DAAC. When working in Python: we recommend using `xarray` for `nc` files and `pickle`/`numpy` for `pickle` files.

There is one directory for each latitude. All of these will contain the following:
- `[LATITUDE].nc`: Satellite-observable variables for the latitude of interest, as well for the latitude directly above and below (plus/minus 0.5 degrees).
   - These include ample description embedded directly into the `xarray` object.
- `[LATITUDE]_moc_depth.pickle` and `[LATITUDE]_moc_density.pickle`: The depth- and density-space MOC, taken as the maximum of the depth- and density-space streamfunctions, respectively.
   - This will read into Python as a `numpy` array, with shape `[# TIME STEPS]`.
- `[LATITUDE]_sf_depth.pickle` and `[LATITUDE]_sf_density.pickle`: The full overturning streamfunction in depth- and density-space, i.e., no maximum has been taken over depth/density yet.
   - This will read into Python as a `numpy` array, with shape `[# VERTICAL LEVELS x # TIME STEPS]`.
- `[LATITUDE]_density_range.pickle`: The density range for the density-space streamfunction. This is different for each latitude.
   - This will read into Python as a `numpy` array, with shape `[# VERTICAL LEVELS]`.

30S includes three extra files for the depth- and density-space streamfunction for only the Atlantic Ocean (`30S_atlantic_sf_depth.pickle` and `30S_atlantic_sf_density.pickle`, respectively) and the density range for the Atlantic Ocean streamfunction (`30S_atlantic_density_range.pickle`).

At the top-level directory, `depth_range.pickle` includes the actual depths for the depth-space streamfunction, which is constant across latitudes. This is taken directly from ECCO's vertical grid geometry profile. This will read into Python as a `numpy` array, with shape `[# VERTICAL LEVELS]`.

All streamfunction and MOC strength measurements have units of Sverdrup, i.e., 10^6 m^3s^-1. Satellite-observable variables each have different units--see data variable descriptions embedded directly into the `xarray` object for more details.

## Full Southern Ocean

We also experiment with machine learning models trained on the entire Southern Ocean, i.e., all latitudes below 30S. The two directories are `inputs` and `moc`:
- `inputs`: Contains `southern_ocean.nc`, which holds all satellite-observable variables for all latitudes in the Southern Ocean, as well as `southern_ocean_floor.nc`, which holds the bottom depth for the entire Southern Ocean.
- `moc`: Contains a single file for each latitude's MOC strength through time. Filenames are in the format `[LATITUDE]_moc.pickle`.
   - These will read into Python as `numpy` arrays, with shape `[# TIME STEPS]`.
