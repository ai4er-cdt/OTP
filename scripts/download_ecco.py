'''
Dataset names (SSH, SST, SSS, OBP, ZWS):
- ECCO_L4_SSH_05DEG_MONTHLY_V4R4
- ECCO_L4_STRESS_05DEG_MONTHLY_V4R4
- ECCO_L4_OBP_05DEG_MONTHLY_V4R4
- ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4

Latitudes:
- 60S
- 55S
- 30S
- 26N

(For streamfunction calculations we additionally need the following variables:
- UVELMASS
- VVELMASS

from the dataset:
- ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4

For downloading of these data, see the script download_flux.py)
'''

from ecco_download import *

# shared folder for data
home = "/mnt/g/My Drive/GTC/_solodoch_data_full"

# name of ecco datasets which contain input variables
datasets = ["ECCO_L4_SSH_05DEG_MONTHLY_V4R4", "ECCO_L4_STRESS_05DEG_MONTHLY_V4R4", 
            "ECCO_L4_OBP_05DEG_MONTHLY_V4R4", "ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4"]

start = "1992-01-01"; end = "2017-12-31"

lats = [60, 70, 120, 232] 
lats_original = ["60S", "55S", "30s", "26N"]

for lat, name in zip(lats, lats_original):
    for shortname in datasets:
        ecco_podaac_download_subset(ShortName=shortname,
                                    StartDate=start, EndDate=end,
                                    download_root_dir=f"{home}/{name}", force_redownload=False,
                                    latitude_isel=[lat-1, lat+2, 1], longitude_isel=[0, 720, 1],
                                    n_workers=6)