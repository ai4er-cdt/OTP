import paths
from ecco_download import *

# shared folder for data
home = "/mnt/g/My Drive/GTC/utility_files"
shortname = "ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_MONTHLY_V4R4"
start = "1992-01-01"
end = "2017-12-31"

"""
home = paths.SOLODOCH_DATA
shortname = "ECCO_L4_BOLUS_LLC0090GRID_MONTHLY_V4R4"
start = "1992-01-01"
end = "2018-01-01"
"""


ecco_podaac_download_subset(
    ShortName=shortname,
    StartDate=start,
    EndDate=end,
    download_root_dir=f"{home}",
    force_redownload=False,
    n_workers=6,
)
