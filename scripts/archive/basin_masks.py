import sys
import math

import xarray as xr
import numpy as np

sys.path.append("ECCOv4-py")
import ecco_v4_py as ecco

def get_longitudes_at_latitude(lat, basin, geometry_filepath, just_bounds = False):

    """
    Return the relevant longitudes for a given latitude and basin.

    Parameters
    ----------
    lat : float 
        latitude of interest
    basin : string
        basin name, corresponding to the available basins in the get_available_basin_names_solodoch()
        function below
    geometry_filepath : string
        local filepath to the ECCO geometry .nc file
    just_bounds : boolean
        should we just return bounds for the longitudes, e.g., the ranges?

    Returns
    -------
    lon : np.array
        list of relevant longitudes for the latitude and basin or list of longitude bound tuples
    """

    # Extract the mask for the input basin
    basin_mask = get_basin_solodoch(basin, geometry_filepath)
    
    # Resample to normal geo coordinate system at 1 degree resolution
    new_grid_lon_centers, new_grid_lat_centers, mask_nearest_1deg = resample_to_1deg(basin_mask, geometry_filepath)

    # Match query latitude with the closest available latitude based on grid center latitudes
    closest_index = np.argmin(np.abs(new_grid_lat_centers[ : , 0] - lat))
    new_lat = new_grid_lat_centers[closest_index]

    # Extract the relevant longitudes for the input latitude
    mask_at_lat = mask_nearest_1deg[np.argwhere((new_grid_lat_centers == new_lat))[0][0]]
    relevant_lons = new_grid_lon_centers[new_grid_lat_centers == new_lat][mask_at_lat.astype(bool)]

    if just_bounds:
        relevant_lons = extract_bounds(relevant_lons)

    return relevant_lons

def get_basin_solodoch(basin, geometry_filepath):

    """
    Helper function to extract the basin masks as defined in Solodoch et al. (2023).
    """

    assert basin in get_available_basin_names_solodoch(), f'{basin} not a valid basin, must be in {get_available_basin_names_solodoch()}'

    xds_geom = xr.open_dataset(geometry_filepath)

    max_lat = 55
    min_lat = -34

    all_oceans = ecco.get_basin_mask(ecco.get_available_basin_names(), mask = xr.ones_like(xds_geom.YC))

    # Southern Ocean: everything below 34S
    below_min_lat = xr.ones_like(xds_geom.YC).where(min_lat >= xds_geom.YC, 0).compute()
    southern = np.logical_and(all_oceans, below_min_lat).astype(int)

    if basin == 'southern':
        return southern

    # Atlantic Ocean: all of the default ECCO Atlantic basin between 55N and 34S, minus the Mediterranean
    minmax_lat = xr.ones_like(xds_geom.YC).where((xds_geom.YC >= min_lat), 0).where((max_lat >= xds_geom.YC), 0).compute()
    med_mask = ecco.get_basin_mask(basin_name = 'med', mask = xr.ones_like(xds_geom.YC))
    full_atl_mask = np.logical_and(minmax_lat, np.logical_not(med_mask)).astype(int)

    atlantic = ecco.get_basin_mask(basin_name = 'atlExt', mask = full_atl_mask)

    if basin == 'atlantic':
        return atlantic

    # Indo-Pacific Ocean: everything between 55N and 34S that isn't in the Atlantic or Southern Oceans 
    indo_pacific = np.logical_and(np.logical_and(np.logical_and(np.logical_and(all_oceans, minmax_lat), np.logical_not(southern)), np.logical_not(atlantic)), np.logical_not(med_mask)).astype(int)
    
    if basin == 'indo-pacific':
        return indo_pacific

def get_available_basin_names_solodoch():

    """
    Helper function to check the available basin names.
    """

    avail_basins = ['atlantic', 'indo-pacific', 'southern']

    return avail_basins

def resample_to_1deg(mask, geometry_filepath, new_grid_delta_lat = 1, new_grid_delta_lon = 1, new_grid_min_lat = -90, new_grid_max_lat = 90, new_grid_min_lon = -180, new_grid_max_lon = 180):
    
    """
    Helper function to resample from the native ECCO grid to 1 degree latitude-longitude grid.
    """
    
    xds_geom = xr.open_dataset(geometry_filepath)
    
    new_grid_lon_centers, new_grid_lat_centers, _, _, mask_nearest_1deg = ecco.resample_to_latlon(xds_geom.XC, 
                                                                                                  xds_geom.YC, 
                                                                                                  mask,
                                                                                                  new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,
                                                                                                  new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,
                                                                                                  fill_value = None, 
                                                                                                  mapping_method = 'nearest_neighbor',
                                                                                                  radius_of_influence = 120000)
    
    return new_grid_lon_centers, new_grid_lat_centers, mask_nearest_1deg

def extract_bounds(lon_list):

    """
    Helper function to find change points in a longitude list, i.e., where there is a change of more than one degree.
    """

    bounds = []

    curr_lon = lon_list[0]

    for i in range(len(lon_list)):
        if i == len(lon_list) - 1:
            bounds.append((math.floor(curr_lon), math.ceil(lon_list[i])))
        else:
            diff = abs(lon_list[i] - lon_list[i + 1])

            if diff > 1:
                bounds.append((math.floor(curr_lon), math.ceil(lon_list[i])))
                curr_lon = lon_list[i + 1]

    return bounds

def get_lats_of_interest_solodoch(basin):

    """
    Helper function to get the latitudes of interest for reproducing Solodoch et al. (2023).
    """

    if basin == 'atlantic':
        return [26.5]
    elif basin == 'southern':
        return [-55, -60]
    elif basin == 'indo-pacific':
        return [-30]

if __name__ == '__main__':
    geom_fp = 'ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc'

    basin = 'indo-pacific'
    lats_of_interest = get_lats_of_interest_solodoch(basin)

    for lat in lats_of_interest:
        lon_list = get_longitudes_at_latitude(lat, basin, geom_fp, just_bounds = True)
        print(lon_list)