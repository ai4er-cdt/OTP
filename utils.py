import sys

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
    just_bounds : boolean
        should we just return the min/max longitude?

    Returns
    -------
    lon : np.array
        list of relevant longitudes for the latitude and basin OR longitude bounds
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
        relevant_lons = np.array([relevant_lons[0], relevant_lons[-1]])

    return relevant_lons

def get_basin_solodoch(basin, geometry_filepath):
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

    # Atalntic Ocean: all of the default ECCO Atlantic basin between 55N and 34S, minus the Mediterranean
    minmax_lat = xr.ones_like(xds_geom.YC).where((xds_geom.YC >= min_lat), 0).where((max_lat >= xds_geom.YC), 0).compute()
    med_mask = ecco.get_basin_mask(basin_name = 'med', mask = xr.ones_like(xds_geom.YC))
    full_atl_mask = np.logical_and(minmax_lat, np.logical_not(med_mask)).astype(int)

    atlantic = ecco.get_basin_mask(basin_name = 'atlExt', mask = full_atl_mask)

    if basin == 'atlantic':
        return atlantic

    # Indo-Pacific Ocean: everything between 55N and 34S that isn't in the Atlantic or Southern Oceans 
    indo_pacific = np.logical_and(np.logical_and(np.logical_and(all_oceans, minmax_lat), np.logical_not(southern)), np.logical_not(atlantic)).astype(int)
    
    if basin == 'indo-pacific':
        return indo_pacific

def get_available_basin_names_solodoch():
    avail_basins = ['atlantic', 'indo-pacific', 'southern']

    return avail_basins

def resample_to_1deg(mask, geometry_filepath, new_grid_delta_lat = 1, new_grid_delta_lon = 1, new_grid_min_lat = -90, new_grid_max_lat = 90, new_grid_min_lon = -180, new_grid_max_lon = 180):
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

if __name__ == '__main__':
    geom_fp = 'ECCO_L4_GEOMETRY_LLC0090GRID_V4R4/GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc'

    print(get_longitudes_at_latitude(20, 'indo-pacific', geom_fp))