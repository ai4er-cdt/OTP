import numpy as np
import xarray as xr
import ecco_v4_py as ecco

METERS_CUBED_TO_SVERDRUPS = 10**-6


def parse_coords(ds, coords, coordlist):
    if coords is not None:
        return coords
    else:
        for f in set(["maskW", "maskS"]).intersection(ds.reset_coords().keys()):
            coordlist.append(f)

        if "time" in ds.dims:
            coordlist.append("time")

        dsout = ds[coordlist]
        if "domain" in ds.attrs:
            dsout.attrs["domain"] = ds.attrs["domain"]
        return dsout


def _initialize_trsp_data_array(coords, lat_vals):
    """Create an xarray DataArray with time, depth, and latitude dims

    Parameters
    ----------
    coords : xarray Dataset
        contains LLC coordinates 'k' and (optionally) 'time'
    lat_vals : int or array of ints
        latitude value(s) rounded to the nearest degree
        specifying where to compute transport

    Returns
    -------
    ds_out : xarray Dataset
        Dataset with the variables
            'trsp_z'
                zero-valued DataArray with time (optional),
                depth, and latitude dimensions
            'Z'
                the original depth coordinate
    """

    lat_vals = np.array(lat_vals) if isinstance(lat_vals, list) else lat_vals
    lat_vals = np.array([lat_vals]) if np.isscalar(lat_vals) else lat_vals
    lat_vals = xr.DataArray(lat_vals, coords={"lat": lat_vals}, dims=("lat",))

    xda = xr.zeros_like(coords["k"] * lat_vals)
    xda = (
        xda if "time" not in coords.dims else xda.broadcast_like(coords["time"]).copy()
    )

    # Convert to dataset to add Z coordinate
    xds = xda.to_dataset(name="trsp_z")
    xds["Z"] = coords["Z"]
    xds = xds.set_coords("Z")

    return xds


def calc_meridional_stf(
    ds, lat_vals, doFlip=True, basin_name=None, coords=None, grid=None
):
    """Compute the meridional overturning streamfunction in Sverdrups
    at specified latitude(s)

    Parameters
    ----------
    ds : xarray DataSet
        must contain UVELMASS,VVELMASS, drF, dyG, dxG
    lat_vals : float or list
        latitude value(s) rounded to the nearest degree
        specifying where to compute overturning streamfunction
    doFlip : logical, optional
        True: integrate from "bottom" by flipping Z dimension before cumsum(),
        then multiply by -1. False: flip neither dim nor sign.
    basin_name : string, optional
        denote ocean basin over which to compute streamfunction
        If not specified, compute global quantity
        see utils.get_available_basin_names for options
    coords : xarray Dataset
        separate dataset containing the coordinate information
        YC, drF, dyG, dxG, optionally maskW, maskS
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see ecco_utils.get_llc_grid
        see also the [xgcm documentation](https://xgcm.readthedocs.io/en/latest/grid_topology.html)

    Returns
    -------
    ds_out : xarray Dataset
        with the following variables
            moc
                meridional overturning strength as maximum of streamfunction
                in depth space, with dimensions 'time' (if in dataset), and 'lat'
            psi_moc
                meridional overturning streamfunction across the section in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
            trsp_z
                freshwater transport across section at each depth level in Sv
                with dimensions 'time' (if in given dataset), 'lat', and 'k'
    """

    # get coords
    coords = parse_coords(ds, coords, ["Z", "YC", "drF", "dyG", "dxG"])

    # Compute volume transport
    trsp_x = ds["UVELMASS"] * coords["drF"] * coords["dyG"]
    trsp_y = ds["VVELMASS"] * coords["drF"] * coords["dxG"]

    # Creates an empty streamfunction
    ds_out = meridional_trsp_at_depth(
        trsp_x,
        trsp_y,
        lat_vals=lat_vals,
        coords=coords,
        basin_name=basin_name,
        grid=grid,
    )

    psi_moc = ds_out["trsp_z"].copy(deep=True)

    # Flip depth dimension, take cumulative sum, flip back
    if doFlip:
        psi_moc = psi_moc.isel(k=slice(None, None, -1))

    # Should this be done with a grid object???
    psi_moc = psi_moc.cumsum(dim="k")

    if doFlip:
        psi_moc = -1 * psi_moc.isel(k=slice(None, None, -1))

    # Add to dataset
    ds_out["psi_moc"] = psi_moc

    # Compute overturning strength
    ds_out["moc"] = ds_out["psi_moc"].max(dim="k")

    # Convert to Sverdrups
    for fld in ["trsp_z", "psi_moc", "moc"]:
        ds_out[fld] = ds_out[fld] * METERS_CUBED_TO_SVERDRUPS
        ds_out[fld].attrs["units"] = "Sv"

    # Name the fields here, after unit conversion which doesn't keep attrs
    ds_out["trsp_z"].attrs["name"] = "volumetric trsp per depth level"
    ds_out["psi_moc"].attrs["name"] = "meridional overturning streamfunction"
    ds_out["moc"].attrs["name"] = "meridional overturning strength"

    return ds_out


def meridional_trsp_at_depth(
    xfld, yfld, lat_vals, coords, basin_name=None, grid=None, less_output=True
):
    """
    Compute transport of vector quantity at each depth level
    across latitude(s) defined in lat_vals

    Parameters
    ----------
    xfld, yfld : xarray DataArray
        3D spatial (+ time, optional) field at west and south grid cell edges
    lat_vals : float or list
        latitude value(s) specifying where to compute transport
    coords : xarray Dataset
        only needs YC, and optionally maskW, maskS (defining wet points)
    basin_name : string, optional
        denote ocean basin over which to compute streamfunction
        If not specified, compute global quantity
        see get_basin.get_available_basin_names for options
    grid : xgcm Grid object, optional
        denotes LLC90 operations for xgcm, see ecco_utils.get_llc_grid
        see also the [xgcm documentation](https://xgcm.readthedocs.io/en/latest/grid_topology.html)

    Returns
    -------
    ds_out : xarray Dataset
        with the main variable
            'trsp_z'
                transport of vector quantity across denoted latitude band at
                each depth level with dimensions 'time' (if in given dataset),
                'k' (depth), and 'lat'
    """

    if grid is None:
        grid = ecco.get_llc_grid(coords)

    # Initialize empty DataArray with coordinates and dims
    ds_out = _initialize_trsp_data_array(coords, lat_vals)

    # Get basin mask
    maskW = coords["maskW"] if "maskW" in coords else xr.ones_like(xfld)
    maskS = coords["maskS"] if "maskS" in coords else xr.ones_like(yfld)

    if not isinstance(basin_name, list):
        basin_name = [basin_name]

    print(basin_name)

    if basin_name[0] is not None:
        mask_totW_list = []
        mask_totS_list = []
        for basin in basin_name:
            mask_totW_list.append(ecco.get_basin_mask(basin, maskW))
            mask_totS_list.append(ecco.get_basin_mask(basin, maskS))

    maskW = sum(mask_totW_list)
    maskS = sum(mask_totS_list)

    # These sums are the same for all lats, therefore precompute to save
    # time
    tmp_x = xfld.where(maskW)
    tmp_y = yfld.where(maskS)

    for lat in ds_out.lat.values:
        if not less_output:
            print("calculating transport for latitutde ", lat)

        # Compute mask for particular latitude band
        lat_maskW, lat_maskS = ecco.vector_calc.get_latitude_masks(
            lat, coords["YC"], grid
        )

        # Sum horizontally
        lat_trsp_x = (tmp_x * lat_maskW).sum(dim=["i_g", "j", "tile"])
        lat_trsp_y = (tmp_y * lat_maskS).sum(dim=["i", "j_g", "tile"])

        ds_out["trsp_z"].loc[{"lat": lat}] = lat_trsp_x + lat_trsp_y

    return ds_out
