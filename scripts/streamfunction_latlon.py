from typing import List
import numpy as np
import xarray as xr
from geopy.distance import distance
from scipy.integrate import simpson
from gsw.conversions import CT_from_pt
from gsw.density import sigma2
from scipy.interpolate import interp1d
from tqdm import tqdm

def psi_depth(d: float|np.ndarray, v: np.ndarray, Z: np.ndarray, d_lon: np.ndarray) -> np.ndarray:
    n_lons, n_lats, n_depths, n_times = v.shape
    if type(d) == np.ndarray: 
        assert len(d) == n_lats
        ixs = [list(Z).index(x) for x in d]
    else:
        ixs = [list(Z).index(d)]*n_lats
    out = np.zeros((n_lats, n_times))
    for lat in range(n_lats):
        ix = ixs[lat]
        for t in range(n_times):
            if ix == 0: 
                out[lat, t] = 0
                continue
            # f: (n_lons, n_depths)
            f = v[:, lat, :, t]
            # inner: (n_depths,)
            inner = np.array([simpson(y=f[:, depth], x=d_lon[lat, :]) for depth in range(n_depths)])
            outer = simpson(y=inner[:ix], x=Z[:ix])
            out[lat, t] = outer
    out = -out / 1e6
    return out

def psi_density(d: float|np.ndarray, 
                v: np.ndarray, 
                Z: np.ndarray, 
                d_lon: np.ndarray,
                density: np.ndarray) -> np.ndarray:
    n_lons, n_lats, n_depths, n_times = v.shape
    if type(d) == np.ndarray: 
        assert len(d) == n_lats
    else:
        d = [d]*n_lats
    # calculate isopycnal depths (storing indices as the actual depths don't really matter)
    isopycnals = np.zeros((n_lats, n_lons, n_times))
    for lat in range(n_lats):
        for lon in range(n_lons):
            for t in range(n_times):
                column = density[lon, lat, :, t]
                # find the index of the first depth over which we cross this density
                ix = np.argmax(column >= d[lat])
                isopycnals[lat, lon, t] = ix

    out = np.empty((n_lats, n_times))
    for lat in range(n_lats):
        for t in range(n_times):
            # f: (n_lons, n_depths)
            f = v[:, lat, :, t]
            # integrate over depth first (because it varies by longitude)
            inner = []
            for l in range(n_lons):
                ix = int(isopycnals[lat, l, t])
                if ix == 0: 
                    inner.append(0)
                else:
                    inner.append(simpson(y=f[l, :ix], x=Z[:ix]))
            # inner now has shape (n_lons,)
            outer = simpson(y=inner, x=d_lon[lat, :])
            out[lat, t] = outer
    out = -out / 1e6
    return out

def get_moc_strength(section: str,
                     data_path: str="/mnt/g/My Drive/GTC/ecco_data_full",
                     use_density: bool=False,
                     density_precision: int=10,
                     use_bolus: bool=False,
                     interpolate: bool=False,
                     interp_precision: int=10) -> np.ndarray:
    
    sections = ["26N", "30S", "55S", "60S", "southern_ocean"]
    # this governs the original order of coordinate axes
    coordinates = ["longitude", "latitude", "Z", "time"]
    if section not in sections: raise Exception(f"argument 'section' must be in: {sections}")

    print("reading velocities...")
    # get monthly mean velocity
    vm = xr.open_mfdataset(f"{data_path}/{section}/ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4/*nc",
                            coords="minimal",
                            data_vars="minimal",
                            parallel=True, compat="override")
    vm = vm[["NVEL"]].transpose(*coordinates)
    vm = vm.rename({"NVEL": "vm"})

    data = vm
    if use_bolus:
        print("adding bolus velocities...")
        # get bolus velocity
        ve = xr.open_mfdataset(f"{data_path}/{section}/ECCO_L4_BOLUS_05DEG_MONTHLY_V4R4/*nc",
                                coords="minimal",
                                data_vars="minimal",
                                parallel=True, compat="override")
        ve = ve[["NVELSTAR"]].transpose(*coordinates)
        ve = ve.rename({"NVELSTAR": "ve"})
        data = xr.merge([vm, ve], join="inner")
        ve = data["ve"].to_numpy(); ve = np.nan_to_num(ve)
        ve = np.flip(ve, axis=2)

    vm = data["vm"].to_numpy(); vm = np.nan_to_num(vm)
    vm = np.flip(vm, axis=2)
    v = vm + ve if use_bolus else vm
    lats = data["latitude"].to_numpy()
    lons = data["longitude"].to_numpy()
    # flip depth axis because we integrate from the bottom up
    Z = np.flip(data["Z"].to_numpy())
    # get distance between each lat/lon in meters
    grid = [[(lat, lon) for lon in lons] for lat in lats]
    d_lon = np.array([[distance(lat[0], p).meters for p in lat] for lat in grid])
    # geopy returns the shortest distance - we need to account for this
    for lat in range(d_lon.shape[0]):
        is_monotonic = (np.diff(d_lon[lat]) > 0).all()
        if not is_monotonic:
            halfway = max(d_lon[lat])
            half_ix = np.argmax(d_lon[lat] == halfway)
            delta = halfway - d_lon[lat, half_ix+1:]
            d_lon[lat, half_ix+1:] =  halfway + delta

    if use_density:
        print("using density, so reading potential temperature and absolute salinity...")
        # open potential temperature and absolute salinity - needed to calculate sigma_2 -> isopycnals
        data = xr.open_mfdataset(f"{data_path}/{section}/ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4/*nc",
                                coords="minimal",
                                data_vars="minimal",
                                parallel=True, compat="override")
        data = data[["SALT", "THETA"]]
        print("converting potential temperature to conservative temperature...")
        ct = CT_from_pt(data["THETA"], data["SALT"])
        print("calculating potential density at 2000 decibars (sigma_2)")
        density = sigma2(data["SALT"], ct)
        # convert values to float16 to reduce size
        # indices don't support float16 (why tho), so we'll settle for float32
        for c in ["Z", "latitude", "longitude"]:
            density.coords[c] = density.coords[c].astype("float32")
        density = density.astype("float16")
        # flip along the depth axis
        density = density.isel(Z=slice(None, None, -1))
        # re-order coordinates just in case they have been shuffled around
        density = density.transpose(*coordinates)
        density = density.to_numpy()

        density_range = density.flatten()
        density_range = density_range[~np.isnan(density_range)]
        density_range = np.linspace(min(density_range), max(density_range), density_precision)

    Z_interpolated = Z
    if interpolate:
        # interpolate depth onto a regular grid
        Z_interpolated = np.linspace(Z[0], Z[-1], num=interp_precision)
        
        # efficient interpolating by unfolding
        if use_density:
            print(f"interpolating density (over {interp_precision} depths)...")
            temp = density.reshape(-1, len(Z))
            temp_interpolated = np.empty((temp.shape[0], interp_precision))
            for i, profile in enumerate(temp):
                f = interp1d(Z, profile, bounds_error=False, fill_value="extrapolate")
                temp_interpolated[i] = f(Z_interpolated)
            density = temp_interpolated.reshape(len(lons), len(lats), interp_precision, density.shape[-1])

        print(f"interpolating velocity (over {interp_precision} depths)...")
        temp = v.reshape(-1, len(Z))
        temp_interpolated = np.empty((temp.shape[0], interp_precision))
        for i, profile in enumerate(temp):
            f = interp1d(Z, profile, bounds_error=False, fill_value="extrapolate")
            temp_interpolated[i] = f(Z_interpolated)
        v = temp_interpolated.reshape(len(lons), len(lats), interp_precision, v.shape[-1])

    psi = psi_density if use_density else psi_depth
    args = [v, Z_interpolated, d_lon]
    if use_density: args += [density]
    psi_domain = density_range if use_density else Z_interpolated

    # calculate moc strength in depth space
    streamfunction = [psi(d, *args) for d in tqdm(psi_domain)]
    streamfunction = np.array(streamfunction)
    # find the depth/density with largest absolute time-averaged streamfunction
    d_0 = Z_interpolated[np.argmax(abs(np.mean(streamfunction, axis=-1)), axis=0)]
    # calculate moc strength through this value
    moc_strength = psi(d_0, *args)
    # multiply by the sign
    sign = (np.mean(moc_strength, axis=-1) > 0) * 2. - 1
    moc_strength *= sign[:, np.newaxis]
    return moc_strength