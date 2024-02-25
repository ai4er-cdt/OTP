from typing import Optional
import numpy as np
import xarray as xr
from gsw.conversions import CT_from_pt
from gsw.density import sigma2
from geopy.distance import distance
from scipy.integrate import simpson
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle


def lat_depth(v: np.ndarray, 
              x_Z: np.ndarray, 
              x_lon: np.ndarray, 
              d: np.ndarray, 
              lat: int) -> np.ndarray:
    _, n_lons, _, n_times = v.shape
    ix = list(x_Z).index(d[lat])+1
    moc = np.empty(n_times)
    for t in range(n_times):
        # inner integral over depth
        inner = simpson(y=np.nan_to_num(v[lat, :, :ix, t]), x=x_Z[:ix], axis=1)
        # outer integral over longitude
        moc[t] = -simpson(y=inner, x=x_lon[lat, :]) / 1e6
    return moc

def lat_density(v: np.ndarray, 
                x_Z: np.ndarray, 
                x_lon: np.ndarray, 
                lighter: np.ndarray,
                empty: np.ndarray,
                isopycnals: np.ndarray,
                lat: int) -> np.ndarray:
    _, n_lons, _, n_times = v.shape
    moc = np.empty(n_times)

    for t in range(n_times):
        for l in range(n_lons):
            if not (lighter[lat, l, t] or empty[lat, l, t]):
                ix = isopycnals[lat, l, t]
            else:
                ix = 0
            v[lat, l, ix:, t] = 0.

        # inner integral over depth
        inner = simpson(y=v[lat, :, :, t], x=x_Z, axis=1)
        # outer integral over longitude
        moc[t] = -simpson(y=inner, x=x_lon[lat, :]) / 1e6
    return moc

def psi(d: float|np.ndarray, 
        v: np.ndarray, 
        x_lon: np.ndarray, 
        x_Z: np.ndarray,
        s2: Optional[np.ndarray],
        use_density: bool=False) -> np.ndarray:
    n_lats, _, n_depths, _ = v.shape
    d = np.array([d]*n_lats) if np.isscalar(d) else np.asarray(d)
    assert len(d) == n_lats
    # find water columns lighter than d
    lighter = np.less_equal(s2[:, :, 0, :], d[:, np.newaxis, np.newaxis])
    # find missing water columns
    empty = np.isnan(s2).all(axis=2)
    # find isopycnals (defaults to 0)
    isopycnals = np.argmax(np.less_equal(s2, d[:, np.newaxis, np.newaxis, np.newaxis]), axis=2)
    # any defaults (0 values) represent entire water columns which are heavier than d
    isopycnals[isopycnals == 0] = n_depths
    if use_density:
        f = lat_density
        args = [v, x_Z, x_lon, lighter, empty, isopycnals]
    else:
        f = lat_depth
        args = [v, x_Z, x_lon, d]

    # parallelise over latitudes
    # NOTE: in my testing, I found that parallelising over latitudes was more efficient than any other dimension
    with Pool(cpu_count()) as pool:
            results = pool.starmap(f, [(*args, lat) for lat in range(n_lats)]) 
    return np.array(results)

def calculate_moc(section: str,
                  use_bolus: bool=True,
                  use_density: bool=True,
                  density_precision: int=100,
                  data_path: str="/mnt/g/My Drive/GTC/ecco_data",
                  plot_path: str="/mnt/g/My Drive/GTC/EDA/moc/latlon",
                  display_plot: bool=False) -> np.ndarray:
    
    sections = ["26N", "30S", "55S", "60S", "southern_ocean"]
    coordinates = ["latitude", "longitude", "Z", "time"]
    assert section in sections
    print("fetching monthly mean velocities")
    vm = xr.open_mfdataset(f"{data_path}_full/{section}/ECCO_L4_OCEAN_VEL_05DEG_MONTHLY_V4R4/*.nc",
                        coords="minimal",
                        data_vars="minimal",
                        parallel=True, compat="override")
    vm = vm[["NVEL"]].transpose(*coordinates).isel(Z=slice(None, None, -1)).fillna(0.)
    vm = vm.rename({"NVEL": "vm"})
    if use_bolus:
        print("fetching bolus velocities")
        ve = xr.open_mfdataset(f"{data_path}_full/{section}/ECCO_L4_BOLUS_05DEG_MONTHLY_V4R4/*.nc",
                            coords="minimal",
                            data_vars="minimal",
                            parallel=True, compat="override")
        ve = ve[["NVELSTAR"]].transpose(*coordinates).isel(Z=slice(None, None, -1)).fillna(0.)
        ve = ve.rename({"NVELSTAR": "ve"})
    if use_density:
        print("using density: fetching temperature and salinity for calculation")
        density = xr.open_mfdataset(f"{data_path}_full/{section}/ECCO_L4_TEMP_SALINITY_05DEG_MONTHLY_V4R4/*.nc",
                                    coords="minimal",
                                    data_vars="minimal",
                                    parallel=True, compat="override")
        density = density[["THETA", "SALT"]].transpose(*coordinates).isel(Z=slice(None, None, -1))
        ct = CT_from_pt(SA=density["SALT"], pt=density["THETA"])
        s2 = sigma2(SA=density["SALT"], CT=ct).to_dataset()
        s2 = s2.rename({list(s2.data_vars)[0]: "sigma_2"})
        
    v = vm["vm"] + ve["ve"] if use_bolus else vm["vm"]
    time = vm["time"].to_numpy()
    grid = np.array([[(lat, lon) for lon in v["longitude"].to_numpy()] for lat in v["latitude"].to_numpy()])
    # get x-coordinates of longitude measurements
    x_lon = np.array([[0.]+[distance(latitude[i+1], latitude[i]).meters
                            for i in range(grid.shape[1]-1)] 
                            for latitude in grid])
    # rounding to cm
    x_lon = np.round(np.cumsum(x_lon, -1), 2)
    # get x-coordinates of depth measurements
    x_Z = v["Z"].to_numpy()
    # unsure if -ve is a problem but getting rid of them just in case
    x_Z += max(abs(x_Z))
    print("loading data into memory")
    v_np = v.to_numpy()
    if use_density:
        s2_np = s2["sigma_2"].to_numpy()
        temp = s2_np.flatten()
        temp = temp[~np.isnan(temp)]
        sf_range = np.linspace(min(temp), max(temp), density_precision)
    else:
        s2_np = None
        sf_range = x_Z

    args = [v_np, x_lon, x_Z, s2_np, use_density]
    # calculate the streamfunction at all possible densities/depths
    if use_density: print("calculating streamfunction for all densities")
    else: print("calculating streamfunction for all depths")
    streamfunction = np.array([psi(d, *args) for d in tqdm(sf_range)])
    # find the density/depth with the largest absolute time-averaged streamfunction
    d_0 = sf_range[np.argmax(abs(np.mean(streamfunction, axis=-1)), axis=0)]
    # calculate moc strength at d_0
    print("calculating moc strength")
    moc = psi(d_0, *args)

    outfile = f"/mnt/g/My Drive/GTC/ecco_data_minimal/{section}_moc"
    outfile = f"{outfile}_density.pickle" if use_density else f"{outfile}_depth.pickle"
    print("done!")
    print(f"saving moc to {outfile}")
    outfile = open(outfile, "wb")
    pickle.dump(moc, outfile); outfile.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_title = f"{section}: MOC Strength"
    plot_title = f"{plot_title} (density-space)" if use_density else f"{plot_title} (depth-space)"
    ax.set_title(plot_title)
    ax.set_xlabel("Year"); ax.set_ylabel("[Sv]")
    ax.plot(time, moc[1, :], color="red", linestyle="-", linewidth="1")
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    figtitle = f"{plot_path}/{section}"
    figtitle = f"{figtitle}_density.png" if use_density else f"{figtitle}_depth.png"
    print(f"saving plot to {figtitle}")
    plt.savefig(figtitle, dpi=400)
    if display_plot: plt.show()
    else: plt.close()
    return moc
