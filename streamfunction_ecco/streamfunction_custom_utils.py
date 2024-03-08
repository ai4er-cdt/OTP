import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def get_gridllc0090_mask(ds, target_latitude, longitudes):
    FLIPPED_TILES = [7, 8, 9, 10, 11, 12, 13]

    # Split up the separate sections for the longitude list
    longitude_sections = np.split(longitudes, np.where(np.diff(longitudes) > 1)[0] + 1)

    # Create an empty mask
    mask = xr.DataArray(data=0, dims=ds["YC"].dims, coords=ds["YC"].coords)

    # Iterate over each "tile" of the grid
    for tile_num in ds["tile"].values:

        if tile_num + 1 in FLIPPED_TILES:
            latitude_dim = "j"
        else:
            latitude_dim = "i"

        tile_lats = ds["YC"].sel(tile=tile_num)
        tile_lons = ds["XC"].sel(tile=tile_num)

        tile_mask = abs(tile_lats - target_latitude)
        zeros = xr.zeros_like(tile_mask)

        first_col = tile_lats.isel(**{latitude_dim: 0})
        lowest_lat = first_col.values.max()
        highest_lat = first_col.values.min()

        if (
            max(lowest_lat, highest_lat) > target_latitude
            and min(lowest_lat, highest_lat) < target_latitude
        ):

            row_to_mask = int(tile_mask.isel(**{latitude_dim: 0}).argmin().values)

            for longitude in longitude_sections:
                if tile_num + 1 in FLIPPED_TILES:
                    column_mask = (tile_lons[:, row_to_mask] >= longitude[0]) & (
                        tile_lons[:, row_to_mask] <= longitude[-1]
                    )
                    zeros.loc[column_mask, row_to_mask] = 1
                else:
                    column_mask = (tile_lons[row_to_mask, :] >= longitude[0]) & (
                        tile_lons[row_to_mask, :] <= longitude[-1]
                    )
                    zeros.loc[row_to_mask, column_mask] = 1

        # Update the mask DataArray for the current face
        mask.loc[dict(tile=tile_num)] = zeros.astype(int)
    return mask


def plot_2D_streamfunction(stf_ds, moc_param="psi_moc", title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(stf_ds["time"], stf_ds[moc_param])
    plt.xlabel("Time")
    plt.ylabel("PSI in layer with maximal density level")
    if title is None:
        title = "PSI Streamfunction"
    plt.title(title)
    plt.grid(True)
    plt.show()
