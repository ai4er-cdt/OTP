import paths
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ECCO_GRID = xr.open_dataset(paths.geom_fp)


def find_depths_of_interest(PSI, depth_grid=ECCO_GRID["Z"]):
    # Limit the range in which the northwards flow can occur
    relevant_PSI = PSI.isel(k=slice(0, int(PSI.sizes["k"] / 2))).fillna(0)
    max_indices = relevant_PSI["psi_moc"].argmax(dim="k")
    depth_max_values = depth_grid.isel(k=max_indices)
    PSI["idx_depth_max"] = max_indices
    PSI["depth_max"] = depth_max_values

    # Needed to avoid using minimum values above the depth for the maximal value
    relevant_PSI = PSI.isel(k=slice(max_indices.max().item(), PSI.sizes["k"])).fillna(0)
    min_indices = max_indices + relevant_PSI["psi_moc"].argmin(dim="k")

    # Mask the values where the moc strenght is not below 0
    moc_values = PSI["psi_moc"].isel(k=min_indices)
    min_masked_indices = xr.where(moc_values < 0, min_indices, np.nan)
    depth_min_values = xr.where(moc_values < 0, depth_grid.isel(k=min_indices), np.nan)
    PSI["idx_depth_min"] = min_masked_indices
    PSI["depth_min"] = depth_min_values

    print("len", len(max_indices))

    return _get_turning_point(PSI, depth_grid, min_masked_indices, max_indices)


def _get_turning_point(PSI, depth_grid, min_indices, max_indices):
    zero_indices = []

    # Iterate over each timestep
    for time in range(len(PSI["time"].values)):
        if not np.isnan(min_indices[time].item()):

            min_idx = int(min_indices[time].item())
            max_idx = int(max_indices[time].item())

            psi_moc_slice = PSI["psi_moc"].isel(
                {
                    "time": time,
                    "k": slice(min(min_idx, max_idx), max(min_idx, max_idx)),
                }
            )

            signs = np.sign(psi_moc_slice.squeeze())
            signchange = ((np.roll(signs, 1) - signs) != 0).astype(int)
            cross_indices = np.where(signchange != 0)

            if len(cross_indices[0]) == 0:
                cross_idx = max(min_idx, max_idx)
            else:
                cross_idx = np.where(np.diff(signs))[0][0] + min(min_idx, max_idx)

            zero_indices.append(cross_idx)

        else:
            zero_indices.append(np.nan)

    zero_indices = xr.DataArray(zero_indices, dims="time", coords={"time": PSI["time"]})
    depth_zero_values = xr.where(
        zero_indices.notnull(),
        depth_grid.isel(k=zero_indices.fillna(0).astype(int)),
        np.nan,
    )
    PSI["idx_depth_zero"] = zero_indices
    PSI["depth_zero"] = depth_zero_values
    return PSI


def plot_3_cells(psi_data, latitude, str=""):
    plt.figure(figsize=(10, 4))

    plt.plot(
        psi_data["time"],
        psi_data[f"{str}depth_max"],
        color="blue",
        label="Max northwards flow",
    )
    plt.plot(
        psi_data["time"],
        psi_data[f"{str}depth_zero"],
        color="green",
        label="Turning point (zero net flow)",
    )
    plt.plot(
        psi_data["time"],
        psi_data[f"{str}depth_min"],
        color="blue",
        linestyle="dashed",
        label="Max southwards flow",
    )

    plt.title(f"Analysis streamfunction {latitude}")
    plt.xlabel("time")
    plt.ylabel("Depth below sea level (m)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def plot_distances(psi_data, latitude):
    plt.figure(figsize=(10, 4))

    plt.plot(
        psi_data["time"],
        psi_data["dist_upper_cell"],
        color="blue",
        label="Width upper cell",
    )
    plt.plot(
        psi_data["time"],
        psi_data["dist_lower_cell"],
        color="green",
        label="Width lower cell",
    )

    plt.title(f"Analysis streamfunction {latitude}")
    plt.xlabel("time")
    plt.ylabel("Depth below sea level (m)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
