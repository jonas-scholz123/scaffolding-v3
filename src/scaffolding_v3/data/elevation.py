import numpy as np
import scipy.ndimage
import xarray as xr

from scaffolding_v3.config import Paths


def load_elevation_data(
    paths: Paths, ppu: int, include_tpi: bool = False
) -> xr.Dataset:
    elevation = xr.load_dataset(paths.elevation)

    coarsen = {
        "lat": _coarsen(elevation["lat"], ppu),
        "lon": _coarsen(elevation["lon"], ppu),
    }
    elevation = elevation.coarsen(coarsen, boundary="trim").mean()  # type: ignore

    if include_tpi:
        elevation = _add_tpi(elevation)

    return elevation


def _add_tpi(aux: xr.Dataset) -> xr.Dataset:
    """
    Add topographic position index to elevation dataset.
    """
    # Resolutions in coordinate values along the spatial row and column dimensions
    #   Here we assume the elevation is on a regular grid, so the first difference
    #   is equal to all others.
    coord_names = list(aux.dims)
    resolutions = np.array(
        [np.abs(np.diff(aux.coords[coord].values)[0]) for coord in coord_names]
    )

    for window_size in [0.1, 0.05, 0.025]:
        smoothed_elev_da = aux["height"].copy(deep=True)

        # Compute gaussian filter scale in terms of grid cells
        scales = window_size / resolutions

        smoothed_elev_da.data = scipy.ndimage.gaussian_filter(
            smoothed_elev_da.data, sigma=scales, mode="nearest"
        )

        TPI_da = aux["height"] - smoothed_elev_da
        aux[f"TPI_{window_size}"] = TPI_da

    return aux


def _coarsen(high_res, ppu: int):
    """
    Coarsen factor for shrinking something high-res to PPU resolution.
    """
    factor = len(high_res) // (ppu)
    return int(factor)


if __name__ == "__main__":
    paths = Paths()
    print(load_elevation_data(paths, 500))
