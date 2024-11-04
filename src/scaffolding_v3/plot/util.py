import deepsensor.torch  # noqa
import numpy as np
import xarray as xr


def make_uniform_grid(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, resolution: float
) -> xr.DataArray:
    lons = np.arange(min_lon, max_lon + resolution, resolution)
    lats = np.arange(min_lat, max_lat + resolution, resolution)

    data = np.zeros((len(lats), len(lons)))
    return xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
    )
