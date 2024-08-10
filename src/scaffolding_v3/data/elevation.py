from scaffolding_v3.config import Paths
import xarray as xr


def load_elevation_data(paths: Paths, ppu: int) -> xr.Dataset:
    elevation = xr.load_dataset(paths.elevation)
    coarsen = {
        "lat": _coarsen(elevation["lat"], ppu),
        "lon": _coarsen(elevation["lon"], ppu),
    }

    elevation = elevation.coarsen(coarsen, boundary="trim").mean()
    return elevation


def _coarsen(high_res, ppu: int):
    """
    Coarsen factor for shrinking something high-res to PPU resolution.
    """
    factor = len(high_res) // (ppu)
    return int(factor)


if __name__ == "__main__":
    paths = Paths()
    print(load_elevation_data(paths, 500))
