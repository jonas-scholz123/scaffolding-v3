import urllib.request
from zipfile import ZipFile

import rioxarray as rxr
from loguru import logger
from mlbnb.file import ensure_parent_exists
from xarray import DataArray, Dataset

from scaffolding_v3.config import Paths, SrtmConfig

cfg = SrtmConfig()


def download_srtm(paths: Paths):
    zipped_fpath = paths.raw_data / "srtm" / "zipped" / "srtm.zip"
    ensure_parent_exists(zipped_fpath)
    if not zipped_fpath.exists():
        logger.info("Downloading SRTM dataset.")
        urllib.request.urlretrieve(cfg.srtm_url, zipped_fpath)

    unzipped_fpath = paths.raw_data / "srtm" / "unzipped"
    ensure_parent_exists(unzipped_fpath)
    if not unzipped_fpath.exists():
        logger.info("Unzipping SRTM dataset.")
        with ZipFile(zipped_fpath, "r") as zip_ref:
            zip_ref.extractall(unzipped_fpath)
    logger.success("SRTM dataset downloaded and unzipped.")


def process_srtm(paths: Paths):
    logger.info("Processing SRTM dataset...")
    elevation: Dataset | DataArray = rxr.open_rasterio(
        paths.raw_data / "srtm" / "unzipped" / "srtm_germany_dtm.tif"
    )  # type: ignore
    elevation = elevation.rename(
        {
            "x": "lon",
            "y": "lat",
        }
    )
    elevation = elevation.sel(band=1).drop_vars("band").drop_vars("spatial_ref")
    elevation.name = "height"
    ensure_parent_exists(paths.elevation)
    elevation.to_netcdf(paths.elevation)
    logger.success("SRTM dataset processed.")


if __name__ == "__main__":
    paths = Paths()
    download_srtm(paths)
    process_srtm(paths)
