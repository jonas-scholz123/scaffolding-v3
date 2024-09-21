# %%
from scaffolding_v3.config import GeoConfig, Era5Config, Paths
from dotenv import load_dotenv
from mlbnb.file import ensure_parent_exists
import os
import cdsapi
import xarray as xr

era5_config = Era5Config()
paths = Paths()
geo_config = GeoConfig()

load_dotenv()

key = os.getenv("CDS_API_KEY")
client = cdsapi.Client(key=key, url=era5_config.era5_url)
coords = [
    geo_config.max_lat,
    geo_config.min_lon,
    geo_config.min_lat,
    geo_config.max_lon,
]

START_YEAR = 2006
END_YEAR = 2011

raw_path = paths.raw_data / "era5" / f"era5_t2m_{START_YEAR}-{END_YEAR}.grib"
ensure_parent_exists(raw_path)
ensure_parent_exists(paths.era5)


def get_all_hours() -> list[str]:
    return [f"{i:02d}:00" for i in range(24)]


def get_all_days() -> list[str]:
    return [f"{i:02d}" for i in range(1, 32)]


def get_all_months() -> list[str]:
    return [f"{i:02d}" for i in range(1, 13)]


def get_years_inclusive(start: int, end: int) -> list[str]:
    return [str(i) for i in range(start, end + 1)]


# %%
config = {
    "product_type": "reanalysis",
    "variable": "2m_temperature",
    "year": get_years_inclusive(START_YEAR, END_YEAR),
    "month": get_all_months(),
    "day": get_all_days(),
    "time": get_all_hours(),
    "area": coords,
    "format": "grib",
}

client.retrieve(
    "reanalysis-era5-single-levels",
    config,
    str(raw_path),
)
# %%

ds = xr.open_dataset(raw_path, engine="cfgrib")
# %%
ds = ds.rename({"latitude": "lat", "longitude": "lon"})
ds.to_netcdf(paths.era5)
