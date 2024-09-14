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

raw_path = paths.raw_data / "era5" / "era5_t2m.grib"

ensure_parent_exists(raw_path)
ensure_parent_exists(paths.era5)
# %%

client.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": "2023",
        "month": "02",
        "day": "05",
        "time": "04:00",
        #'year': [
        #    '2012', '2013', '2014',
        #    '2015', '2016', '2017',
        #    '2018', '2019', '2020',
        #    '2021', '2022',
        # ],
        #'month': [
        #    '01', '02', '03',
        #    '04', '05', '06',
        #    '07', '08', '09',
        #    '10', '11', '12',
        # ],
        #'day': [
        #    '01', '02', '03',
        #    '04', '05', '06',
        #    '07', '08', '09',
        #    '10', '11', '12',
        #    '13', '14', '15',
        #    '16', '17', '18',
        #    '19', '20', '21',
        #    '22', '23', '24',
        #    '25', '26', '27',
        #    '28', '29', '30',
        #    '31',
        # ],
        #'time': [
        #    '00:00', '01:00', '02:00',
        #    '03:00', '04:00', '05:00',
        #    '06:00', '07:00', '08:00',
        #    '09:00', '10:00', '11:00',
        #    '12:00', '13:00', '14:00',
        #    '15:00', '16:00', '17:00',
        #    '18:00', '19:00', '20:00',
        #    '21:00', '22:00', '23:00',
        # ],
        "area": coords,
        "format": "grib",
    },
    str(raw_path),
)

# %%
ds = xr.open_dataset(raw_path, engine="cfgrib")
ds = ds.rename({"latitude": "lat", "longitude": "lon"})
ds.to_netcdf(paths.era5)

ds = xr.open_dataset(paths.era5)
