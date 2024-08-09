import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Tuple
import urllib.request
import os
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile, BadZipFile
from scaffolding_v3.config import DwdConfig, Paths
from scaffolding_v3.data.dwd import get_dwd_data
from mlbnb.file import ensure_parent_exists
import geopandas as gpd
from loguru import logger

paths = Paths()
dwd_cfg = DwdConfig()

zipped_dir = paths.raw_data / "dwd" / "airtemp2m2" / "zipped"
unzipped_dir = paths.raw_data / "dwd" / "airtemp2m2" / "unzipped"


def download_dwd(urls: list[str]):
    os.makedirs(zipped_dir, exist_ok=True)
    os.makedirs(unzipped_dir, exist_ok=True)

    logger.info("Downloading DWD data")
    for url in tqdm(urls):
        fname = url.split("/")[-1]

        if url.endswith("zip"):
            out_fpath = zipped_dir / fname
        else:
            out_fpath = unzipped_dir / fname
        if os.path.exists(out_fpath):
            continue
        urllib.request.urlretrieve(url, out_fpath)

    logger.info("Unzipping DWD data")
    for url in tqdm(urls):
        fname = url.split("/")[-1]
        if not fname.endswith("zip"):
            continue

        zip_fpath = f"{zipped_dir}/{fname}"

        fname_no_ext = fname.split(".")[0]
        out_dir = f"{unzipped_dir}/{fname_no_ext}"
        os.makedirs(out_dir, exist_ok=True)

        try:
            with ZipFile(zip_fpath, "r") as f:
                f.extractall(out_dir)
        except BadZipFile:
            logger.warning(
                "{} is corrupted. Please delete it and restart the process.", fname
            )


def load_station_df(fpath):
    df = pd.read_csv(fpath, delimiter=";", encoding="latin-1")
    df.columns = ["station_id", "time", "QN_9", "t2m", "RF_TU", "eor"]
    # drop relative humidity, "end of record" column.
    df = df.drop(["RF_TU", "eor", "QN_9"], axis=1)
    # Filter date.
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H")

    # Filter invalid values.
    df = df[df["t2m"] != -999.0]
    return df


def load_station_metadata(fpath):
    meta_columns = [
        "station_id",
        "height",
        "lat",
        "lon",
        "from_date",
        "to_date",
        "station_name",
    ]
    df = pd.read_csv(fpath, delimiter=";", encoding="latin-1")
    df.columns = meta_columns

    df["from_date"] = pd.to_datetime(df["from_date"], format="%Y%m%d")

    # Last "to_date" is empty because it represents current location.
    df["to_date"] = pd.to_datetime(df["to_date"], format="%Y%m%d", errors="coerce")
    df.loc[df.index[-1], "to_date"] = pd.to_datetime("today").normalize()
    return df


def process_dwd():
    dfs = []
    meta_dfs = []
    root = unzipped_dir
    for subdir in tqdm(os.listdir(root)):
        dir_path = root / subdir
        if not os.path.isdir(dir_path):
            continue

        # Data:
        fname = [n for n in os.listdir(dir_path) if n.startswith("produkt")][0]
        fpath = root / subdir / fname
        dfs.append(load_station_df(fpath))

        # Metadata:
        fname = [n for n in os.listdir(dir_path) if n.startswith("Metadaten_Geo")][0]
        fpath = root / subdir / fname
        meta_dfs.append(load_station_metadata(fpath))

    df = pd.concat(dfs, ignore_index=True)
    meta_df = pd.concat(meta_dfs, ignore_index=True)
    geometry = gpd.points_from_xy(meta_df.lon, meta_df.lat)
    meta_df = gpd.GeoDataFrame(meta_df, geometry=geometry)
    meta_df.crs = dwd_cfg.crs_str

    # Filter days with not enoungh data. (Mainly at the start of dataset period).
    counts = df.set_index("time").groupby("time").count()
    good_times = counts[counts["station_id"] > 400].index.unique()
    df = df[df["time"].isin(good_times)].reset_index(drop=True)  # type: ignore

    # cache for faster loading in future.
    ensure_parent_exists(paths.dwd)
    ensure_parent_exists(paths.dwd_meta)
    df.to_feather(paths.dwd)
    meta_df.to_feather(paths.dwd_meta)


def download_value_stations():
    zip_fpath = paths.raw_data / "dwd" / "value" / "value_stations.zip"
    ensure_parent_exists(zip_fpath)

    if not zip_fpath.exists():
        url = dwd_cfg.value_url
        urllib.request.urlretrieve(url, zip_fpath)

    with ZipFile(zip_fpath, "r") as f:
        f.extract("VALUE_53_ECAD_Germany_v1/stations.txt", paths.raw_data / "value")


def process_value_stations() -> None:
    df = pd.read_csv(
        paths.raw_data / "value" / "VALUE_53_ECAD_Germany_v1" / "stations.txt"
    )
    df.columns = [
        "station_id",
        "station_name",
        "lon",
        "lat",
        "height",
        "source",
    ]

    # StationID does not match with DWD dataset.
    df = df.drop(["station_id", "source"], axis=1)

    # Strip whitespace:
    df["station_name"] = df["station_name"].str.strip()

    geometry = gpd.points_from_xy(df["lon"], df["lat"])
    df = gpd.GeoDataFrame(df, geometry=geometry)
    df.crs = dwd_cfg.crs_str
    df.to_feather(paths.value_stations)


def train_val_test_dts(dts):
    """
    This follows the google paper's sampling strategy.
    """
    dts = list(dts)

    train, val, test = [], [], []

    # 19 days.
    train_duration = 19 * 24
    # 2 days.
    val_duration = 2 * 24
    # 2.5 days.
    test_duration = 5 * 12

    # 2 days skipped at borders.
    skip_duration = 2 * 24

    i = 0
    while i < len(dts):
        # Add training datetimes.
        train += dts[i : i + train_duration]
        i += train_duration + skip_duration
        if i >= len(dts):
            break

        val += dts[i : i + val_duration]
        i += val_duration + skip_duration
        if i >= len(dts):
            break

        test += dts[i : i + test_duration]
        i += test_duration + skip_duration
        if i >= len(dts):
            break

    # Make sure there's no overlap.
    assert (
        set(train).isdisjoint(val)
        and set(val).isdisjoint(test)
        and set(test).isdisjoint(train)
    )

    return train, val, test


def split(df, dts, station_ids) -> Tuple[pd.DataFrame]:
    """
    Split a dataframe by BOTH datetimes and station ids.

    Returns: (split, remainder): pd.DataFrame
    """

    split = df.query("station_id in @station_ids and time in @dts")
    remainder = df.query("station_id not in @station_ids and time not in @dts")

    return split, remainder  # type: ignore


def get_test_station_ids():

    df = pd.read_feather(paths.value_stations)

    # Strip whitespace:
    df["station_name"] = df["station_name"].str.strip()

    geometry = gpd.points_from_xy(df["lon"], df["lat"])
    df = gpd.GeoDataFrame(df, geometry=geometry)
    df.crs = dwd_cfg.crs_str

    # This is a projected crs, so we can use distance as a metric.
    projected_crs = "EPSG:25832"
    meta_df = gpd.read_feather(paths.dwd_meta)
    meta_df = meta_df.to_crs(projected_crs)

    df = df.to_crs(projected_crs)

    test_station_ids = set(df.sjoin_nearest(meta_df)["station_id"])  # type: ignore
    return test_station_ids


def distance_matrix(gdf1, gdf2):
    # Station distance matrix:
    return gdf1.geometry.apply(lambda g: gdf2.distance(g))


def save_station_splits():
    full = get_dwd_data(paths)
    gdf = full.groupby("station_id").first()
    station_ids = gdf.index.get_level_values("station_id").sort_values()
    sdf = pd.DataFrame(index=station_ids)
    sdf["set"] = "trainval"
    sdf["order"] = 0

    test_station_ids = get_test_station_ids()

    # Define test stations.
    sdf.loc[list(test_station_ids), "set"] = "test"

    # Remaining stations are for training (+ validation).
    train_station_ids = list(set(station_ids) - set(test_station_ids))

    # Set order so that first 20 stations are subset of first 100 stations etc.
    sdf.loc[train_station_ids, "order"] = list(range(len(train_station_ids)))

    sdf = sdf.reset_index()

    ensure_parent_exists(paths.station_splits)
    sdf.to_feather(paths.station_splits)


def save_datetime_splits():
    full = get_dwd_data(paths)
    dts = full.index.get_level_values("time").unique()
    train_dts, val_dts, test_dts = train_val_test_dts(dts)
    df = pd.DataFrame(index=dts.sort_values())
    df["set"] = None
    df.loc[train_dts] = "train"
    df.loc[val_dts] = "val"
    df.loc[test_dts] = "test"
    df = df.reset_index()
    ensure_parent_exists(paths.time_splits)
    df.to_feather(paths.time_splits)


def extract_links() -> list[str]:
    response = requests.get(dwd_cfg.dwd_url)
    data_links = []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=True)

        for link in links:
            href = link["href"]
            full_url = urljoin(dwd_cfg.dwd_url, href)
            if full_url.endswith(".zip") or full_url.endswith(".txt"):
                data_links.append(full_url)

        return data_links
    else:
        raise ValueError(
            f"Failed to fetch the page. Status code: {response.status_code}"
        )


if __name__ == "__main__":
    if not paths.dwd.exists() or not paths.dwd_meta.exists():
        logger.info("Downloading and processing DWD data.")
        data_links = extract_links()
        download_dwd(data_links)
        process_dwd()
        logger.success("DWD data downloaded and processed.")
    if not paths.value_stations.exists():
        logger.info("Downloading and processing value stations.")
        download_value_stations()
        process_value_stations()
        logger.success("Value stations downloaded and processed.")
    if not paths.station_splits.exists() or not paths.time_splits.exists():
        logger.info("Defining data splits.")
        save_station_splits()
        save_datetime_splits()
        logger.success("Data splits defined.")
