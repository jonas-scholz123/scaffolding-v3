#%%
from pathlib import Path
from typing import Optional, Callable
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from loguru import logger
import deepsensor.torch
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader


from mlbnb.rand import split, sample
from mlbnb.cache import CachedDataset
from scaffolding_v3.data.dataprovider import DataProvider
from scaffolding_v3.config import Paths
from scaffolding_v3.data.elevation import load_elevation_data
from tqdm import tqdm


def get_dwd_data(paths: Paths) -> gpd.GeoDataFrame:
    """
    Load and join the DWD data from the preprocessed feather files.
    """
    logger.info("Loading DWD data")
    meta_df = gpd.read_feather(paths.dwd_meta)

    chunks = []

    df = pd.read_feather(paths.dwd)

    chunk_size = 100000
    
    for chunk_idx in tqdm(range(0, len(df), chunk_size)):
        chunk = df.iloc[chunk_idx:chunk_idx + chunk_size]
        chunk = meta_df.merge(chunk, on="station_id")
        chunk["date_str"] = pd.to_datetime(chunk["time"]).dt.strftime("%Y-%m-%d")
        chunk = chunk.query("time <= to_date and time >= from_date")
        chunk = chunk.set_index(["time", "station_id"])
        chunk = chunk.drop(["from_date", "to_date", "date_str"], axis=1)
        chunks.append(chunk)

    df = pd.concat(chunks)

    return df


def get_data_processor(
    paths: Paths, full: pd.DataFrame, elevation: xr.Dataset
) -> DataProcessor:
    if (paths.data_processor_dir / "data_processor_config.json").exists():
        return DataProcessor(str(paths.data_processor_dir))
    logger.info("Caching data processor.")
    df_unnormed = to_deepsensor_df(full)
    data_processor = DataProcessor(x1_name="lat", x2_name="lon")
    data_processor(df_unnormed)

    data_processor(elevation, method="min_max")

    data_processor.save(str(paths.data_processor_dir))
    logger.success("Data processor cached.")
    return data_processor


def to_deepsensor_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe indexed by [time, lat, lon] with only the temperature column T2M.
    """
    return df.reset_index().set_index(["time", "lat", "lon"])[["t2m"]]  # type: ignore


class DwdStationDataset(Dataset):
    def __init__(
        self,
        raw_context: pd.DataFrame,
        raw_target: pd.DataFrame,
        raw_aux: xr.Dataset,
        hires_raw_aux: xr.Dataset,
        times: list[pd.Timestamp],
        data_processor: DataProcessor,
        eval_mode: bool,
    ):
        self.times = times
        self.eval_mode = eval_mode
        self.data_processor = data_processor
        context: pd.DataFrame = self.data_processor(raw_context)  # type: ignore
        target: pd.DataFrame = self.data_processor(raw_target)  # type: ignore
        aux: xr.Dataset = self.data_processor(raw_aux)  # type: ignore
        high_res_aux: xr.Dataset = self.data_processor(hires_raw_aux)  # type: ignore

        self.task_loader = TaskLoader(
            context=[context, aux],
            target=[target],
            time_freq="h",
            discrete_xarray_sampling=True,  # TODO: into config
            aux_at_targets=high_res_aux,
        )
        self.task_loader.load_dask()

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        time = self.times[idx]
        if self.eval_mode:
            return self.task_loader(
                time, context_sampling=["all", "all"], target_sampling="all"
            )

        context_frac = np.random.rand()

        return self.task_loader(
            time,
            context_sampling=[context_frac, "all"],
            target_sampling="all",
        )


class DwdDataProvider(DataProvider):
    def __init__(
        self,
        paths: Paths,
        num_stations: int,
        num_times: int,
        val_fraction: float,
        aux_ppu: int,
        cache: bool,
        daily_averaged: bool,
    ) -> None:
        self.paths = paths
        self.num_stations = num_stations
        self.num_times = num_times
        self.val_fraction = val_fraction
        self.cache = cache

        self.df = get_dwd_data(paths)

        self.station_splits = pd.read_feather(paths.station_splits)
        self.time_splits = pd.read_feather(paths.time_splits)

        self.elevation = load_elevation_data(paths, aux_ppu)
        self.high_res_elevation = load_elevation_data(paths, 500)

        if daily_averaged:
            self.df = self.df.reset_index()
            self.df = self.df.groupby(["lat", "lon", "station_id", "station_name", "geometry"]).resample("D", on="time").mean()[["t2m"]]
            self.df = self.df.reset_index().set_index(["time", "lat", "lon"]).sort_index()

            self.time_splits = self.time_splits.resample("D", on="time").first().reset_index()

        self.data_processor = get_data_processor(paths, self.df, self.elevation)


        total_num_stations = len(self.station_splits.query("set == 'trainval'"))
        if num_stations > total_num_stations:
            logger.warning(
                "Requested {} stations, but only {} are available. Using all stations.",
                num_stations,
                total_num_stations,
            )
            self.num_stations = total_num_stations
        
        total_num_times = len(self.time_splits.query("set == 'train' or set == 'val'"))
        if num_times > total_num_times:
            logger.warning(
                "Requested {} times, but only {} are available. Using all times.",
                num_times,
                total_num_times,
            )
            self.num_times = total_num_times


        self.trainset = None
        self.valset = None
        self.testset = None

    def _split_train_val_test(self) -> tuple[Dataset, Dataset, Dataset]:
        stations = self.station_splits.query("set == 'trainval'")
        stations = stations.sort_values("order", ascending=True)
        stations = stations.head(self.num_stations)

        val_stations, train_stations = split(stations["station_id"], self.val_fraction)
        train_stations = list(train_stations)
        val_stations = list(val_stations)

        num_train_times = int(self.num_times * (1 - self.val_fraction))
        num_val_times = self.num_times - num_train_times

        train_times = self.time_splits.query("set == 'train'")["time"]
        train_times = list(sample(train_times, num_train_times))  # type: ignore

        val_times = self.time_splits.query("set == 'val'")["time"]
        val_times = list(sample(val_times, num_val_times))  # type: ignore

        test_ids = list(self.station_splits.query("set == 'test'")["station_id"])
        test_times = list(self.time_splits.query("set == 'test'")["time"])

        trainset = self.gen_trainset(train_stations, train_stations, train_times, False)
        valset = self.gen_trainset(train_stations, val_stations, val_times, True)
        testset = self.gen_trainset(train_stations, test_ids, test_times, True)

        logger.info(
            "Split times into train: {} val: {} test: {}",
            len(trainset),
            len(valset),
            len(testset),
        )

        return trainset, valset, testset

    def gen_trainset(
        self,
        context_stations: list[int],
        target_stations: list[int],
        times: list[pd.Timestamp],
        eval_mode: bool,
    ) -> DwdStationDataset:

        logger.debug("Generating dataset for {} stations at {} times", len(target_stations), len(times))

        context = self.df.query("station_id in @context_stations and time in @times")
        target = self.df.query("station_id in @target_stations and time in @times")

        context = to_deepsensor_df(context)
        target = to_deepsensor_df(target)

        dataset =  DwdStationDataset(
            context,
            target,
            self.elevation,
            self.high_res_elevation,
            times,
            self.data_processor,
            eval_mode,
        )


        if self.cache:
            return CachedDataset(dataset)

    def get_train_data(self) -> Dataset:
        if self.trainset is None:
            self.trainset, self.valset, self.testset = self._split_train_val_test()
        return self.trainset

    def get_val_data(self) -> Dataset:
        if self.valset is None:
            self.trainset, self.valset, self.testset = self._split_train_val_test()
        return self.valset

    def get_test_data(self) -> Dataset:
        if self.testset is None:
            self.trainset, self.valset, self.testset = self._split_train_val_test()
        return self.testset

    def get_collate_fn(self) -> Callable:
        return lambda x: x

class DailyAveragedDwdDataProvider(DwdDataProvider):
    """
    Computes daily mean temperatures for each station.
    """
    def __init__(
        self,
        paths: Paths,
        num_stations: int,
        num_times: int,
        val_fraction: float,
        aux_ppu: int,
        cache: bool,
    ) -> None:
        super().__init__(paths, num_stations, num_times, val_fraction, aux_ppu, cache, True)

    def gen_trainset(
        self,
        context_stations: list[int],
        target_stations: list[int],
        times: list[pd.Timestamp],
        eval_mode: bool,
    ) -> DwdStationDataset:
        
        logger.debug("Generating dataset for {} stations at {} times", len(target_stations), len(times))

        context = self.df.query("station_id in @context_stations and time in @times")
        target = self.df.query("station_id in @target_stations and time in @times")

        context = to_deepsensor_df(context)
        target = to_deepsensor_df(target)

        target = target.groupby("station_id").resample("D").mean().reset_index()
        target = target.set_index(["time", "station_id"])[["t2m"]]

        dataset =  DwdStationDataset(
            context,
            target,
            self.elevation,
            self.high_res_elevation,
            times,
            self.data_processor,
            eval_mode,
        )

        if self.cache:
            return CachedDataset(dataset)

#%%
if __name__ == "__main__":
    data = DwdDataProvider(Paths(), 500, 1000, 0.2, 500)
    train = data.get_train_data()