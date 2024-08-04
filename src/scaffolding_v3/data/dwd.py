from pathlib import Path
from typing import Optional, Callable
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from loguru import logger
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader


from mlbnb.rand import split, sample
from scaffolding_v3.data.dataprovider import DataProvider
from scaffolding_v3.config import Paths
from scaffolding_v3.data.elevation import load_elevation_data


def get_dwd_data(paths: Paths):
    """
    Load and join the DWD data from the preprocessed feather files.
    """
    logger.info("Loading DWD data")
    meta_df = gpd.read_feather(paths.dwd_meta)
    df = pd.read_feather(paths.dwd)
    df = meta_df.merge(df, on="station_id")
    df = df.set_index(["time", "station_id"])
    df = df.drop(["from_date", "to_date"], axis=1)
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
        times: list[pd.Timestamp],
        data_processor: DataProcessor,
        eval_mode: bool,
        paths: Paths,
    ):
        self.times = times
        self.eval_mode = eval_mode
        self.data_processor = data_processor
        context: pd.DataFrame = self.data_processor(raw_context)  # type: ignore
        target: pd.DataFrame = self.data_processor(raw_target)  # type: ignore
        aux: xr.Dataset = self.data_processor(raw_aux)  # type: ignore

        self.task_loader = TaskLoader(
            context=[context, aux],
            target=[target],
            time_freq="h",
            discrete_xarray_sampling=True,  # TODO: into config
        )

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
    ) -> None:
        self.paths = paths
        self.num_stations = num_stations
        self.num_times = num_times
        self.val_fraction = val_fraction

        self.df = get_dwd_data(paths)
        self.station_splits = pd.read_feather(paths.station_splits)
        self.time_splits = pd.read_feather(paths.time_splits)
        self.elevation = load_elevation_data(paths, aux_ppu)

        self.data_processor = get_data_processor(paths, self.df, self.elevation)

        self.trainset = None
        self.valset = None

    def _split_train_val_test(self) -> tuple[Dataset, Dataset, Dataset]:
        stations = self.station_splits.query("set == 'trainval'")
        stations = stations.sort_values("order", ascending=True)
        stations = stations.head(self.num_stations)

        train_stations, val_stations = split(stations["station_id"], self.val_fraction)
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

        return trainset, valset, testset

    def gen_trainset(
        self,
        context_stations: list[int],
        target_stations: list[int],
        times: list[pd.Timestamp],
        eval_mode: bool,
    ) -> DwdStationDataset:
        context = self.df.query("station_id in @context_stations and time in @times")
        target = self.df.query("station_id in @target_stations and time in @times")

        context = to_deepsensor_df(context)
        target = to_deepsensor_df(target)

        return DwdStationDataset(
            context,
            target,
            self.elevation,
            times,
            self.data_processor,
            eval_mode,
            self.paths,
        )

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


if __name__ == "__main__":
    data = DwdDataProvider(Paths(), 100, 1000, 0.2, 500)
    train = data.get_train_data()
    print(next(iter(train)))
