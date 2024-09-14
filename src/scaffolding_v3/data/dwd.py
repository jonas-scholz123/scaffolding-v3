from typing import Callable, Iterable, Literal
import pandas as pd
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from loguru import logger
from deepsensor.data.processor import DataProcessor
from deepsensor.data.loader import TaskLoader


from mlbnb.rand import split, sample
from mlbnb.cache import CachedDataset
from scaffolding_v3.data.dataprovider import DataProvider
from scaffolding_v3.config import Paths
from scaffolding_v3.data.elevation import load_elevation_data


def get_dwd_data(
    paths: Paths,
    stations: Literal["all", "train", "test"] = "all",
    date_range: tuple[str, str] = ("2006-01-01", "2024-01-01"),
    columns: list[str] = ["t2m"],
    daily_averaged: bool = False,
) -> pd.DataFrame:
    """
    Load and join the DWD data from the preprocessed feather files.
    """
    logger.info("Loading DWD data for '{}' stations between {}", stations, date_range)
    start = pd.Timestamp(date_range[0])
    end = pd.Timestamp(date_range[1])

    filters = [
        ("time", ">=", start),
        ("time", "<", end),
    ]

    match stations:
        case "train":
            filters += [("is_test_station", "==", False)]
        case "test":
            filters += [("is_test_station", "==", True)]
        case "all":
            pass

    df = pd.read_parquet(paths.dwd, filters=filters, columns=columns)

    if daily_averaged:
        df = df.reset_index()
        df = (
            df.groupby(["station_id", "lat", "lon"])
            .resample("D", on="time")
            .mean()[columns]
        )
        df = (
            df.reset_index()
            .set_index(["time", "station_id", "lat", "lon"])
            .sort_index()
        )
    return df


def get_data_processor(paths: Paths) -> DataProcessor:
    if (paths.data_processor_dir / "data_processor_config.json").exists():
        return DataProcessor(str(paths.data_processor_dir))
    full = get_dwd_data(paths, date_range=("2006-01-01", "2024-01-01"))
    elevation = load_elevation_data(paths, 500)
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


# Combines a primary datasource, like DWD, with secondary datasources
# like elevation, land use, etc.
class CombinedDataset(Dataset):
    def __init__(
        self,
        task_loader: TaskLoader,
        times: list[pd.Timestamp],
        include_context_in_target: bool,
        eval_mode: bool,
    ) -> None:
        self.task_loader = task_loader
        self.times = times
        self.include_context_in_target = include_context_in_target
        self.eval_mode = eval_mode

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

        if self.include_context_in_target:
            return self.task_loader(
                time,
                context_sampling=[context_frac, "all"],
                target_sampling="all",
            )
        else:
            return self.task_loader(
                time,
                context_sampling=["split", "all"],
                target_sampling="split",
                split_frac=context_frac,
            )


class DwdDataProvider(DataProvider):
    def __init__(
        self,
        paths: Paths,
        num_stations: int,
        num_times: int,
        val_fraction: float,
        aux_ppu: int,
        hires_aux_ppu: int,
        cache: bool,
        daily_averaged: bool,
        include_context_in_target: bool,
        include_aux_at_target: bool,
        train_range: tuple[str, str],
        test_range: tuple[str, str],
    ) -> None:
        self.paths = paths
        self.num_stations = num_stations
        self.num_times = num_times
        self.val_fraction = val_fraction
        self.cache = cache
        self.train_range = train_range
        self.test_range = test_range
        self.daily_averaged = daily_averaged
        self.include_context_in_target = include_context_in_target
        self.include_aux_at_target = include_aux_at_target

        self.elevation = load_elevation_data(paths, aux_ppu)
        self.high_res_elevation = load_elevation_data(paths, hires_aux_ppu)
        self.data_processor = get_data_processor(paths)

        self.trainset = None
        self.valset = None
        self.testset = None

    def get_train_data(self) -> Dataset:
        if self.trainset is None:
            self.trainset, self.valset = self._load_train_val()
        return self.trainset

    def get_val_data(self) -> Dataset:
        if self.valset is None:
            self.trainset, self.valset = self._load_train_val()
        return self.valset

    def _load_train_val(self) -> tuple[Dataset, Dataset]:
        df = get_dwd_data(
            self.paths,
            stations="train",
            date_range=self.train_range,
            daily_averaged=self.daily_averaged,
        )

        self._ensure_enough_data(df)

        stations: pd.Series = df.reset_index()["station_id"].drop_duplicates()  # type: ignore

        val_stations, train_stations = split(stations, self.val_fraction)

        times: pd.Series = df.reset_index()["time"].drop_duplicates()  # type: ignore
        num_train_times = int(self.num_times * (1 - self.val_fraction))

        # Want a representative sample of times but avoid close times
        # to prevent leakage
        times = sample(times, self.num_times).sort_values()
        train_times: pd.Series = times[:num_train_times]  # type: ignore
        val_times: pd.Series = times[num_train_times:]  # type: ignore

        trainset = self._split_into_dataset(
            df, train_stations, train_stations, train_times, False
        )
        valset = self._split_into_dataset(
            df, train_stations, val_stations, val_times, True
        )

        logger.info(
            "Split data into train: {} val: {}",
            len(trainset),  # type: ignore
            len(valset),  # type: ignore
        )

        return trainset, valset

    def _ensure_enough_data(self, df: pd.DataFrame) -> None:
        total_num_stations = len(df.reset_index()["station_id"].drop_duplicates())
        if self.num_stations > total_num_stations:
            logger.warning(
                "Requested {} stations, but only {} are available. Using all stations.",
                self.num_stations,
                total_num_stations,
            )
            self.num_stations = total_num_stations

        total_num_times = len(df.reset_index()["time"].drop_duplicates())
        if self.num_times > total_num_times:
            logger.warning(
                "Requested {} times, but only {} are available. Using all times.",
                self.num_times,
                total_num_times,
            )
            self.num_times = total_num_times

    def _split_into_dataset(
        self,
        df: pd.DataFrame,
        context_stations: pd.Series,
        target_stations: pd.Series,
        times: pd.Series,
        eval_mode: bool,
    ) -> Dataset:

        logger.debug(
            "Generating dataset for {} stations at {} times",
            len(target_stations),
            len(times),
        )

        context = df.query("station_id in @context_stations and time in @times")
        target = df.query("station_id in @target_stations and time in @times")

        context = to_deepsensor_df(context)
        target = to_deepsensor_df(target)

        return self._to_dataset(context, target, times, eval_mode)

    def _to_dataset(
        self,
        raw_context: pd.DataFrame,
        raw_target: pd.DataFrame,
        times: Iterable[pd.Timestamp],
        eval_mode: bool,
    ) -> Dataset:
        context = to_deepsensor_df(raw_context)
        target = to_deepsensor_df(raw_target)

        context: pd.DataFrame = self.data_processor(context)  # type: ignore
        target: pd.DataFrame = self.data_processor(target)  # type: ignore

        aux: xr.Dataset = self.data_processor(self.elevation)  # type: ignore
        hires_aux: xr.Dataset = self.data_processor(self.high_res_elevation)  # type: ignore

        aux_at_target = hires_aux if self.include_aux_at_target else None

        task_loader = TaskLoader(
            context=[raw_context, aux],
            target=[raw_target],
            discrete_xarray_sampling=True,  # TODO: into config
            aux_at_targets=hires_aux,
            links=[(0, 0)],
        )

        dataset = CombinedDataset(
            task_loader,
            list(times),
            self.include_context_in_target,
            eval_mode,
        )

        # Always cache eval datasets to remove randomness
        if self.cache or eval_mode:
            return CachedDataset(dataset)
        return dataset

    def get_test_data(self) -> Dataset:
        if self.testset is None:
            self.testset = self._load_test()
        return self.testset

    def _load_test(self) -> Dataset:
        train = get_dwd_data(
            self.paths,
            stations="train",
            date_range=self.test_range,
            daily_averaged=self.daily_averaged,
        )
        test = get_dwd_data(
            self.paths,
            stations="test",
            date_range=self.test_range,
            daily_averaged=self.daily_averaged,
        )

        times = test.reset_index()["time"].drop_duplicates()
        return self._to_dataset(train, test, times, True)

    def get_collate_fn(self) -> Callable:
        return lambda x: x
