from typing import Literal

import pandas as pd
from deepsensor.data.processor import DataProcessor
from loguru import logger
from mlbnb.rand import sample, split

from scaffolding_v3.config import Paths
from scaffolding_v3.data.dataprovider import DataProvider, DeepSensorDataset
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


class DwdDataProvider(DataProvider):
    def __init__(
        self,
        paths: Paths,
        num_stations: int,
        num_times: int,
        val_fraction: float,
        daily_averaged: bool,
        train_range: tuple[str, str],
        test_range: tuple[str, str],
    ) -> None:
        self.paths = paths
        self.num_stations = num_stations
        self.num_times = num_times
        self.val_fraction = val_fraction
        self.train_range = train_range
        self.test_range = test_range
        self.daily_averaged = daily_averaged

        self.trainset = None
        self.valset = None
        self.testset = None

    def get_train_data(self) -> DeepSensorDataset:
        if self.trainset is None:
            self.trainset, self.valset = self._load_train_val()
        return self.trainset

    def get_val_data(self) -> DeepSensorDataset:
        if self.valset is None:
            self.trainset, self.valset = self._load_train_val()
        return self.valset

    def _load_train_val(self) -> tuple[DeepSensorDataset, DeepSensorDataset]:
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
            "Split data into train: {} val: {}", len(trainset.times), len(valset.times)
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
    ) -> DeepSensorDataset:
        logger.debug(
            "Generating dataset for {} stations at {} times",
            len(target_stations),
            len(times),
        )

        context = df.query("station_id in @context_stations and time in @times")
        target = df.query("station_id in @target_stations and time in @times")

        context = to_deepsensor_df(context)
        target = to_deepsensor_df(target)

        return DeepSensorDataset(context, target, list(times))

    def get_test_data(self) -> DeepSensorDataset:
        if self.testset is None:
            self.testset = self._load_test()
        return self.testset

    def _load_test(self) -> DeepSensorDataset:
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

        return DeepSensorDataset(train, test, list(times))
