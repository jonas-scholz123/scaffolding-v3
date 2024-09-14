from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import xarray as xr


@dataclass
class DeepSensorDataset:
    context: pd.DataFrame | xr.Dataset
    target: pd.DataFrame | xr.Dataset
    times: list[pd.Timestamp]


class DataProvider(ABC):
    """
    Interface for data providers.

    Data providers are responsible for splitting the data into
    non-overlapping train, validation, and test sets.
    """

    @abstractmethod
    def get_train_data(self) -> DeepSensorDataset:
        pass

    @abstractmethod
    def get_val_data(self) -> DeepSensorDataset:
        pass

    @abstractmethod
    def get_test_data(self) -> DeepSensorDataset:
        pass
