from abc import abstractmethod, ABC
from torch.utils.data import Dataset
from typing import Callable


class DataProvider(ABC):
    """
    Interface for data providers.

    Data providers are responsible for splitting the data into
    non-overlapping train, validation, and test sets.
    """

    @abstractmethod
    def get_train_data(self) -> Dataset:
        pass

    @abstractmethod
    def get_val_data(self) -> Dataset:
        pass

    @abstractmethod
    def get_test_data(self) -> Dataset:
        pass

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        pass
