from pathlib import Path
from mlbnb.dataprovider import DataProvider
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Normalize, ToTensor, Compose
import torch


class MnistDataProvider(DataProvider):
    def __init__(self, root: Path, normalization: Normalize, val_fraction: float):
        self._root = root
        self._transform = Compose([ToTensor(), normalization])
        self._val_fraction = val_fraction
        self._train_dataset, self._val_dataset = self._load_and_split_data()

    def _load_and_split_data(self) -> tuple[Dataset, Dataset]:
        full_dataset = MNIST(
            root=self._root, train=True, download=True, transform=self._transform
        )

        dataset_size = len(full_dataset)
        val_size = int(dataset_size * self._val_fraction)
        train_size = dataset_size - val_size

        self._train_dataset, self._val_dataset = random_split(
            full_dataset,
            [1 - self._val_fraction, self._val_fraction],
            generator=torch.default_generator,
        )
        return self._train_dataset, self._val_dataset

    def get_train_data(self) -> Dataset:
        return self._train_dataset

    def get_val_data(self) -> Dataset:
        return self._val_dataset

    def get_test_data(self) -> Dataset:
        return MNIST(
            root=self._root, train=False, download=True, transform=self._transform
        )
