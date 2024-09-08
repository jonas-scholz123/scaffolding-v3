# %%
from torch.utils.data import Dataset
from typing import Callable
import xarray as xr
from scaffolding_v3.config import Paths
from scaffolding_v3.data.dataprovider import DataProvider
from scaffolding_v3.data.dwd import get_data_processor

import deepsensor.torch
from deepsensor.data.loader import TaskLoader
from deepsensor import plot

paths = Paths()

def load_era5() -> xr.Dataset:
    return xr.open_dataset(paths.era5)

class Era5Dataset(Dataset):
    def __init__(self, ds: xr.Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        return self.ds.isel(time=idx)

class Era5DataProvider(DataProvider):
    def __init__(self):
        self.ds = load_era5()
        self.trainset = None
        self.valset = None
        self.testset = None

    def get_train_data(self) -> Dataset:
        return Era5Dataset(self.ds)

    def get_val_data(self) -> Dataset:
        return Era5Dataset(self.ds)

    def get_test_data(self) -> Dataset:
        return Era5Dataset(self.ds)

    def get_collate_fn(self) -> Callable:
        return lambda x: x

# %%
raw_era5 = load_era5()
data_processor = get_data_processor(paths)

era5 = data_processor(raw_era5)

# %%
task_loader = TaskLoader(
    context = [era5],
    target = [era5]
)

test_time = era5.time.values
task = task_loader(test_time, context_sampling=0.03, target_sampling="all")
plot.task(task, task_loader)
# %%
