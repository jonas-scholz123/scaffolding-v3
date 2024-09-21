import numpy as np
import pandas as pd
from config import DataConfig, Paths
from data.dataprovider import DataProvider, DeepSensorDataset
from data.elevation import load_elevation_data
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from hydra.utils import instantiate
from mlbnb.cache import CachedDataset
from mlbnb.types import Split
from torch.utils.data import Dataset


class TaskLoaderDataset(Dataset):
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


def make_dataset(
    data_config: DataConfig,
    paths: Paths,
    data_provider: DataProvider,
    split: Split,
    data_processor: DataProcessor,
) -> TaskLoaderDataset:

    match split:
        case Split.TRAIN:
            ds = data_provider.get_train_data()
            eval_mode = False
        case Split.VAL:
            ds = data_provider.get_val_data()
            eval_mode = True
        case Split.TEST:
            ds = data_provider.get_test_data()
            eval_mode = True
        case _:
            raise ValueError(f"Unknown split: {split}")

    task_loader = make_taskloader(data_config, paths, data_processor, ds)

    dataset = TaskLoaderDataset(
        task_loader,
        list(ds.times),
        data_config.include_context_in_target,
        eval_mode,
    )

    # Always cache eval datasets to remove randomness
    if data_config.cache or eval_mode:
        return CachedDataset(dataset)  # type: ignore
    return dataset


def make_taskloader(
    data_config: DataConfig,
    paths: Paths,
    data_processor: DataProcessor,
    ds: DeepSensorDataset,
) -> TaskLoader:
    raw_elevation = load_elevation_data(paths, data_config.ppu)
    raw_hires_elevation = load_elevation_data(paths, data_config.hires_ppu)

    # Normalise the data
    context = data_processor(ds.context)
    target = data_processor(ds.target)

    elevation = data_processor(raw_elevation)
    hires_elevation = data_processor(raw_hires_elevation)

    aux_at_targets = hires_elevation if data_config.include_aux_at_targets else None

    # Need to use _partial_ here because of a bug in deepsensor which
    # doesn't pick up a type correctly when instantiated via hydra.
    return instantiate(data_config.task_loader)(
        context=[context, elevation],
        target=[target],
        links=[(0, 0)],
        aux_at_targets=aux_at_targets,
    )
