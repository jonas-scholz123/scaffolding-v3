import torch
from hydra.utils import instantiate
from mlbnb.types import Split
from torch.utils.data import Dataset

from config.config import DataConfig


def make_dataset(
    data_cfg: DataConfig, split: Split, generator: torch.Generator
) -> Dataset:
    return instantiate(data_cfg.dataset, split=split, generator=generator)
