from mlbnb.types import Split
import torch
from torch.utils.data import Dataset

from scaffolding_v3.config import DataConfig
from hydra.utils import instantiate


def make_dataset(
    data_cfg: DataConfig, split: Split, generator: torch.Generator
) -> Dataset:
    return instantiate(data_cfg.dataset, split=split, generator=generator)
