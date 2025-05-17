from typing import Any

import torch
from mlbnb.types import Split
from torch.utils.data import Dataset


def make_dataset(
    data_partial: Any, split: Split, generator: torch.Generator
) -> Dataset:
    return data_partial.dataset(split=split, generator=generator)
