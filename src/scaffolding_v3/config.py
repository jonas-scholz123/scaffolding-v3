from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf.omegaconf import MISSING


class CheckpointOccasion(Enum):
    BEST = "best"
    LATEST = "latest"


@dataclass
class Paths:
    data: Path
    raw_data: Path
    output: Path
    weights: Path


@dataclass
class DataConfig:
    dataset: dict
    trainloader: dict
    testloader: dict
    cache: bool
    in_channels: int
    num_classes: int
    sidelength: int


@dataclass
class ExecutionConfig:
    dry_run: bool
    epochs: int
    seed: int
    start_from: Optional[CheckpointOccasion]
    start_weights: Optional[str]


@dataclass
class OutputConfig:
    save_checkpoints: bool
    out_dir: Path
    use_wandb: bool
    wandb_project: str
    log_gradients: bool
    gradient_log_freq: int
    use_tqdm: bool
    log_level: str
    plot: bool
    sample_indices: tuple[int, ...]


@dataclass
class RuntimeConfig:
    device: str
    root: str


@dataclass
class Config:
    runtime: RuntimeConfig = MISSING
    data: DataConfig = MISSING
    output: OutputConfig = MISSING
    paths: Paths = MISSING
    execution: ExecutionConfig = MISSING
    model: dict = MISSING
    loss: dict = MISSING
    optimizer: dict = MISSING
    scheduler: dict = MISSING


def _get_runtime_cfg() -> RuntimeConfig:
    """
    Get the runtime configuration, containing values that the yaml config needs that are
    only available at runtime.
    """
    root = str(Path(__file__).resolve().parent.parent.parent)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return RuntimeConfig(device=device, root=root)


def init_config() -> ConfigStore:
    cs = ConfigStore.instance()

    cs.store(name="base_cfg", node=Config(runtime=_get_runtime_cfg()))

    return cs
