from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf.omegaconf import MISSING

# This isn't perfect, the type annotation approach is nicer but doesn't work with omegaconf
SKIP_KEYS = {
    "output",
    "start_from",
    "_partial_",
    "root",
    "testloader",
    "paths",
    "epochs",
}


@dataclass
class Paths:
    root: Path
    data: Path
    raw_data: Path
    output: Path
    weights: Path


@dataclass
class DatasetConfig:
    paths: Paths
    val_fraction: float
    _target_: str
    _partial_: bool


@dataclass
class DataConfig:
    cache: bool
    in_channels: int
    num_classes: int
    sidelength: int
    trainloader: dict
    testloader: dict
    dataset: DatasetConfig


class CheckpointOccasion(Enum):
    BEST = "best"
    LATEST = "latest"


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
    plotter: dict


@dataclass
class RuntimeConfig:
    device: str
    root: Path


@dataclass
class Config:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = MISSING
    generator: dict = MISSING
    model: dict = MISSING
    loss: dict = MISSING
    optimizer: dict = MISSING
    output: OutputConfig = MISSING
    scheduler: dict = MISSING
    paths: Paths = MISSING
    execution: ExecutionConfig = MISSING


def _get_runtime_cfg() -> RuntimeConfig:
    """
    Get the runtime configuration, containing values that the yaml config needs that are
    only available at runtime.
    """
    root = Path(__file__).resolve().parent.parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return RuntimeConfig(device=device, root=root)


def init_configs() -> ConfigStore:
    cs = ConfigStore.instance()

    cs.store(name="base_cfg", node=Config(runtime=_get_runtime_cfg()))

    return cs
