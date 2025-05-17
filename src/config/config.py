from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

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

root = Path(__file__).resolve().parent.parent.parent


@dataclass
class Paths:
    root: Path = root
    data: Path = root / "_data"
    raw_data: Path = data / "_raw"
    output: Path = root / "_output"
    weights: Path = root / "_weights"


@dataclass
class ModelConfig:
    _target_: str = "scaffolding_v3.model.convnet.ConvNet"
    conv_channels: tuple[int, ...] = (32, 64)
    linear_channels: tuple[int, ...] = (128,)
    use_dropout: bool = True


@dataclass
class LossConfig:
    _target_: str = "torch.nn.NLLLoss"


@dataclass
class OptimizerConfig:
    lr: float = 0.1


@dataclass
class AdamConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"


@dataclass
class AdadeltaConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adadelta"


@dataclass
class DataloaderConfig:
    batch_size: int
    shuffle: bool
    _target_: str = "torch.utils.data.DataLoader"


@dataclass
class TrainLoaderConfig(DataloaderConfig):
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 0
    multiprocessing_context: Optional[str] = None


@dataclass
class TestLoaderConfig(DataloaderConfig):
    batch_size: int = 1000
    shuffle: bool = False


@dataclass
class DatasetConfig:
    paths: Paths = field(default_factory=Paths)
    val_fraction: float = 0.1


@dataclass
class MnistDatasetConfig(DatasetConfig):
    _target_: str = "scaffolding_v3.data.mnist.MnistDataset"


@dataclass
class Cifar10DatasetConfig(DatasetConfig):
    _target_: str = "scaffolding_v3.data.cifar10.Cifar10Dataset"


@dataclass
class DataConfig:
    dataset: DatasetConfig = MISSING
    trainloader: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    testloader: TestLoaderConfig = field(default_factory=TestLoaderConfig)
    cache: bool = False
    in_channels: int = 1
    num_classes: int = 10
    sidelength: int = 28


@dataclass
class MnistDataConfig(DataConfig):
    dataset: DatasetConfig = field(default_factory=MnistDatasetConfig)
    in_channels: int = 1
    num_classes: int = 10
    sidelength: int = 28


@dataclass
class Cifar10DataConfig(DataConfig):
    dataset: DatasetConfig = field(default_factory=Cifar10DatasetConfig)
    in_channels: int = 3
    num_classes: int = 10
    sidelength: int = 32


@dataclass
class SchedulerConfig:
    pass


@dataclass
class StepLRConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 10
    gamma: float = 0.8


class CheckpointOccasion(Enum):
    BEST = "best"
    LATEST = "latest"


@dataclass
class ExecutionConfig:
    device: str = "cpu"
    dry_run: bool = True
    epochs: int = 15
    seed: int = 42
    start_from: Optional[CheckpointOccasion] = CheckpointOccasion.LATEST
    start_weights: Optional[str] = None


@dataclass
class OutputConfig:
    save_checkpoints: bool = True
    out_dir: Path = root / "_output"
    use_wandb: bool = False
    wandb_project: str = "scaffolding-v3"
    log_gradients: bool = False
    gradient_log_freq: int = 100
    use_tqdm: bool = False
    log_level: str = "INFO"
    plot: bool = True
    sample_indices: tuple[int, ...] = (0, 1, 2, 3)


defaults = [
    "_self_",
    {"data": "mnist"},
    {"mode": "dev"},
]


@dataclass
class Config:
    defaults: list = field(default_factory=lambda: defaults)
    hydra: dict = field(default_factory=dict)
    data: DataConfig = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=AdadeltaConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scheduler: Optional[SchedulerConfig] = field(default_factory=StepLRConfig)
    paths: Paths = field(default_factory=Paths)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    hydra: dict = field(default_factory=lambda: {"mode": "RUN"})


def load_config() -> ConfigStore:
    cs = ConfigStore.instance()

    cs.store(group="data", name="mnist", node=MnistDataConfig)
    cs.store(group="data", name="cifar10", node=Cifar10DataConfig)

    # Can't use structured configs here, might need more investigation?
    cs.store(
        group="mode",
        name="dev",
        package="_global_",
        node={},
    )

    cs.store(
        group="mode",
        name="prod",
        package="_global_",
        node={
            "execution": {
                "dry_run": False,
            },
            "output": {
                "use_wandb": True,
            },
        },
    )

    cs.store(name="train", node=Config())
    return cs
