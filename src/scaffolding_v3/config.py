from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

root = Path(__file__).resolve().parent.parent.parent


@dataclass
class LossConfig:
    _target_: str = "torch.nn.functional.nll_loss"
    _partial_: bool = True


@dataclass
class ModelConfig:
    _target_: str = "models.convnet.ConvNet"


@dataclass
class OptimizerConfig:
    lr: float = 1.0


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


@dataclass
class TestLoaderConfig(DataloaderConfig):
    batch_size: int = 1000
    shuffle: bool = False


@dataclass
class DatasetConfig:
    _target_: str = "torchvision.datasets.MNIST"
    root: Path = root / "_data"
    train: bool = True
    download: bool = True


@dataclass
class TrainsetConfig(DatasetConfig):
    train: bool = True


@dataclass
class TestsetConfig(DatasetConfig):
    train: bool = False
    download: bool = False


@dataclass
class NormalizationConfig:
    _target_: str = "torchvision.transforms.Normalize"
    mean: tuple = (0.1307,)
    std: tuple = (0.3081,)


@dataclass
class DataConfig:
    trainloader: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    testloader: TestLoaderConfig = field(default_factory=TestLoaderConfig)
    trainset: DatasetConfig = field(default_factory=TrainsetConfig)
    testset: TestsetConfig = field(default_factory=TestsetConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class SchedulerConfig:
    pass


@dataclass
class StepLRConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 1
    gamma: float = 0.7


class CheckpointOccasion(Enum):
    BEST = "best"
    LATEST = "latest"


@dataclass
class ExecutionConfig:
    device: str = "mps"
    dry_run: bool = False
    epochs: int = 10
    seed: int = 1
    start_from: Optional[CheckpointOccasion] = CheckpointOccasion.LATEST


@dataclass
class OutputConfig:
    save_model: bool = True
    out_dir: Path = root / "_output"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=AdadeltaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scheduler: Optional[SchedulerConfig] = field(default_factory=StepLRConfig)


# This isn't perfect, the type annotation approach is nicer but doesn't work with omegaconf
SKIP_KEYS = {"data", "output", "dry_run", "start_from", "download", "_partial_"}
