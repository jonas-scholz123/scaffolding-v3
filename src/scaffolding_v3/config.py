from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

root = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelConfig:
    pass


@dataclass
class ConvNetConfig(ModelConfig):
    _target_: str = "models.convnet.ConvNet"


@dataclass
class OptimizerConfig:
    lr: float = 1.0


@dataclass
class AdadeltaConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adadelta"


@dataclass
class DataloaderConfig:
    _target_: str = "torch.utils.data.DataLoader"
    batch_size: int = 64
    shuffle: bool = True


@dataclass
class TrainLoaderConfig(DataloaderConfig):
    pass


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
class DataConfig:
    trainloader: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    testloader: TestLoaderConfig = field(default_factory=TestLoaderConfig)
    trainset: DatasetConfig = field(default_factory=TrainsetConfig)
    testset: TestsetConfig = field(default_factory=TestsetConfig)


@dataclass
class SchedulerConfig:
    pass


@dataclass
class StepLRConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 1
    gamma: float = 0.7


@dataclass
class ExecutionConfig:
    device: str = "mps"
    dry_run: bool = False
    epochs: int = 10
    seed: int = 1


@dataclass
class OutputConfig:
    save_model: bool = False
    out_dir: Path = root / "_output"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ConvNetConfig)
    optimizer: OptimizerConfig = field(default_factory=AdadeltaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scheduler: Optional[SchedulerConfig] = field(default_factory=StepLRConfig)
