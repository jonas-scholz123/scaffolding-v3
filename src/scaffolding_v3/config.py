from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

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
    device: str = "cuda"
    dry_run: bool = True
    epochs: int = 100
    seed: int = 42
    start_from: Optional[CheckpointOccasion] = CheckpointOccasion.LATEST
    start_weights: Optional[str] = None


@dataclass
class OutputConfig:
    save_checkpoints: bool = True
    out_dir: Path = root / "_output"
    use_wandb: bool = False
    use_tqdm: bool = False
    log_level: str = "INFO"
    plot: bool = True
    sample_indices: tuple[int, ...] = (0, 1, 2, 3)


defaults = [
    "_self_",
    {"data": "mnist"},
    {"mode": "dev"},
    {"runner": "default"},
    {"override hydra/sweeper": "optuna"},
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


def load_config() -> None:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    cs.store(group="data", name="mnist", node=MnistDataConfig)
    cs.store(group="data", name="cifar10", node=Cifar10DataConfig)

    cs.store(
        group="runner",
        name="default",
        package="_global_",
        node={
            "hydra": {
                "mode": "RUN",  # Workaround for https://github.com/facebookresearch/hydra/issues/2262
                "sweeper": {"n_jobs": 1},
            },
            "output": {"use_tqdm": True},
            "data": {
                "trainloader": {"num_workers": 0, "multiprocessing_context": None}
            },
        },
    )

    cs.store(
        group="runner",
        name="parallel",
        package="_global_",
        node={
            "defaults": [
                {"override /hydra/launcher": "submitit_local"},
            ],
            "hydra": {
                "launcher": {
                    "timeout_min": 600,
                    "mem_gb": 8,
                    "gpus_per_node": 1,
                },
                "sweeper": {
                    # Close to ideal on my local PC with 4090.
                    "n_jobs": 5,
                },
                "mode": "MULTIRUN",
            },
            # Workers spawned by the dataloader don't get along with submitit/joblib
            "data": {
                "trainloader": {"num_workers": 0, "multiprocessing_context": None}
            },
            # Submitit logs to files, tqdm spams those files.
            "output": {"use_tqdm": False},
        },
    )

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

    hyperparameter_opt_params = {
        # Limit dataset size/training length for faster sweeps
        "execution.epochs": 10,
        # Find good learning rate
        "optimizer.lr": "tag(log, interval(5e-5, 5e-2))",
    }
    cs.store(
        name="hyperparam_opt",
        node=Config(
            defaults=defaults
            + [
                {"override /hydra/sweeper": "optuna"},
                {"override /hydra/sweeper/sampler": "tpe"},
            ],
            hydra={
                "sweeper": {
                    "params": hyperparameter_opt_params,
                },
            },
        ),
    )

    sweep_params = {
        "execution.seed": "42,43,44",
    }
    cs.store(
        name="sweep",
        node=Config(
            defaults=defaults
            + [
                {"override /hydra/sweeper/sampler": "grid"},
            ],
            hydra={
                "sweeper": {
                    "params": sweep_params,
                }
            },
        ),
    )

    cs.store(name="train", node=Config())
