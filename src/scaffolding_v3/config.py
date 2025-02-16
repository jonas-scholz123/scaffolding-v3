from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from omegaconf.omegaconf import MISSING

# This isn't perfect, the type annotation approach is nicer but doesn't work with omegaconf
SKIP_KEYS = {
    "output",
    "start_from",
    "download",
    "_partial_",
    "root",
    "testloader",
    "paths",
    "epochs",
    "geo",
}

root = Path(__file__).resolve().parent.parent.parent
ppu = 150


@dataclass
class Paths:
    root: Path = root
    data: Path = root / "_data"
    raw_data: Path = data / "_raw"
    elevation: Path = data / "elevation" / "elevation.nc"
    dwd: Path = data / "dwd" / "dwd.parquet"
    dwd_meta: Path = data / "dwd" / "dwd_meta.parquet"
    era5: Path = data / "era5" / "era5_t2m.nc"
    value_stations: Path = data / "dwd" / "value_stations.parquet"
    data_processor_dir: Path = data / "dwd"
    output: Path = root / "_output"
    weights: Path = root / "_weights"


@dataclass
class ModelConfig:
    _target_: str = "deepsensor.model.ConvNP"
    internal_density: int = ppu
    unet_channels: tuple = (64,) * 4  # type: ignore
    # aux_t_mlp_layers: tuple = (64,) * 3  # type: ignore
    likelihood: str = "cnp"
    # Approximately from deepsensor (inferred on DWD data)
    encoder_scales: float = 0.003333
    decoder_scale: float = 0.006666
    verbose: bool = False


@dataclass
class OptimizerConfig:
    lr: float = 5e-5


@dataclass
class AdadeltaConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adadelta"


@dataclass
class AdamConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"


@dataclass
class DataloaderConfig:
    batch_size: int
    shuffle: bool
    _target_: str = "torch.utils.data.DataLoader"


@dataclass
class TrainLoaderConfig(DataloaderConfig):
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 0
    multiprocessing_context: Optional[str] = None


@dataclass
class TestLoaderConfig(DataloaderConfig):
    batch_size: int = 1000
    shuffle: bool = False


@dataclass
class DataProviderConfig:
    paths: Paths = field(default_factory=Paths)
    val_fraction: float = 0.1
    train_range: tuple[str, str] = ("2006-01-01", "2023-01-01")
    test_range: tuple[str, str] = ("2023-01-01", "2024-01-01")
    num_times: int = 10000


@dataclass
class DwdDataProviderConfig(DataProviderConfig):
    _target_: str = "scaffolding_v3.data.dwd.DwdDataProvider"
    num_stations: int = 500
    daily_averaged: bool = False


@dataclass
class Era5DataProviderConfig(DataProviderConfig):
    _target_: str = "scaffolding_v3.data.era5.Era5DataProvider"
    train_range: tuple[str, str] = ("2006-01-01", "2011-01-01")
    test_range: tuple[str, str] = ("2011-01-01", "2012-01-01")
    num_times: int = 10000


@dataclass
class DwdConfig:
    dwd_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/historical/"
    value_url = "https://www.value-cost.eu/sites/default/files/VALUE_ECA_53_Germany_spatial_v1.zip"
    crs_str = "EPSG:4326"


@dataclass
class SrtmConfig:
    srtm_url = "https://www.opendem.info/downloads/srtm_germany_dtm.zip"


@dataclass
class Era5Config:
    era5_url = "https://cds.climate.copernicus.eu/api/v2"


# Germany bounding box
@dataclass
class GeoConfig:
    min_lat: float = 47.2
    max_lat: float = 54.95
    min_lon: float = 5.8
    max_lon: float = 15.05
    crs_str: str = "EPSG:4326"


@dataclass
class TaskLoaderConfig:
    _target_: str = "deepsensor.data.loader.TaskLoader"
    _partial_: bool = True
    discrete_xarray_sampling: bool = True


@dataclass
class Era5TaskLoaderConfig(TaskLoaderConfig):
    discrete_xarray_sampling: bool = False


@dataclass
class DataConfig:
    data_provider: DataProviderConfig = MISSING
    task_loader: TaskLoaderConfig = field(default_factory=TaskLoaderConfig)
    trainloader: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    testloader: TestLoaderConfig = field(default_factory=TestLoaderConfig)
    include_aux_at_targets: bool = True
    include_context_in_target: bool = False
    include_tpi: bool = True
    ppu: int = ppu
    hires_ppu: int = 2000
    cache: bool = False


@dataclass
class Era5DataConfig(DataConfig):
    data_provider: DataProviderConfig = field(default_factory=Era5DataProviderConfig)
    task_loader: TaskLoaderConfig = field(default_factory=Era5TaskLoaderConfig)
    include_context_in_target: bool = True


@dataclass
class DwdDataConfig(DataConfig):
    data_provider: DataProviderConfig = field(default_factory=DwdDataProviderConfig)


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
    epochs: int = 300
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
    plot_time: str = "2023-06-01 00:00:00"


defaults = [
    "_self_",
    {"data": "real"},
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
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scheduler: Optional[SchedulerConfig] = field(default_factory=StepLRConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    paths: Paths = field(default_factory=Paths)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


def load_config() -> None:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    cs.store(group="data", name="sim", node=Era5DataConfig)
    cs.store(group="data", name="real", node=DwdDataConfig)

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
                "trainloader": {"num_workers": 8, "multiprocessing_context": "spawn"}
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
        node={
            "data": {
                "data_provider": {
                    "train_range": ["2016-01-01", "2016-02-01"],
                }
            }
        },
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

    cs.store(
        name="pretrain",
        node=Config(
            defaults=defaults + [{"override /data": "sim"}],
        ),
    )

    hyperparameter_opt_params = {
        # Limit dataset size/training length for faster sweepsa
        "execution.epochs": 10,
        "data.data_provider.num_times": 10000,
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

    finetune_params = {
        "data.data_provider.num_stations": "20,100,500",
        # "data.data_provider.num_times": "400,2000,10000",
        "execution.seed": "42,43",
    }
    cs.store(
        name="finetune",
        node=Config(
            defaults=defaults
            + [
                {"override /hydra/sweeper/sampler": "grid"},
                {"override /data": "real"},
            ],
            hydra={
                "sweeper": {
                    "params": finetune_params,
                }
            },
            execution=ExecutionConfig(
                epochs=40, start_weights="best_era5_continuous.pt"
            ),
            # Slightly smaller learning rate for fine-tuning
            optimizer=AdamConfig(lr=5e-5),
            # Batch size 1 for fine-tuning
            data=DwdDataConfig(trainloader=TrainLoaderConfig(batch_size=1)),
        ),
    )
