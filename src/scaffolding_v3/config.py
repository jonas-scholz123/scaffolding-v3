from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

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


@dataclass
class ModelConfig:
    _target_: str = "deepsensor.model.ConvNP"
    internal_density: int = ppu
    unet_channels: tuple = (64,) * 4
    aux_t_mlp_layers: tuple = (64,) * 3
    likelihood: str = "cnp"


@dataclass
class OptimizerConfig:
    lr: float = 1e-3


@dataclass
class AdadeltaConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adadelta"


@dataclass
class AdamConfig(OptimizerConfig):
    _target_: str = "torch.optim.Adam"
    lr: float = 5e-5


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


@dataclass
class TestLoaderConfig(DataloaderConfig):
    batch_size: int = 1000
    shuffle: bool = False


@dataclass
class DataProviderConfig:
    num_stations: int = 500
    num_times: int = 10000
    val_fraction: float = 0.1
    aux_ppu: int = ppu
    hires_aux_ppu: int = 2000
    cache: bool = False
    daily_averaged: bool = False
    train_range: tuple[str, str] = ("2006-01-01", "2023-01-01")
    test_range: tuple[str, str] = ("2023-01-01", "2024-01-01")
    include_context_in_target: bool = False
    include_aux_at_target: bool = False
    paths: Paths = field(default_factory=Paths)


@dataclass
class DwdDataProviderConfig(DataProviderConfig):
    _target_: str = "data.dwd.DwdDataProvider"


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
class DataConfig:
    trainloader: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    testloader: TestLoaderConfig = field(default_factory=TestLoaderConfig)
    data_provider: DataProviderConfig = field(default_factory=DwdDataProviderConfig)
    task_loader: TaskLoaderConfig = field(default_factory=TaskLoaderConfig)
    include_aux_at_targets: bool = True
    include_context_in_target: bool = False
    ppu: int = ppu
    hires_ppu: int = 2000
    cache: bool = False


@dataclass
class SchedulerConfig:
    pass


@dataclass
class StepLRConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.StepLR"
    step_size: int = 10
    gamma: float = 0.5


class CheckpointOccasion(Enum):
    BEST = "best"
    LATEST = "latest"


@dataclass
class ExecutionConfig:
    device: str = "cuda"
    dry_run: bool = True
    epochs: int = 80
    seed: int = 42
    start_from: Optional[CheckpointOccasion] = CheckpointOccasion.LATEST


@dataclass
class OutputConfig:
    save_model: bool = True
    out_dir: Path = root / "_output"
    use_wandb: bool = False
    log_level: str = "INFO"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scheduler: Optional[SchedulerConfig] = field(default_factory=StepLRConfig)
    paths: Paths = field(default_factory=Paths)


def load_config() -> None:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(
        name="dev",
        node=Config(
            data=DataConfig(
                data_provider=DwdDataProviderConfig(
                    train_range=("2006-01-01", "2007-01-01")
                )
            ),
        ),
    )
    cs.store(
        name="prod",
        node=Config(
            execution=ExecutionConfig(dry_run=False),
            output=OutputConfig(use_wandb=True),
        ),
    )
