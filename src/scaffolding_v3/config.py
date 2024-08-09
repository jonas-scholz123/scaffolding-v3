from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

root = Path(__file__).resolve().parent.parent.parent
ppu = 200


@dataclass
class Paths:
    root: Path = root
    data: Path = root / "_data"
    raw_data: Path = root / "_data" / "_raw"
    elevation: Path = data / "elevation" / "elevation.nc"
    dwd: Path = data / "dwd" / "dwd.feather"
    dwd_meta: Path = data / "dwd" / "dwd_meta.feather"
    value_stations: Path = data / "dwd" / "value_stations.feather"
    station_splits: Path = data / "dwd" / "station_splits.feather"
    time_splits: Path = data / "dwd" / "time_splits.feather"
    data_processor_dir: Path = data / "dwd"
    output: Path = root / "_output"


@dataclass
class ModelConfig:
    _target_: str = "deepsensor.model.ConvNP"
    internal_density: int = ppu


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
    batch_size: int = 4
    shuffle: bool = True


@dataclass
class TestLoaderConfig(DataloaderConfig):
    batch_size: int = 1000
    shuffle: bool = False


@dataclass
class DataProviderConfig:
    pass


@dataclass
class DwdDataProviderConfig(DataProviderConfig):
    _target_: str = "data.dwd.DwdDataProvider"
    num_stations: int = 500
    num_times: int = 10000
    val_fraction: float = 0.1
    aux_ppu: int = ppu
    paths: Paths = field(default_factory=Paths)


@dataclass
class DwdConfig:
    dwd_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/recent/"
    value_url = "https://www.value-cost.eu/sites/default/files/VALUE_ECA_53_Germany_spatial_v1.zip"
    crs_str = "EPSG:4326"


@dataclass
class SrtmConfig:
    srtm_url = "https://www.opendem.info/downloads/srtm_germany_dtm.zip"


@dataclass
class DataConfig:
    trainloader: TrainLoaderConfig = field(default_factory=TrainLoaderConfig)
    testloader: TestLoaderConfig = field(default_factory=TestLoaderConfig)
    data_provider: DataProviderConfig = field(default_factory=DwdDataProviderConfig)


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
    device: str = "cuda"
    dry_run: bool = True
    epochs: int = 10
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


# This isn't perfect, the type annotation approach is nicer but doesn't work with omegaconf
SKIP_KEYS = {
    "output",
    "start_from",
    "download",
    "_partial_",
    "root",
    "testloader",
    "paths",
}
