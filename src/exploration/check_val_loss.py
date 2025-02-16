# %%
import hydra
from mlbnb.types import Split
from omegaconf import DictConfig

from scaffolding_v3.config import (
    Config,
    DataConfig,
    DwdDataProviderConfig,
    ExecutionConfig,
    OutputConfig,
    Paths,
)
from scaffolding_v3.data.dataset import make_dataset
from scaffolding_v3.data.dwd import get_data_processor

# %%
cfg = DictConfig(
    Config(
        execution=ExecutionConfig(dry_run=False),
        output=OutputConfig(use_wandb=True),
        data=DataConfig(data_provider=DwdDataProviderConfig(daily_averaged=False)),
    )
)
paths = Paths()

data_processor = get_data_processor(paths, cfg.data)
data_provider = hydra.utils.instantiate(cfg.data.data_provider)
valset = make_dataset(cfg.data, cfg.paths, data_provider, Split.VAL, data_processor)

task_loader = valset.task_loader
model = hydra.utils.instantiate(cfg.model, data_processor, task_loader)
# %%
from deepsensor.plot import context_encoding

task = valset[0]

_ = context_encoding(model, task, task_loader)

# %%
task
