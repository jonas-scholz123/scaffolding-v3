from mlbnb.types import Split
import torch
from torch.utils.data import DataLoader
from loguru import logger
import hydra
import deepsensor.torch # noqa

from deepsensor.train.train import set_gpu_default_device

from scaffolding_v3.config import Config, DwdDataConfig, Era5DataConfig
from scaffolding_v3.data.dataprovider import DataProvider
from scaffolding_v3.data.dataset import make_dataset
from scaffolding_v3.data.dwd import get_data_processor


cfg = Config(
    data=Era5DataConfig()
)
set_gpu_default_device()
torch.manual_seed(0)


generator = torch.Generator(device="cpu").manual_seed(0)
logger.info("Instantiating dependencies")

data_processor = get_data_processor(cfg.paths)

# Create the primary data source (ERA5/DWD)
data_provider: DataProvider = hydra.utils.instantiate(cfg.data.data_provider)

trainset = make_dataset(
    cfg.data, cfg.paths, data_provider, Split.TRAIN, data_processor
)

train_loader = DataLoader(trainset, batch_size=1, shuffle=False, generator=generator, collate_fn=lambda x: x)


for batch in train_loader:
    print(batch)

    for task in batch:
        print(type(task["X_c"][0]))
    break
