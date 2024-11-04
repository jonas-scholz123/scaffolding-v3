# %%
import deepsensor.torch  # noqa
from hydra.utils import instantiate

from scaffolding_v3.config import Config, Era5DataConfig
from scaffolding_v3.data.dwd import get_data_processor

cfg = Config(data=Era5DataConfig())

data_provider = instantiate(cfg.data.data_provider)
data_processor = get_data_processor(cfg.paths)
# %%
td = data_provider.get_train_data()
# 9.6
print(td.context["t2m"].mean())
