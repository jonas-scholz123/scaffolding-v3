# %%
from hydra import compose

from config.config import Config, load_config
from scaffolding_v3.util.explore import load_best_weights
from scaffolding_v3.util.instantiate import Experiment

cs = load_config()

query_cfg: Config = compose(  # type: ignore
    config_name="train", overrides=["mode=prod", "data.testloader.batch_size=1"]
)

d = Experiment.from_config(query_cfg)
load_best_weights(d.model, query_cfg)
