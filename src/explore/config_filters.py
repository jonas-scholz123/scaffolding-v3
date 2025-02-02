# %%
from hydra import compose

from scaffolding_v3.config import Config, load_config
from scaffolding_v3.util.explore import load_best_weights
from scaffolding_v3.util.instantiate import Dependencies

cs = load_config()

query_cfg: Config = compose(
    config_name="train", overrides=["mode=prod", "data.testloader.batch_size=1"]
)  # type: ignore

d = Dependencies.from_config(query_cfg)
load_best_weights(d.model, query_cfg)
