# %%

from scaffolding_v3.util.explore import load_best_weights
from scaffolding_v3.util.instantiate import Experiment, load_config

query_cfg = load_config(mode="prod", overrides=["data.testloader.batch_size=1"])

d = Experiment.from_config(query_cfg)
load_best_weights(d.model, query_cfg)

d.model
