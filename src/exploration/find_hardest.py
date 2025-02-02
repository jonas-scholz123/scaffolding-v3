# %%
import torch
import torch.optim.lr_scheduler
from hydra import compose
from hydra.initialize import initialize
from loguru import logger
from mlbnb.examples import find_best_examples
from omegaconf import OmegaConf

from scaffolding_v3.config import (
    Config,
    load_config,
)
from scaffolding_v3.instantiate import Dependencies


# %%
cs = load_config()
initialize(config_path=None)
# %%
cfg: Config = compose(
    config_name="train", overrides=["mode=prod", "data.testloader.batch_size=1"]
)  # type: ignore

print(OmegaConf.to_yaml(cfg.data.testloader))
# %%
torch.set_default_device(cfg.execution.device)

d = Dependencies.from_config(cfg)

logger.info("Instantiating dependencies")
_ = d.checkpoint_manager.reproduce_model(d.model, "best")
# %%

def compute_task_loss(task: tuple):
    X, y = task
    X = X.to(cfg.execution.device)
    y = y.to(cfg.execution.device)
    y_hat = d.model(X)
    return d.loss_fn(y_hat, y)


tasks, losses = find_best_examples(d.val_loader, compute_task_loss, 4, mode="easiest")

d.plotter._sample_tasks = [(t[0][0], int(t[1][0])) for t in tasks]
d.plotter._num_samples = 4
d.plotter.plot_prediction(d.model)
