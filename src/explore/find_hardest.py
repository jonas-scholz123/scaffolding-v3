# %%
from mlbnb.examples import find_best_examples

from scaffolding_v3.util.explore import load_best_weights, setup
from scaffolding_v3.util.instantiate import Experiment

cfg = setup(overrides=["data.testloader.batch_size=1"])

d = Experiment.from_config(cfg)

load_best_weights(d.model, cfg)


def compute_task_loss(task: tuple) -> float:
    X, y = task
    X = X.to(cfg.runtime.device)
    y = y.to(cfg.runtime.device)
    y_hat = d.model(X)
    return d.loss_fn(y_hat, y)


tasks, losses = find_best_examples(d.val_loader, compute_task_loss, 4, mode="easiest")
if d.plotter is None:
    raise ValueError("Plotter is not defined in the experiment configuration.")
d.plotter._sample_tasks = [(t[0][0], int(t[1][0])) for t in tasks]
d.plotter._num_samples = 4
d.plotter.plot_prediction(d.model)
