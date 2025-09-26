# %%
from mlbnb.examples import find_best_examples

from scaffolding_v3.model.classification import ClassificationModule
from scaffolding_v3.util.instantiate import Experiment

path = "../../_output/2025-06-25_13-59_gentle_aardvark"
exp = Experiment.from_path(path, checkpoint="best")


def compute_task_loss(task: tuple) -> float:
    device = exp.cfg.runtime.device
    X, y = task
    X = X.to(device)
    y = y.to(device)
    return exp.model(X, y).item()


dataloader = exp.val_loader
# Avoid multiprocessing issues in Jupyter notebooks
dataloader.num_workers = 0

tasks, losses = find_best_examples(dataloader, compute_task_loss, 4, mode="hardest")

exp.plotter._sample_tasks = [(t[0][0], int(t[1][0])) for t in tasks]
exp.plotter._num_samples = 4

assert isinstance(exp.model, ClassificationModule)
exp.plotter.plot_prediction(exp.model)
