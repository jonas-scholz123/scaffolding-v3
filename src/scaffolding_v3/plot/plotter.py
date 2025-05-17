# %%
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from loguru import logger
from matplotlib.figure import Figure
from mlbnb.paths import ExperimentPath
from torch.utils.data import Dataset


class Plotter:
    def __init__(
        self,
        device: str | torch.device,
        test_data: Dataset,
        save_to: Optional[ExperimentPath] = None,
        sample_indices: Sequence[int] = [0],
    ):
        self._device = device
        self._sample_tasks: list = [test_data[i] for i in sample_indices]
        self._num_samples = len(sample_indices)
        self._test_data = test_data
        self._dir = save_to

    def plot_prediction(self, model: nn.Module, epoch: int = 0) -> Optional[Figure]:
        if self._is_image_classification_task(self._sample_tasks[0]):
            fig = self._plot_mnist(model)
        else:
            logger.warning("Unplottable target type: %s", type(self._sample_tasks[0]))
            return None

        if fig:
            self._save_or_show(fig, f"{epoch}_prediction.png")

    def _is_image_classification_task(self, task: Any) -> bool:
        if (
            isinstance(task, tuple)
            and len(task) == 2
            and isinstance(task[0], torch.Tensor)
            and isinstance(task[1], int)
        ):
            return True
        return False

    def _plot_mnist(self, model: nn.Module) -> Optional[Figure]:
        fig, axs = plt.subplots(
            1, self._num_samples, figsize=(10, 5 * self._num_samples), squeeze=False
        )
        for i, task in enumerate(self._sample_tasks):
            img, target = task
            img = img.to(self._device)
            # Squeeze and/or unsqueeze to ensure image is C_W_H:
            if img.dim() == 2:
                img = img.unsqueeze(0).unsqueeze(0)
            elif img.dim() == 3:
                img = img
            pred = model(img.unsqueeze(0)).argmax().item()
            # Normalize image to [0, 1]:
            img = (img - img.min()) / (img.max() - img.min())
            img = img.permute(1, 2, 0)
            axs[0, i].imshow(img.squeeze().cpu().numpy(), cmap="grey")  # type: ignore
            axs[0, i].set_title(f"True: {target}, Pred: {pred}")  # type: ignore
            if target != pred:
                axs[0, i].title.set_color("red")  # type: ignore
        return fig

    def _save_or_show(self, fig: Figure, fname: str) -> None:
        if self._dir:
            fig.savefig(self._dir.at(fname), bbox_inches="tight", dpi=300)
        else:
            plt.show()
