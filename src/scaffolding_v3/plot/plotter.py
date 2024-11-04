# %%
from typing import Optional

import deepsensor.torch  # noqa
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
import xarray as xr
from deepsensor import Task
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from hydra.utils import instantiate
from loguru import logger
from matplotlib.figure import Figure
from mlbnb.paths import ExperimentPath

from scaffolding_v3.config import Config, DwdDataConfig, Era5DataConfig
from scaffolding_v3.data.dataprovider import DeepSensorDataset
from scaffolding_v3.data.dataset import make_taskloader
from scaffolding_v3.data.dwd import DwdDataProvider, get_data_processor
from scaffolding_v3.data.era5 import Era5DataProvider, load_era5
from scaffolding_v3.plot.gridded import plot_gridded
from scaffolding_v3.plot.offgrid import plot_offgrid


class Plotter:
    def __init__(
        self,
        cfg: Config,
        data_processor: DataProcessor,
        test_data: DeepSensorDataset,
        dir: Optional[ExperimentPath] = None,
    ):
        self.cfg = cfg
        geo = cfg.geo
        self.bounds = (geo.min_lon, geo.max_lon, geo.min_lat, geo.max_lat)
        self.data_processor = data_processor
        self.task_loader = make_taskloader(
            cfg.data, cfg.paths, data_processor, test_data
        )
        self.time_str = cfg.output.plot_time
        self.time: pd.Timestamp = pd.Timestamp(self.time_str)  # type: ignore
        if self.time not in test_data.times:
            logger.warning(
                "Timestamp {} not in test data. Using last timestamp {} instead.",
                self.time,
                test_data.times[-1],
            )
            self.time = test_data.times[-1]
        self.sample_task: Task = self.task_loader(
            self.time,
            context_sampling=[0.05, "all"],
            target_sampling="all",
            seed_override=cfg.execution.seed,
        )  # type: ignore
        self.test_data = test_data
        self.dir = dir

    def plot_prediction(self, model: ConvNP, epoch: int = 0) -> Optional[Figure]:
        target = self.test_data.target["t2m"]
        if isinstance(target, xr.DataArray):
            fig = plot_gridded(
                self.sample_task,
                target,
                self.data_processor,
                model,
                self.bounds,
            )
            return fig
        elif isinstance(target, pd.Series):
            context: pd.Series = self.test_data.context["t2m"]  # type: ignore
            truth = pd.concat([context, target])
            fig = plot_offgrid(
                self.sample_task,
                truth,
                self.data_processor,
                model,
                self.bounds,
            )
        else:
            logger.warning("Unplottable target type: %s", type(target))
            return None

        self._save_or_show(fig, f"{epoch}_{self.time_str}_prediction.png")

    def _save_or_show(self, fig: Figure, fname: str) -> None:
        if self.dir:
            fig.savefig(self.dir.at(fname), bbox_inches="tight", dpi=300)
        else:
            plt.show()

    def plot_task(
        self, task: Optional[Task] = None, epoch: int = 0
    ) -> Optional[Figure]:
        if not task:
            task = self.sample_task

        deepsensor.plot.task(self.sample_task, self.task_loader)
        fig = plt.gcf()
        self._save_or_show(fig, f"{epoch}_{self.time_str}_task.pdf")
        return fig

    def plot_context_encoding(
        self, model: ConvNP, task: Optional[Task] = None, epoch: int = 0
    ) -> Optional[Figure]:
        if not task:
            task = self.sample_task

        fig: Figure = deepsensor.plot.context_encoding(
            model,
            self.sample_task,
            self.task_loader,
        )  # type: ignore
        self._save_or_show(fig, f"{epoch}_{self.time_str}_context_encoding.pdf")
        return fig


if __name__ == "__main__":
    cfg = Config(data=Era5DataConfig(include_context_in_target=True))
    cfg = Config(data=DwdDataConfig())

    data_processor = get_data_processor(cfg.paths)

    era5_data = load_era5(cfg.paths)["t2m"]

    data_provider: Era5DataProvider | DwdDataProvider = instantiate(
        cfg.data.data_provider
    )
    dataset = data_provider.get_test_data()
    task_loader = make_taskloader(cfg.data, cfg.paths, data_processor, dataset)

    generator = torch.Generator(device=cfg.execution.device).manual_seed(
        cfg.execution.seed
    )

    model = hydra.utils.instantiate(cfg.model, data_processor, task_loader)

    time = dataset.times[0]
    task: Task = task_loader(time, context_sampling=0.05)  # type: ignore

    bounds = (cfg.geo.min_lon, cfg.geo.max_lon, cfg.geo.min_lat, cfg.geo.max_lat)

    plotter = Plotter(cfg, data_processor, dataset)
    plotter.plot_task()
    plotter.plot_context_encoding(model)
    plotter.plot_prediction(model)
