# %%
from typing import Optional

import cartopy.feature as feature
import deepsensor.torch  # noqa
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from cartopy import crs as ccrs
from deepsensor import Task
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from hydra.utils import instantiate
from loguru import logger
from matplotlib.figure import Figure
from mlbnb.paths import ExperimentPath

from scaffolding_v3.config import Config, Era5DataConfig
from scaffolding_v3.data.dataprovider import DeepSensorDataset
from scaffolding_v3.data.dataset import make_taskloader
from scaffolding_v3.data.dwd import get_data_processor
from scaffolding_v3.data.era5 import Era5DataProvider, load_era5


def make_uniform_grid(
    min_lon: float, max_lon: float, min_lat: float, max_lat: float, resolution: float
) -> xr.DataArray:
    lons = np.arange(min_lon, max_lon + resolution, resolution)
    lats = np.arange(min_lat, max_lat + resolution, resolution)

    data = np.zeros((len(lats), len(lons)))
    return xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
    )


def plot_prediction_and_errors(
    task: Task,
    truth: xr.DataArray,
    data_processor: DataProcessor,
    model: ConvNP,
    bounds: tuple[float, float, float, float],
    resolution: float = 0.1,
) -> Figure:
    """
    Plot truth, predicted mean, std, errors in a row.
    """

    prediction = model.predict(task, X_t=truth)["t2m"]
    mean_ds, _ = prediction["mean"], prediction["std"]

    X_t_dense = make_uniform_grid(*bounds, resolution)

    prediction = model.predict(task, X_t=X_t_dense)["t2m"]
    mean_ds_dense, std_ds_dense = prediction["mean"], prediction["std"]

    err_da = mean_ds - truth

    sel = dict(time=task["time"])
    era5_data = truth.sel(sel)
    mean_data = mean_ds_dense.sel(sel)
    std_data = std_ds_dense.sel(sel)
    error_data = err_da.sel(sel)

    fig = plot_era5_prediction_and_errors(
        era5_data,
        mean_data,
        std_data,
        error_data,
        data_processor,
        task,
        bounds,
    )

    plt.close()
    plt.clf()
    return fig


def plot_era5_prediction_and_errors(
    era5_data: xr.DataArray,
    mean_data: xr.DataArray,
    std_data: xr.DataArray,
    error_data: xr.DataArray,
    data_processor: DataProcessor,
    task: Task,
    bounds: tuple[float, float, float, float],
) -> Figure:
    proj = ccrs.TransverseMercator(central_longitude=10, approx=False)
    subplots = plt.subplots(
        subplot_kw={"projection": proj}, nrows=1, ncols=4, figsize=(10, 2.5)
    )
    fig = subplots[0]
    axs: np.ndarray = subplots[1]  # type: ignore

    era5_plot = era5_data.plot(cmap="seismic", ax=axs[0], transform=ccrs.PlateCarree())  # type: ignore
    cbar = era5_plot.colorbar
    vmin, vmax = cbar.vmin, cbar.vmax

    axs[0].set_title("ERA5")

    mean_data.plot(
        cmap="seismic",
        ax=axs[1],
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )  # type: ignore
    axs[1].set_title("ConvNP mean")
    std_data.plot(cmap="Greys", ax=axs[2], transform=ccrs.PlateCarree())  # type: ignore
    axs[2].set_title("ConvNP std dev")
    error_data.plot(cmap="seismic", ax=axs[3], transform=ccrs.PlateCarree())  # type: ignore
    axs[3].set_title("ConvNP error")

    context_axs = [ax for i, ax in enumerate(axs) if i != 1]
    deepsensor.plot.offgrid_context(
        context_axs,
        task,
        data_processor,
        add_legend=False,
        transform=ccrs.PlateCarree(),
        plot_target=False,
        context_set_idxs=0,
        s=3**2,
        linewidth=0.5,
    )

    for ax in axs:
        ax.add_feature(feature.BORDERS, linewidth=0.25)  # type: ignore
        ax.coastlines(linewidth=0.25)  # type: ignore

    return fig


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
                "Timestamp {} not in test data. Using first timestamp instead.",
                self.time,
            )
            self.time = test_data.times[0]
        self.sample_task: Task = self.task_loader(
            self.time, context_sampling=[0.05, "all"], target_sampling="all"
        )  # type: ignore
        self.test_data = test_data
        self.dir = dir

    def plot_prediction(self, model: ConvNP, epoch: int = 0) -> Optional[Figure]:
        target = self.test_data.target["t2m"]
        if isinstance(target, xr.DataArray):
            fig = plot_prediction_and_errors(
                self.sample_task,
                target,
                self.data_processor,
                model,
                self.bounds,
            )
            self._save_or_show(fig, f"{epoch}_{self.time_str}_prediction.pdf")
            return fig
        else:
            logger.warning("DWD plotting not yet supported.")

    def _save_or_show(self, fig: Figure, fname: str) -> None:
        if self.dir:
            fig.savefig(self.dir.at(fname), bbox_inches="tight")
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

    data_processor = get_data_processor(cfg.paths)

    era5_data = load_era5(cfg.paths)["t2m"]

    data_provider: Era5DataProvider = instantiate(cfg.data.data_provider)
    dataset = data_provider.get_test_data()
    task_loader = make_taskloader(cfg.data, cfg.paths, data_processor, dataset)

    generator = torch.Generator(device=cfg.execution.device).manual_seed(
        cfg.execution.seed
    )

    model = hydra.utils.instantiate(cfg.model, data_processor, task_loader)

    time = dataset.times[0]
    task: Task = task_loader(time, context_sampling=0.05)  # type: ignore

    bounds = (cfg.geo.min_lon, cfg.geo.max_lon, cfg.geo.min_lat, cfg.geo.max_lat)
    plot_prediction_and_errors(task, era5_data, data_processor, model, bounds)

    plotter = Plotter(cfg, data_processor, dataset)
    plotter.plot_task()
    plotter.plot_context_encoding(model)
