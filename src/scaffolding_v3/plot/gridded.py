import cartopy.feature as feature
import deepsensor.torch  # noqa
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from deepsensor import Task
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from matplotlib.figure import Figure

from scaffolding_v3.plot.util import make_uniform_grid


def plot_gridded(
    task: Task,
    truth: xr.DataArray,
    data_processor: DataProcessor,
    model: ConvNP,
    bounds: tuple[float, float, float, float],
    resolution: float = 0.01,
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
