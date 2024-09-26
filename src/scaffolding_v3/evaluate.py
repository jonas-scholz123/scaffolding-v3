# %%
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cf
import deepsensor.plot
import deepsensor.torch
import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepsensor.train.train import set_gpu_default_device
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath, config_to_filepath
from omegaconf import DictConfig

from scaffolding_v3.config import (
    SKIP_KEYS,
    Config,
    DataConfig,
    DwdDataProviderConfig,
    ExecutionConfig,
    OutputConfig,
    Paths,
)
from scaffolding_v3.data.dwd import get_data_processor
from scaffolding_v3.data.elevation import load_elevation_data

# %%


fontsize = 14
crs = ccrs.PlateCarree()

params = {
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "font.size": fontsize,
    "figure.titlesize": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.family": "sans-serif",
    "figure.facecolor": "w",
}


mpl.rcParams.update(params)


def gen_test_fig(
    era5_raw_ds=None,
    mean_ds=None,
    std_ds=None,
    samples_ds=None,
    task=None,
    extent: Any = None,
    add_colorbar=False,
    var_cmap="jet",
    var_clim=None,
    std_cmap="Greys",
    std_clim=None,
    var_cbar_label=None,
    std_cbar_label=None,
    fontsize=None,
    figsize=(15, 5),
):
    if var_clim is None and era5_raw_ds is not None and mean_ds is not None:
        vmin = np.array(min(era5_raw_ds.min(), mean_ds.min()))
        vmax = np.array(max(era5_raw_ds.max(), mean_ds.max()))
    elif var_clim is not None:
        vmin, vmax = var_clim
    else:
        vmin = None
        vmax = None

    if std_clim is None and std_ds is not None:
        std_vmin = np.array(std_ds.min())
        std_vmax = np.array(std_ds.max())
    elif std_clim is not None:
        std_vmin, std_vmax = std_clim
    else:
        std_vmin = None
        std_vmax = None

    ncols = 0
    if era5_raw_ds is not None:
        ncols += 1
    if mean_ds is not None:
        ncols += 1
    if std_ds is not None:
        ncols += 1
    if samples_ds is not None:
        ncols += samples_ds.shape[0]

    res = plt.subplots(1, ncols, subplot_kw=dict(projection=crs), figsize=figsize)
    fig = res[0]
    axes: np.ndarray = res[1]  # type: ignore

    axis_i = 0
    if era5_raw_ds is not None:
        ax = axes[axis_i]
        # era5_raw_ds.sel(lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())).plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=False)
        era5_raw_ds.plot(
            ax=ax,
            cmap=var_cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=add_colorbar,
            cbar_kwargs=dict(label=var_cbar_label),
        )
        ax.set_title("ERA5", fontsize=fontsize)
        axis_i += 1

    if mean_ds is not None:
        ax = axes[axis_i]
        mean_ds.plot(
            ax=ax,
            cmap=var_cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=add_colorbar,
            cbar_kwargs=dict(label=var_cbar_label),
        )
        ax.set_title("ConvNP mean", fontsize=fontsize)
        axis_i += 1

    if samples_ds is not None:
        for i in range(samples_ds.shape[0]):
            ax = axes[axis_i]
            samples_ds.isel(sample=i).plot(
                ax=ax,
                cmap=var_cmap,
                vmin=vmin,
                vmax=vmax,
                add_colorbar=add_colorbar,
                cbar_kwargs=dict(label=var_cbar_label),
            )
            ax.set_title(f"ConvNP sample {i+1}", fontsize=fontsize)
            axis_i += 1

    if std_ds is not None:
        ax = axes[axis_i]
        std_ds.plot(
            ax=ax,
            cmap=std_cmap,
            add_colorbar=add_colorbar,
            vmin=std_vmin,
            vmax=std_vmax,
            cbar_kwargs=dict(label=std_cbar_label),
        )
        ax.set_title("ConvNP std dev", fontsize=fontsize)
        axis_i += 1

    for ax in axes:
        ax.add_feature(cf.BORDERS)
        ax.coastlines()
        if extent is not None:
            ax.set_extent(extent)

        ax.set_xticks(np.linspace(extent[0], extent[1], num=5), crs=crs)
        ax.set_yticks(np.linspace(extent[2], extent[3], num=5), crs=crs)
        ax.set_xticklabels(np.linspace(extent[0], extent[1], num=5), fontsize=fontsize)
        ax.set_yticklabels(np.linspace(extent[2], extent[3], num=5), fontsize=fontsize)

    if task is not None:
        deepsensor.plot.offgrid_context(axes, task, data_processor, task_loader)
    return fig, axes


cfg = DictConfig(
    Config(
        execution=ExecutionConfig(dry_run=False),
        output=OutputConfig(use_wandb=True),
        data=DataConfig(data_provider=DwdDataProviderConfig(daily_averaged=False)),
    )
)
paths = Paths()

if cfg.execution.device == "cuda":
    set_gpu_default_device()

data_processor = get_data_processor(paths)

data_provider = hydra.utils.instantiate(cfg.data.data_provider)

collate_fn = data_provider.get_collate_fn()

test_dataset = data_provider.get_test_data()
task_loader = test_dataset.task_loader

model = hydra.utils.instantiate(cfg.model, data_processor, task_loader)

path = ExperimentPath(config_to_filepath(cfg, cfg.output.out_dir, SKIP_KEYS))
print(path)
checkpoint_manager = CheckpointManager(path)

hires_aux_raw_ds = load_elevation_data(paths, 2000)

# %%

checkpoint = checkpoint_manager.load_checkpoint("best")
model.model.load_state_dict(checkpoint.model_state)

min_lat, max_lat = 47.5, 55
min_lon, max_lon = 6, 15

X_t = hires_aux_raw_ds.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
X_t = X_t.coarsen(lat=2, lon=2, boundary="trim").mean()  # type: ignore

if cfg.data.data_provider.daily_averaged:
    test_time = pd.Timestamp("2023-02-05")
else:
    test_time = pd.Timestamp("2023-02-05 04:00:00")

# for context_sampling in [20, 100, "all"]:
for context_sampling in ["all"]:
    test_task = task_loader(
        test_time,
        context_sampling=[context_sampling, "all"],
        target_sampling="all",
    )
    pred = model.predict(
        test_task,
        X_t=X_t,
    )["t2m"]

    mean_ds, std_ds = pred["mean"], pred["std"]

    fig, axes = gen_test_fig(
        # era5_raw_ds.sel(time=test_task['time'], lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())),
        None,
        mean_ds,
        std_ds,
        task=test_task,
        add_colorbar=True,
        var_cbar_label="2m temperature [°C]",
        std_cbar_label="std dev [°C]",
        # std_clim=(1, 3),
        # var_clim=(0, 10),
        extent=(min_lon, max_lon, min_lat, max_lat),
        figsize=(20, 20 / 3),
    )
    plt.show()
# %%
latmax = 48.5
latmin = 47.5
lonmax = 13
lonmin = 11

# latmin = 51.0
# latmax = 53
# lonmin = 10
# lonmax = 12

# latmin = 51.0
# latmax = 53
# lonmin = 7
# lonmax = 9
fig, axes = gen_test_fig(
    # era5_raw_ds.sel(time=test_task['time'], lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())),
    None,
    mean_ds.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)),  # type: ignore
    std_ds.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)),  # type: ignore
    task=test_task,  # type: ignore
    add_colorbar=True,
    var_cbar_label="2m temperature [°C]",
    std_cbar_label="std dev [°C]",
    extent=(lonmin, lonmax, latmin, latmax),
    figsize=(20, 20 / 3),
)

# %%
