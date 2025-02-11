# %%
import os

import cartopy.crs as ccrs
import cartopy.feature as cf
import deepsensor.torch
import lab as B
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from deepsensor import Task
from deepsensor.train.train import set_gpu_default_device, train_epoch
from hydra.utils import instantiate
from mlbnb.types import Split
from tqdm import tqdm

from scaffolding_v3.config import Config, DwdDataConfig, DwdDataProviderConfig, Paths
from scaffolding_v3.data.dataset import make_dataset
from scaffolding_v3.data.dwd import get_data_processor
from scaffolding_v3.data.elevation import load_elevation_data


def compute_val_loss(model, val_tasks):
    val_losses = []
    for task in val_tasks:
        val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
    return np.mean(val_losses)


fontsize = 14

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
    extent=None,
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

    fig, axes = plt.subplots(1, ncols, subplot_kw=dict(projection=crs), figsize=figsize)

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
            ax.set_title(f"ConvNP sample {i + 1}", fontsize=fontsize)
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
    if task is not None:
        deepsensor.plot.offgrid_context(axes, task, data_processor, task_loader)
    print("Done")
    return fig, axes


crs = ccrs.PlateCarree()

USE_TOMS_DATA = False
USE_TPI = True
USE_ALL_MY_DATA = False

set_gpu_default_device()

cfg = Config(
    data=DwdDataConfig(
        include_context_in_target=False,
        include_tpi=USE_TPI,
        data_provider=DwdDataProviderConfig(
            daily_averaged=True,
            num_times=1000,
        ),
    )
)

model_folder = "/home/jonas/Documents/code/scaffolding-v3/tom/models/stationinterp/"
fig_folder = "/home/jonas/Documents/code/scaffolding-v3/tom/figures/stationinterp/"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

paths = Paths()
# %%
data_provider = instantiate(cfg.data.data_provider)
data_processor = get_data_processor(paths, cfg.data)
print(data_processor)
trainset = make_dataset(cfg.data, cfg.paths, data_provider, Split.TRAIN, data_processor)
valset = make_dataset(cfg.data, cfg.paths, data_provider, Split.VAL, data_processor)
# %%
task_loader = trainset.task_loader
model = instantiate(cfg.model, data_processor, task_loader)
# %%
train_tasks: list[Task] = list(trainset)  # type: ignore
fig = deepsensor.plot.context_encoding(model, train_tasks[0], task_loader, size=7)
val_tasks: list[Task] = list(valset)  # type: ignore
val_dates = valset.times
val_task_loader = valset.task_loader
# %%
n_epochs = 60
train_losses = []
val_losses = []

val_loss_best = np.inf

for epoch in tqdm(range(n_epochs)):
    train_tasks: list[Task] = list(trainset)  # type: ignore
    batch_losses = train_epoch(model, train_tasks)
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    val_loss = compute_val_loss(model, val_tasks)
    val_losses.append(val_loss)

    if val_loss < val_loss_best:
        import os

        import torch

        print("new best")

        val_loss_best = val_loss
        torch.save(model.model.state_dict(), model_folder + "model.pt")

    print(f"Epoch {epoch} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(train_losses, label="train")
ax.plot(val_losses, label="val")
ax.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
fname = "loss.png"
fig.savefig(os.path.join(fig_folder, fname), bbox_inches="tight")

# %%
model.model.load_state_dict(torch.load(model_folder + "model.pt"))

hires_aux_raw_ds = load_elevation_data(paths, 2000)

for i in range(3):
    dt = val_dates[i]

    # for context_sampling in [20, 100, "all"]:
    for context_sampling in [100]:
        test_task = val_task_loader(dt, [context_sampling, "all"])
        pred = model.predict(
            test_task, X_t=hires_aux_raw_ds.sel(lat=slice(55, 47.5), lon=slice(6, 15))
        )["t2m"]

        mean_ds = pred["mean"]
        std_ds = pred["std"]

        if context_sampling == 100:
            mid_mean_ds = mean_ds
            mid_std_ds = std_ds
            mid_test_task = test_task

        fig, axes = gen_test_fig(
            # era5_raw_ds.sel(time=test_task['time'], lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())),
            None,
            mean_ds.coarsen(lat=5, lon=5, boundary="trim").mean(),
            std_ds.coarsen(lat=5, lon=5, boundary="trim").mean(),
            task=test_task,
            add_colorbar=True,
            var_cbar_label="2m temperature [°C]",
            std_cbar_label="std dev [°C]",
            extent=(6, 15, 47.5, 55),
            figsize=(20, 20 / 3),
        )
        fname = f"downscale_{context_sampling}"
        fig.savefig(os.path.join(fig_folder, fname + ".png"), bbox_inches="tight")
        fig.savefig(
            os.path.join(fig_folder, fname + ".pdf"), bbox_inches="tight", dpi=300
        )

    latmax = 48.0
    latmin = 47.5
    lonmax = 13
    lonmin = 11
    fig, axes = gen_test_fig(
        None,
        mid_mean_ds.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)),  # type: ignore
        mid_std_ds.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)),  # type: ignore
        task=mid_test_task,  # type: ignore
        add_colorbar=True,
        var_cbar_label="2m temperature [°C]",
        std_cbar_label="std dev [°C]",
        extent=(lonmin, lonmax, latmin, latmax),
        figsize=(20, 20 / 3),
    )
    fname = "downscale_zoom_kpng"
    fig.savefig(os.path.join(fig_folder, fname), bbox_inches="tight")

    latmin = 48.5
    latmax = 50.5
    lonmin = 7.5
    lonmax = 9.5

    fig, axes = gen_test_fig(
        None,
        mid_mean_ds.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)),  # type: ignore
        mid_std_ds.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax)),  # type: ignore
        task=mid_test_task,  # type: ignore
        add_colorbar=True,
        var_cbar_label="2m temperature [°C]",
        std_cbar_label="std dev [°C]",
        extent=(lonmin, lonmax, latmin, latmax),
        figsize=(20, 20 / 3),
    )
    fname = "downscale_zoom2_kpng"
    fig.savefig(os.path.join(fig_folder, fname), bbox_inches="tight")
    plt.show()

# %%
