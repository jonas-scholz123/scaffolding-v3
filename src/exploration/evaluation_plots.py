# %%
from pathlib import Path
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
import torch
from deepsensor.train.train import set_gpu_default_device
from omegaconf import DictConfig

from scaffolding_v3.config import (
    Config,
    DataConfig,
    DwdDataProviderConfig,
    ExecutionConfig,
    OutputConfig,
    Paths,
)
from scaffolding_v3.data.dataset import make_taskloader
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

test_dataset = data_provider.get_test_data()
task_loader = make_taskloader(cfg.data, paths, data_processor, test_dataset)

model = hydra.utils.instantiate(cfg.model, data_processor, task_loader)

hires_aux_raw_ds = load_elevation_data(paths, 2000)
# %%

# Best ERA5 model, some artifacts, checkboard pattern
sim_path = Path("/home/jonas/Documents/code/scaffolding-v3/_weights/era5/best_era5.pt")

# "Successful" trained on DWD data from scratch (no pretraining), still get checkerboard pattern. Why not in Toms script?
real_path = Path(
    "/home/jonas/Documents/code/scaffolding-v3/_output/DATA:data_provider=DwdDataProvider_val_fraction=1.0e-01_train_range=2006-01-01_2023-01-01_test_range=2023-01-01_2024-01-01_num_times=10000_num_stations=500_daily_averaged=false_task_loader=TaskLoader_discrete_xarray_sampling=true/trainloader=DataLoader_batch_size=1_shuffle=true_num_workers=0_include_aux_at_targets=true_include_context_in_target=true_ppu=150_hires_ppu=2000_cache=false/MODEL:ConvNP_internal_density=150_unet_channels=64_64_64_64_aux_t_mlp_layers=64_64_64_likelihood=cnp_encoder_scales=3.3e-03_decoder_scale=3.3e-03_verbose=false/OPTIMIZER:Adam_lr=1.0e-05/SCHEDULER:StepLR_step_size=10_gamma=8.0e-01/EXECUTION:device=cuda_dry_run=false_seed=42_use_pretrained=false/checkpoints/best.pt"
)

# Really bad "artifacts"/failed training. What went wrong here?
bad_real_path = Path(
    "/home/jonas/Documents/code/scaffolding-v3/_output/DATA:data_provider=DwdDataProvider_val_fraction=1.0e-01_train_range=2006-01-01_2023-01-01_test_range=2023-01-01_2024-01-01_num_times=10000_num_stations=500_daily_averaged=false_task_loader=TaskLoader_discrete_xarray_sampling=true/trainloader=DataLoader_batch_size=32_shuffle=true_num_workers=0_include_aux_at_targets=true_include_context_in_target=true_ppu=150_hires_ppu=2000_cache=false/MODEL:ConvNP_internal_density=150_unet_channels=64_64_64_64_aux_t_mlp_layers=64_64_64_likelihood=cnp_encoder_scales=3.3e-03_decoder_scale=3.3e-03_verbose=false/OPTIMIZER:Adam_lr=2.5e-03/SCHEDULER:StepLR_step_size=10_gamma=8.0e-01/EXECUTION:device=cuda_dry_run=false_seed=42_use_pretrained=false/checkpoints/best.pt"
)

# "Successful" finetune, quite strong artifacts visible - similar to thesis.
sim2real_path = Path(
    "/home/jonas/Documents/code/scaffolding-v3/_output/DATA:data_provider=DwdDataProvider_val_fraction=1.0e-01_train_range=2006-01-01_2023-01-01_test_range=2023-01-01_2024-01-01_num_times=10000_num_stations=500_daily_averaged=false_task_loader=TaskLoader_discrete_xarray_sampling=true/trainloader=DataLoader_batch_size=32_shuffle=true_num_workers=0_include_aux_at_targets=true_include_context_in_target=true_ppu=150_hires_ppu=2000_cache=false/MODEL:ConvNP_internal_density=150_unet_channels=64_64_64_64_aux_t_mlp_layers=64_64_64_likelihood=cnp_encoder_scales=3.3e-03_decoder_scale=3.3e-03_verbose=false/OPTIMIZER:Adam_lr=1.0e-04/SCHEDULER:StepLR_step_size=10_gamma=8.0e-01/EXECUTION:device=cuda_dry_run=false_seed=42_use_pretrained=true/checkpoints/best.pt"
)


checkpoint = torch.load(real_path)

model.model.load_state_dict(checkpoint.model_state)

min_lat, max_lat = 47.5, 55
min_lon, max_lon = 6, 15

X_t = hires_aux_raw_ds.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
X_t = X_t.coarsen(lat=2, lon=2, boundary="trim").mean()  # type: ignore

if cfg.data.data_provider.daily_averaged:
    test_time = pd.Timestamp("2023-02-05")
else:
    test_time = pd.Timestamp("2023-02-05 04:00:00")

test_time = test_dataset.times[-1]

# for context_sampling in [20, 100, "all"]:
for context_sampling in [100]:
    test_task = task_loader(
        test_time,
        context_sampling=[context_sampling, "all"],
        target_sampling="all",
        seed_override=42,
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
# latmax = 48.5
# latmin = 47.5
# lonmax = 13
# lonmin = 11

# latmax = 54
# latmin = 51
# lonmax = 15
# lonmin = 12

latmin = 50
latmax = 52
lonmin = 10
lonmax = 13

#latmin = 53.0
#latmax = 55
#lonmin = 9
#lonmax = 11

# latmin = 51.0
# latmax = 53
# lonmin = 7
# lonmax = 9

#latmin = 50
#latmax = 52
#lonmin = 6
#lonmax = 8
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
