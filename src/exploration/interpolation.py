#%%
import deepsensor.torch
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP

from deepsensor.train.train import train_epoch, set_gpu_default_device

import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import seaborn as sns
# %%
crs = ccrs.PlateCarree()

use_gpu = True
if use_gpu:
    set_gpu_default_device()
#%%

elevation_path = "../../_data/elevation/elevation.nc"
hires_aux_raw_ds = xr.open_mfdataset(elevation_path).coarsen(lat=5, lon=5, boundary='trim').mean()
hires_aux_raw_ds = hires_aux_raw_ds
print(hires_aux_raw_ds)
#%%
print(hires_aux_raw_ds)
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
hires_aux_raw_ds['height'].plot(ax=ax)
ax.add_feature(cf.BORDERS)
ax.coastlines()
#%%
aux_raw_ds = hires_aux_raw_ds.coarsen(lat=20, lon=20, boundary='trim').mean()["height"]
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
aux_raw_ds.plot(ax=ax)
ax.add_feature(cf.BORDERS)
ax.coastlines()
print(aux_raw_ds)
#%%
# Print resolution of lowres and hires elevation data
print(f"Lowres lat resolution: {np.abs(np.diff(aux_raw_ds.coords['lat'].values)[0]):.4f} degrees")
print(f"Lowres lon resolution: {np.abs(np.diff(aux_raw_ds.coords['lon'].values)[0]):.4f} degrees")
print(f"Hires lat resolution: {np.abs(np.diff(hires_aux_raw_ds.coords['lat'].values)[0]):.4f} degrees")
print(f"Hires lon resolution: {np.abs(np.diff(hires_aux_raw_ds.coords['lon'].values)[0]):.4f} degrees")
#%%
from scaffolding_v3.data.dwd import DwdDataProvider
from scaffolding_v3.config import Paths

ppu = 150
dwd_data_provider = DwdDataProvider(Paths(), 500, 1000, 0.1, ppu)
#%%
station_raw_df = dwd_data_provider.df
station_raw_df = station_raw_df.reset_index()
station_raw_df['time'] = pd.to_datetime(station_raw_df['time'])
station_raw_df = station_raw_df.set_index(["time", "lat", "lon"])
station_raw_df = station_raw_df[["t2m"]]
#%%
data_processor = DataProcessor(x1_name="lat", x1_map=(aux_raw_ds["lat"].min(), aux_raw_ds["lat"].max()), x2_name="lon", x2_map=(aux_raw_ds["lon"].min(), aux_raw_ds["lon"].max()))
station_df = data_processor([station_raw_df])[0]
aux_ds, hires_aux_ds = data_processor([aux_raw_ds, hires_aux_raw_ds], method="min_max")
print(data_processor)
#%%
task_loader = TaskLoader(context=[station_df, aux_ds], target=station_df, aux_at_targets=hires_aux_ds, links=[(0, 0)])
task_loader.load_dask()

print(task_loader)
#%%
model = ConvNP(data_processor, task_loader, unet_channels=(64,)*4, internal_density=ppu, likelihood="cnp")
#%%
times = station_df.index.get_level_values("time").unique()
train_times = times[:int(len(times)*0.8)][::5]
val_times = times[int(len(times)*0.8):][::5]
#%%
from tqdm import tqdm
train_tasks = []
for time in tqdm(train_times):
    split_frac = np.random.uniform(0.0, 1.0)
    task = task_loader(time, context_sampling=["split", "all"], target_sampling="split", split_frac=split_frac)
    train_tasks.append(task)
#%%
val_tasks = []
for time in tqdm(val_times):
    split_frac = np.random.uniform(0.0, 1.0)
    task = task_loader(time, context_sampling=["split", "all"], target_sampling="split", split_frac=split_frac)
    val_tasks.append(task)
#%%
train_tasks[0]
#%%
fig = deepsensor.plot.context_encoding(model, train_tasks[0], task_loader)
#%%
fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(projection=crs))
ax.coastlines()
ax.add_feature(cf.BORDERS)
deepsensor.plot.offgrid_context(ax, train_tasks[4], data_processor, task_loader, plot_target=True, add_legend=True, linewidths=0.5)
plt.show()
#%%
from scaffolding_v3.config import Paths
import lab as B
from tqdm import tqdm

def compute_val_loss(model, val_tasks):
    val_losses = []
    for task in val_tasks:
        val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
    return np.mean(val_losses)

n_epochs = 80
train_losses = []
val_losses = []

val_loss_best = np.inf

for epoch in tqdm(range(n_epochs)):
    batch_losses = train_epoch(model, train_tasks)
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    val_loss = compute_val_loss(model, val_tasks)
    val_losses.append(val_loss)

    print("Epoch: ", epoch)
    print("Val loss: ", val_loss)
    print("Train loss: ", train_loss)
#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(train_losses, label="train")
ax.plot(val_losses, label="val")
ax.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")