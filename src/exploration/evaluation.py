# %%
import hydra
import matplotlib.pyplot as plt
import pandas as pd
from deepsensor.train.train import set_gpu_default_device
from evaluation_plots import gen_test_fig
from mlbnb.checkpoint import CheckpointManager, ExperimentPath
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
from scaffolding_v3.evaluate import drop_boring_cols

# %%
cfg = DictConfig(
    Config(
        execution=ExecutionConfig(dry_run=False),
        output=OutputConfig(use_wandb=True),
        data=DataConfig(data_provider=DwdDataProviderConfig(daily_averaged=False)),
        # data=DataConfig(data_provider=Era5DataProviderConfig()),
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
path = Paths.output / "evaluation_Era5DataProvider.csv"
# path = Paths.output / "evaluation_DwdDataProvider.csv"
df = pd.read_csv(path)
df = df[df["data.data_provider._target_"].str.endswith("Era5DataProvider")]
df = drop_boring_cols(df)
best_df = df[df["test_val_loss"] == df["test_val_loss"].min()]
best_df.T
# %%
df.sort_values("test_val_loss")

# %%

best_path = ExperimentPath(best_df["path"].values[0])

best_path
# %%

cm = CheckpointManager(best_path)
cm.reproduce_model(model.model, "best")

min_lat, max_lat = 47.5, 55
min_lon, max_lon = 6, 15

X_t = hires_aux_raw_ds.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
X_t = X_t.coarsen(lat=2, lon=2, boundary="trim").mean()  # type: ignore

try:
    if cfg.data.data_provider.daily_averaged:
        test_time = pd.Timestamp("2023-02-05")
    else:
        test_time = pd.Timestamp("2023-02-05 04:00:00")
except Exception:
    test_time = pd.Timestamp("2023-02-05 04:00:00")

test_time = test_dataset.times[-1]

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
        extent=(min_lon, max_lon, min_lat, max_lat),
        figsize=(20, 20 / 3),
    )
    plt.show()

latmin = 50
latmax = 52
lonmin = 10
lonmax = 13

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
