# %%
import cartopy.crs as ccrs
import cartopy.feature as feature
import deepsensor.torch  # noqa
import matplotlib.pyplot as plt
import numpy as np
from deepsensor import Task
from deepsensor.model.convnp import ConvNP
from deepsensor.plot import offgrid_context
from hydra.utils import instantiate
from matplotlib.figure import Figure
from mlbnb.types import Split

from scaffolding_v3.config import Config, DataConfig, DwdDataProviderConfig
from scaffolding_v3.data.dataprovider import DataProvider
from scaffolding_v3.data.dataset import make_dataset
from scaffolding_v3.data.dwd import get_data_processor, get_dwd_data
from scaffolding_v3.plot.util import make_uniform_grid

# %%

cfg = Config(data=DataConfig(data_provider=DwdDataProviderConfig()))

data_provider: DataProvider = instantiate(cfg.data.data_provider)
data_processor = get_data_processor(cfg.paths)
dataset = make_dataset(cfg.data, cfg.paths, data_provider, Split.VAL, data_processor)
task_loader = dataset.task_loader
model = instantiate(cfg.model, data_processor, task_loader)
# %%
dwd_data = get_dwd_data(cfg.paths)
# %%
time = dataset.times[0]
task = task_loader(time)

geo = cfg.geo
bounds = (geo.min_lon, geo.max_lon, geo.min_lat, geo.max_lat)


def plot_prediction(task: Task, model: ConvNP, cmap: str = "jet") -> Figure:
    def lons_and_lats(df):
        lats = df.index.get_level_values("lat")
        lons = df.index.get_level_values("lon")
        return lons, lats

    # Get temperature at all stations on the task date.
    truth = dwd_data.xs(time, level="time")["t2m"]

    pred = model.predict(task, X_t=truth)["t2m"]
    mean_ds, std_ds = pred["mean"], pred["std"]
    # Fix rounding errors along dimensions.
    err_da = mean_ds - truth
    err_da = err_da.dropna()

    hires_grid = make_uniform_grid(*bounds, resolution=0.01)
    # Higher resolution prediction everywhere.
    pred = model.predict(task, X_t=hires_grid, resolution_factor=1)["t2m"]
    mean_ds, std_ds = pred["mean"], pred["std"]

    proj = ccrs.TransverseMercator(central_longitude=10, approx=False)

    fig, axs = plt.subplots(
        subplot_kw={"projection": proj},
        nrows=1,
        ncols=4,
        figsize=(10, 2.5),
    )
    if not isinstance(axs, np.ndarray):
        return

    transform = ccrs.PlateCarree()
    vmin, vmax = 0.9 * truth.min(), 1.1 * truth.max()

    s = 3**2

    axs[0].set_title("Truth [°C]")
    im = axs[0].scatter(
        *lons_and_lats(truth),
        s=s,
        c=truth,
        transform=transform,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    fig.colorbar(im, ax=axs[0])

    im = mean_ds.plot(
        cmap=cmap,
        ax=axs[1],
        transform=transform,
        vmin=vmin,
        vmax=vmax,
        extend="both",
    )
    axs[1].set_title("Pred. Mean [°C]")

    im = std_ds.plot(
        cmap="viridis_r",
        ax=axs[2],
        transform=transform,
        extend="both",
    )
    axs[2].set_title("Pred Std. Dev. [°C]")

    axs[3].set_title("Prediction Error [°C]")

    biggest_err = err_da.abs().max()
    im = axs[3].scatter(
        *lons_and_lats(err_da),
        s=s,
        c=err_da,
        cmap="seismic",
        vmin=-biggest_err,
        vmax=biggest_err,
        transform=transform,
    )
    fig.colorbar(im, ax=axs[3])

    # Don't add context points to mean prediction.
    context_axs = [ax for i, ax in enumerate(axs) if i != 1]

    offgrid_context(
        context_axs,
        task,
        data_processor,
        s=3**2,
        linewidths=0.5,
        add_legend=False,
        transform=ccrs.PlateCarree(),
    )

    for ax in axs:
        # Remove annoying label.
        ax.collections[0].colorbar.ax.set_ylabel("")
        ax.set_extent(bounds, crs=transform)
        ax.add_feature(feature.BORDERS, linewidth=0.25)
        ax.coastlines(linewidth=0.25)
    return fig


if isinstance(task, Task):
    fig = plot_prediction(task, model, cmap="jet")
    out_path = cfg.paths.output / "prediction.png"
    fig.savefig(out_path, dpi=300)

# %%
data_provider: DataProvider = instantiate(cfg.data.data_provider)
test_data = data_provider.get_test_data()
# %%
test_data.target["t2m"]
