import xarray as xr

from scaffolding_v3.config import Paths
from scaffolding_v3.data.dataprovider import DataProvider, DeepSensorDataset


def load_era5(paths: Paths) -> xr.Dataset:
    return xr.open_dataset(paths.era5, engine="netcdf4")


class Era5DataProvider(DataProvider):
    def __init__(
        self,
        paths: Paths,
        val_fraction: float,
        train_range: tuple[str, str],
        test_range: tuple[str, str],
    ):
        self.ds = load_era5(paths)
        times = self.ds.time.values
        val_cutoff = int((1 - val_fraction) * len(times))
        self.train_range = (times[0], times[val_cutoff])
        self.val_range = (times[val_cutoff], times[-1])
        self.test_range = test_range

    def get_train_data(self) -> DeepSensorDataset:
        return self._to_dataset(self.train_range)

    def get_val_data(self) -> DeepSensorDataset:
        return self._to_dataset(self.val_range)

    def get_test_data(self) -> DeepSensorDataset:
        return self._to_dataset(self.test_range)

    def _to_dataset(self, range: tuple[str, str]) -> DeepSensorDataset:
        data = self.ds.sel(time=slice(*range))
        return DeepSensorDataset(data, data, data.time.values)
