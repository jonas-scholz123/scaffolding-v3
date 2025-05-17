import pytest
from hydra import compose, initialize

from config.config import Config, init_configs
from scaffolding_v3.util.instantiate import Experiment


@pytest.mark.parametrize(
    "cfg_name,overrides",
    [
        ("base", ["mode=prod", "data=mnist"]),
        ("base", ["mode=dev", "data=cifar10"]),
    ],
)
def test_experiment_init(cfg_name: str, overrides: list[str]) -> None:
    init_configs()
    with initialize(version_base=None, config_path="../src/config"):
        cfg: Config = compose(  # type: ignore
            config_name=cfg_name, overrides=overrides
        )
        Experiment.from_config(cfg)
