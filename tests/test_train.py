"""Checks that determinstic config initializes as expected."""

from hydra import compose, initialize

from config.config import Config, init_configs
from scaffolding_v3.train import Trainer


def test_trainer_initialises() -> None:
    """Checks that the config initializes as expected."""

    init_configs()
    initialize(config_path=None, version_base=None)
    # TODO: make this a parameterised test
    config_name = "train"
    mode = "prod"

    all_overrides = ["mode=" + mode]
    cfg: Config = compose(config_name=config_name, overrides=all_overrides)  # type: ignore
    cfg.runtime.device = "cpu"

    _ = Trainer.from_config(cfg)
