"""Checks that determinstic config initializes as expected."""

from scaffolding_v3.config import Config, load_config
from scaffolding_v3.train import Trainer

from hydra import compose, initialize


def test_trainer_initialises() -> None:
    """Checks that the config initializes as expected."""

    load_config()
    initialize(config_path=None, version_base=None)
    # TODO: make this a parameterised test
    config_name = "train"
    mode = "prod"

    all_overrides = ["mode=" + mode]
    cfg: Config = compose(config_name=config_name, overrides=all_overrides)  # type: ignore
    cfg.execution.device = "cpu"

    _ = Trainer.from_config(cfg)
