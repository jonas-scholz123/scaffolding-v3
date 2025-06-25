"""Checks that determinstic config initializes as expected."""

from scaffolding_v3.train import Trainer
from scaffolding_v3.util.instantiate import Experiment, load_config
import pytest


@pytest.mark.parametrize("mode", ["prod", "dev"])
@pytest.mark.parametrize("data", ["mnist", "cifar10"])
def test_trainer_initialises(mode: str, data: str) -> None:
    """Checks that the config initializes as expected."""

    load_config()
    config_name = "base"

    cfg = load_config(
        config_name=config_name,
        mode=mode,
        data=data,
        config_path="../../config",
    )
    exp = Experiment.from_config(cfg)

    _ = Trainer.from_experiment(exp, cfg)
