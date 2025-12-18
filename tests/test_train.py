"""Checks that determinstic config initializes as expected."""

import pytest

from scaffolding_v3.train import Trainer, train_loop
from scaffolding_v3.util.instantiate import Experiment, load_config


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
        # For remote testing, disable WANDB.
        overrides=["output.use_wandb=False"],
        config_path="../../config",
    )
    exp = Experiment.from_config(cfg)

    _ = Trainer.from_experiment(exp, cfg)


def test_training_determinism() -> None:
    """Checks that training is deterministic by comparing model checksums."""
    overrides = [
        "data=mnist",
        "mode=dev",
        "output.use_wandb=False",
        "execution.num_train_samples=64",  # 2 steps of 32
        "execution.seed=42",
    ]

    # Run 1
    cfg1 = load_config(overrides=overrides, config_path="../../config")
    val_loss1 = train_loop(cfg1)

    # Run 2
    cfg2 = load_config(overrides=overrides, config_path="../../config")
    val_loss2 = train_loop(cfg2)

    assert val_loss1 == val_loss2, (
        f"Non-deterministic training! {val_loss1} != {val_loss2}"
    )
