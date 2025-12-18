"""Checks that determinstic config initializes as expected."""

import hashlib

import pytest
import torch

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


def get_model_checksum(model: torch.nn.Module) -> str:
    """Computes a checksum over the model parameters."""
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


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
    train_loop(cfg1)
    exp1 = Experiment.from_config(cfg1, checkpoint="latest")
    checksum1 = get_model_checksum(exp1.model)

    # Run 2
    cfg2 = load_config(overrides=overrides, config_path="../../config")
    train_loop(cfg2)
    exp2 = Experiment.from_config(cfg2, checkpoint="latest")
    checksum2 = get_model_checksum(exp2.model)

    assert checksum1 == checksum2, f"Non-deterministic training! {checksum1} != {checksum2}"
