from typing import Iterable, Optional

import torch
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize
from loguru import logger
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath, get_experiment_paths
from torch.nn import Module

from config.config import Config, init_configs
from scaffolding_v3.util.config_filter import DryRunFilter, ModelFilter


def setup(
    config_name: str = "train",
    mode: str = "prod",
    overrides: Optional[list[str]] = None,
) -> Config:
    """
    Constructs a configuration object with the given name and overrides, and sets up the environment.

    NOTE: This is intended as a utility for interactive exploration.

    :param config_name: The name of the configuration to load, defaults to "train"
    :param mode: The mode to set, defaults to "prod" (prod | dev)
    :param overrides: A list of additional overrides to apply
    """
    init_configs()
    GlobalHydra.instance().clear()
    initialize(config_path=None, version_base=None)

    all_overrides = ["mode=" + mode] + (overrides or [])
    cfg: Config = compose(config_name=config_name, overrides=all_overrides)  # type: ignore
    torch.set_default_device(cfg.runtime.device)

    return cfg


def load_best_weights(model: Module, cfg: Config) -> None:
    """
    Find and load the best model checkpoint matching the current architecture.
    Uses validation loss as the metric to compare checkpoints.

    :param model: The model to load the checkpoint into
    :param cfg: The configuration
    """
    experiment_paths = get_experiment_paths(
        cfg.paths.output, [ModelFilter(cfg.model), DryRunFilter(False)]
    )
    best_cm = get_best_checkpoint_manager(experiment_paths)

    if not best_cm:
        logger.warning("No matching checkpoint found")
        return

    best_cm.reproduce_model(model, "best")
    logger.info(f"Loaded best checkpoint from {best_cm.dir}")


def get_best_checkpoint_manager(
    experiment_paths: Iterable[ExperimentPath],
) -> Optional[CheckpointManager]:
    """
    Find the checkpoint manager with the best validation loss from a list of experiment paths.

    :param experiment_paths: The experiment paths to search
    """
    checkpoint_managers = [CheckpointManager(path) for path in experiment_paths]

    best_loss = float("inf")
    best_cm = None

    for cm in checkpoint_managers:
        if not (cm.dir / "best.pt").exists():
            logger.warning(f"Checkpoint 'best' at {cm.dir} does not exist")
            continue

        checkpoint = cm.load_checkpoint("best")

        if not checkpoint.other_state:
            logger.warning(f"Checkpoint 'best' at {cm.dir} has no other state")
            continue

        if checkpoint.other_state["best_val_loss"] < best_loss:
            best_cm = cm
            best_loss = checkpoint.other_state["best_val_loss"]

    if best_cm:
        return best_cm
    else:
        logger.warning("No matching checkpoint found")
        return None
