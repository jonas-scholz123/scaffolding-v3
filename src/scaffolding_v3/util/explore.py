from typing import Iterable, Optional

from loguru import logger
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath, get_experiment_paths
from torch.nn import Module

from scaffolding_v3.config import Config
from scaffolding_v3.util.config_filter import DryRunFilter, ModelFilter


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
