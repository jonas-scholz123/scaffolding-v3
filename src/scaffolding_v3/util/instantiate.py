from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.namegen import gen_run_name
from mlbnb.paths import ExperimentPath
from mlbnb.types import Split
from torch import Generator
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from scaffolding_v3.config import Config, init_config
from scaffolding_v3.data.data import make_dataset
from scaffolding_v3.plot.plotter import Plotter


@dataclass
class Experiment:
    cfg: Config
    model: Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    optimizer: Optimizer
    scheduler: Optional[LRScheduler]
    generator: Generator
    experiment_path: ExperimentPath
    checkpoint_manager: CheckpointManager
    plotter: Plotter
    trainer_state: TrainerState

    @staticmethod
    def from_config(
        cfg: Config,
        exp_path: Optional[ExperimentPath] = None,
        checkpoint: Optional[str] = None,
    ) -> "Experiment":
        """
        Instantiates all dependencies for the training loop.

        This is useful for exploration where you want to have easy access to the
        instantiated objects used for training and evaluation.
        """
        generator = Generator(device="cpu").manual_seed(cfg.execution.seed)

        logger.info("Instantiating dependencies")

        trainset = make_dataset(cfg.data, Split.TRAIN, generator)
        valset = make_dataset(cfg.data, Split.VAL, generator)
        testset = make_dataset(cfg.data, Split.TEST, generator)

        train_loader: DataLoader = instantiate(
            cfg.data.trainloader, trainset, generator=generator
        )
        val_loader: DataLoader = instantiate(
            cfg.data.testloader, valset, generator=generator
        )
        test_loader: DataLoader = instantiate(
            cfg.data.testloader, testset, generator=generator
        )

        model: Module = instantiate(cfg.model).to(cfg.runtime.device)

        optimizer: Optimizer = instantiate(cfg.optimizer, model.parameters())

        scheduler: Optional[LRScheduler] = (
            instantiate(cfg.scheduler, optimizer) if cfg.scheduler else None
        )

        if not exp_path:
            path = cfg.paths.output / gen_run_name()
            exp_path = ExperimentPath.from_path(path)
        logger.info("Experiment path: {}", str(exp_path))

        checkpoint_manager = CheckpointManager(exp_path)

        trainer_state = TrainerState(
            step=0, samples_seen=0, epoch=0, best_val_loss=np.inf, val_loss=np.inf
        )

        start_from = cfg.execution.start_from if not checkpoint else checkpoint

        if start_from and checkpoint_manager.checkpoint_exists(start_from):
            checkpoint_manager.reproduce(
                cfg.execution.start_from,  # ty: ignore
                model,
                optimizer,
                generator,
                scheduler,
                trainer_state,
            )

            logger.info(
                "Checkpoint loaded, val loss: {}, samples seen: {}",
                trainer_state.val_loss,
                trainer_state.samples_seen,
            )
        else:
            logger.info("Starting from scratch")

        plotter = Plotter(cfg, valset, exp_path, cfg.output.sample_indices)

        logger.info("Finished instantiating dependencies")

        return Experiment(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            experiment_path=exp_path,
            checkpoint_manager=checkpoint_manager,
            plotter=plotter,
            trainer_state=trainer_state,
        )

    @staticmethod
    def from_path(
        path: str | Path | ExperimentPath, checkpoint: str = "best"
    ) -> "Experiment":
        if not isinstance(path, ExperimentPath):
            exp_path = ExperimentPath.from_path(Path(path))
        cfg: Config = exp_path.get_config()  # type: ignore
        return Experiment.from_config(cfg, exp_path=exp_path, checkpoint=checkpoint)


def load_config(
    config_name: str = "base",
    mode: str = "dev",
    data: str = "mnist",
    overrides: Optional[list[str]] = None,
    config_path: str = "../../config",
) -> Config:
    """
    Load the configuration from the given config name and path.
    """
    init_config()

    all_overrides = [f"mode={mode}", f"data={data}"] + (overrides or [])

    with initialize(config_path=config_path, version_base="1.1"):
        cfg: Config = compose(  # type: ignore
            config_name=config_name, overrides=all_overrides
        )

    return cfg
