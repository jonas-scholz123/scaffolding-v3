from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from mlbnb.checkpoint import CheckpointManager
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

    @staticmethod
    def from_config(cfg: Config):
        """
        Instantiates all dependencies for the training loop.

        This is useful for exploration where you want to have easy access to the
        instantiated objects used for training and evaluation.
        """
        generator = Generator(device=cfg.runtime.device).manual_seed(cfg.execution.seed)

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

        experiment_path = ExperimentPath.from_config(cfg, cfg.paths.output)

        logger.info("Experiment path: {}", str(experiment_path))
        checkpoint_manager = CheckpointManager(experiment_path)

        plotter = Plotter(cfg, valset, experiment_path, cfg.output.sample_indices)

        logger.info("Finished instantiating dependencies")

        return Experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            experiment_path=experiment_path,
            checkpoint_manager=checkpoint_manager,
            plotter=plotter,
        )

    @staticmethod
    def from_path(path: str | Path):
        exp_path = ExperimentPath.from_path(Path(path))
        cfg = exp_path.get_config()
        return Experiment.from_config(cfg)


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
