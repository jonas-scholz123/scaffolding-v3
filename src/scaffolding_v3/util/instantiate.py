from dataclasses import dataclass
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

from config.config import Config, init_configs
from scaffolding_v3.data.data import make_dataset
from scaffolding_v3.plot.plotter import Plotter


@dataclass
class Experiment:
    model: Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    optimizer: Optimizer
    loss_fn: Module
    scheduler: Optional[LRScheduler]
    generator: Generator
    experiment_path: ExperimentPath
    checkpoint_manager: CheckpointManager
    plotter: Optional[Plotter]

    @staticmethod
    def from_config(cfg: Config) -> "Experiment":
        """
        Instantiates all dependencies for the training loop.

        This is useful for exploration where you want to have easy access to the
        instantiated objects used for training and evaluation.
        """
        logger.info("Instantiating dependencies")

        exp = instantiate(cfg)
        rng = exp.generator.manual_seed(cfg.execution.seed)

        trainset = make_dataset(exp.data, Split.TRAIN, rng)
        valset = make_dataset(exp.data, Split.VAL, rng)
        testset = make_dataset(exp.data, Split.TEST, rng)

        train_loader: DataLoader = exp.data.trainloader(trainset, generator=rng)
        val_loader: DataLoader = exp.data.testloader(valset, generator=rng)
        test_loader: DataLoader = exp.data.testloader(testset, generator=rng)

        loss_fn: Module = exp.loss.to(cfg.runtime.device)
        model: Module = exp.model.to(cfg.runtime.device)
        optimizer = exp.optimizer(model.parameters())

        scheduler: Optional[LRScheduler] = (
            exp.scheduler(optimizer) if exp.scheduler else None
        )

        experiment_path = ExperimentPath.from_config(cfg, cfg.paths.output)

        logger.info("Experiment path: {}", str(experiment_path))
        checkpoint_manager = CheckpointManager(experiment_path)

        plotter: Optional[Plotter] = (
            exp.output.plotter(test_data=valset, save_to=experiment_path)
            if exp.output.plotter
            else None
        )

        logger.info("Finished instantiating dependencies")

        return Experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            generator=rng,
            experiment_path=experiment_path,
            checkpoint_manager=checkpoint_manager,
            plotter=plotter,
        )


def load_config(
    config_name: str = "base",
    mode: str = "dev",
    data: str = "mnist",
    overrides: Optional[list[str]] = None,
) -> Config:
    """
    Load the configuration from the given config name and path.
    """
    init_configs()

    all_overrides = [f"mode={mode}", f"data={data}"] + (overrides or [])

    with initialize(config_path="."):
        cfg: Config = compose(  # type: ignore
            config_name=config_name, overrides=all_overrides
        )
        Experiment.from_config(cfg)

    return cfg
