from dataclasses import dataclass
from typing import Optional
from loguru import logger
from torch import Generator
from torch.nn import Module
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mlbnb.types import Split
from mlbnb.paths import ExperimentPath
from mlbnb.checkpoint import CheckpointManager

from scaffolding_v3.config import SKIP_KEYS, Config
from scaffolding_v3.plot.plotter import Plotter
from scaffolding_v3.data.data import make_dataset


@dataclass
class Dependencies:
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
    def from_config(cfg: Config):
        """
        Instantiates all dependencies for the training loop.

        This is useful for exploration where you want to have easy access to the
        instantiated objects used for training and evaluation.
        """
        generator = Generator(device=cfg.execution.device).manual_seed(
            cfg.execution.seed
        )

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

        in_channels = cfg.data.in_channels
        num_classes = cfg.data.num_classes
        sidelength = cfg.data.sidelength

        model: Module = instantiate(
            cfg.model,
            in_channels=in_channels,
            num_classes=num_classes,
            sidelength=sidelength,
        ).to(cfg.execution.device)
        if cfg.execution.compile:
            model.compile()
        loss_fn: Module = instantiate(cfg.loss).to(cfg.execution.device)

        optimizer: Optimizer = instantiate(cfg.optimizer, model.parameters())

        scheduler: Optional[LRScheduler] = (
            instantiate(cfg.scheduler, optimizer) if cfg.scheduler else None
        )

        experiment_path = ExperimentPath.from_config(cfg, cfg.paths.output, SKIP_KEYS)

        logger.info("Experiment path: {}", str(experiment_path))
        checkpoint_manager = CheckpointManager(experiment_path)

        if cfg.output.plot:
            plotter = Plotter(cfg, valset, experiment_path, cfg.output.sample_indices)
        else:
            plotter = None

        logger.info("Finished instantiating dependencies")

        return Dependencies(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            generator=generator,
            experiment_path=experiment_path,
            checkpoint_manager=checkpoint_manager,
            plotter=plotter,
        )
