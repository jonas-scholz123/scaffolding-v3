import warnings
from typing import Optional

import hydra
import numpy as np
import torch
import torch.optim.lr_scheduler
import wandb
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.metric_logger import MetricLogger
from mlbnb.paths import ExperimentPath
from mlbnb.rand import seed_everything
from mlbnb.types import Split
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaffolding_v3.config import (
    SKIP_KEYS,
    CheckpointOccasion,
    Config,
    load_config,
)
from scaffolding_v3.data.data import make_dataset
from scaffolding_v3.evaluate import evaluate
from scaffolding_v3.plot.plotter import Plotter

load_config()


@hydra.main(version_base=None, config_name="train", config_path="")
def main(cfg: Config) -> float:
    try:
        _configure_outputs(cfg)

        logger.debug(OmegaConf.to_yaml(cfg))

        if cfg.execution.device == "cuda":
            torch.set_default_device("cuda")

        seed_everything(cfg.execution.seed)

        trainer = Trainer.from_config(cfg)
        trainer.train_loop()
        if cfg.output.use_wandb:
            wandb.finish()
        return trainer.state.best_val_loss
    except Exception as e:
        logger.exception("An error occurred during training: {}", e)
        raise e


def _configure_outputs(cfg: Config):
    load_dotenv()

    # These are all due to deprecation warnings raised within dependencies.
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="google.protobuf"
    )

    if cfg.output.use_wandb:
        wandb.init(
            project="scaffolding_v3",
            config=OmegaConf.to_container(cfg),  # type: ignore
            dir=cfg.output.out_dir,
        )  # type: ignore


class Trainer:
    cfg: Config
    model: torch.nn.Module
    optimizer: Optimizer
    train_loader: DataLoader
    val_loader: DataLoader
    test_data: DataLoader
    generator: torch.Generator
    experiment_path: ExperimentPath
    checkpoint_manager: CheckpointManager
    scheduler: Optional[LRScheduler]
    plotter: Optional[Plotter]
    state: TrainerState

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        generator: torch.Generator,
        experiment_path: ExperimentPath,
        checkpoint_manager: CheckpointManager,
        scheduler: Optional[LRScheduler],
        plotter: Optional[Plotter],
    ):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.generator = generator
        self.experiment_path = experiment_path
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler
        self.plotter = plotter

        self.state = self._load_initial_state()

    def _load_initial_state(self) -> TrainerState:
        start_from = self.cfg.execution.start_from

        initial_state = TrainerState(
            epoch=0,
            best_val_loss=np.inf,
        )

        weights_name = self.cfg.execution.start_weights
        if weights_name:
            weights_path = self.cfg.paths.weights / weights_name
            CheckpointManager.reproduce_model_from_path(self.model, weights_path)
            logger.info(
                "Pretrained model loaded from path {}, starting from pretrained.",
                weights_path,
            )

        if start_from and self.checkpoint_manager.checkpoint_exists(start_from.value):
            self.checkpoint_manager.reproduce(
                start_from.value,
                self.model,
                self.optimizer,
                self.generator,
                self.scheduler,
                initial_state,
            )

            logger.info(
                "Checkpoint loaded, best val loss: {}, epoch: {}",
                initial_state.best_val_loss,
                initial_state.epoch,
            )
        else:
            logger.info("Starting from scratch")

        if initial_state.epoch < self.cfg.execution.epochs:
            # Checkpoint is at end of epoch, add 1 for next epoch.
            initial_state.epoch += 1
        else:
            logger.info(
                "Run has concluded (epoch {} / {})",
                initial_state.epoch,
                self.cfg.execution.epochs,
            )
        return initial_state

    @staticmethod
    def from_config(cfg: Config) -> "Trainer":
        generator = torch.Generator(device=cfg.execution.device).manual_seed(
            cfg.execution.seed
        )
        logger.warning(generator.device)

        logger.info("Instantiating dependencies")

        trainset = make_dataset(cfg.data, Split.TRAIN, generator)
        valset = make_dataset(cfg.data, Split.VAL, generator)

        train_loader: DataLoader = instantiate(
            cfg.data.trainloader, trainset, generator=generator
        )
        val_loader: DataLoader = instantiate(
            cfg.data.testloader, valset, generator=generator
        )

        in_channels = cfg.data.in_channels
        num_classes = cfg.data.num_classes
        sidelength = cfg.data.sidelength

        model: nn.Module = instantiate(
            cfg.model,
            in_channels=in_channels,
            num_classes=num_classes,
            sidelength=sidelength,
        ).to(cfg.execution.device)
        if cfg.execution.compile:
            model.compile()
        loss_fn: nn.Module = instantiate(cfg.loss).to(cfg.execution.device)

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

        return Trainer(
            cfg,
            model,
            loss_fn,
            optimizer,
            train_loader,
            val_loader,
            generator,
            experiment_path,
            checkpoint_manager,
            scheduler,
            plotter,
        )

    def train_loop(self):
        s = self.state
        self._save_config()

        if self.cfg.output.use_wandb and self.cfg.output.log_gradients:
            wandb.watch(
                self.model,
                self.loss_fn,
                log="all",
                log_freq=self.cfg.output.gradient_log_freq,
            )

        logger.info("Starting training")

        metric_logger = MetricLogger(self.cfg.output.use_wandb)

        if self.plotter:
            self.plotter.plot_prediction(self.model)

        while s.epoch <= self.cfg.execution.epochs:
            logger.info("Starting epoch {} / {}", s.epoch, self.cfg.execution.epochs)
            train_metrics = self.train_step()
            metric_logger.log(s.epoch, train_metrics)
            val_metrics = self.val_step()
            metric_logger.log(s.epoch, val_metrics)

            self.save_checkpoint(CheckpointOccasion.LATEST)

            if val_metrics["val_loss"] < s.best_val_loss:
                logger.success("New best val loss: {}", val_metrics["val_loss"])
                s.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(CheckpointOccasion.BEST)

            if self.scheduler:
                self.scheduler.step()

            if self.plotter:
                self.plotter.plot_prediction(self.model, s.epoch)

            s.epoch += 1
            s.best_val_loss = s.best_val_loss

        logger.success("Finished training")

    def _save_config(self) -> None:
        with self.experiment_path.open("cfg.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    def train_step(self) -> dict[str, float]:
        train_loss = 0.0

        self.model.train()
        iterator = (
            tqdm(self.train_loader) if self.cfg.output.use_tqdm else self.train_loader
        )

        for data, target in iterator:
            self.optimizer.zero_grad()

            data = data.to(self.cfg.execution.device)
            target = target.to(self.cfg.execution.device)

            output = self.model(data)
            batch_loss = self.loss_fn(output, target)

            batch_loss.backward()
            self.optimizer.step()

            train_loss += float(batch_loss.detach().cpu().numpy())

            if self.cfg.execution.dry_run:
                break

        return {"train_loss": train_loss / len(self.train_loader)}

    def save_checkpoint(self, occasion: CheckpointOccasion):
        if self.cfg.output.save_checkpoints:
            self.checkpoint_manager.save_checkpoint(
                occasion.value,
                self.model,
                self.optimizer,
                self.generator,
                self.scheduler,
                self.state,
            )

    def val_step(self) -> dict[str, float]:
        return evaluate(
            self.model,
            self.loss_fn,
            self.val_loader,
            self.cfg.execution.dry_run,
            self.cfg.output.use_tqdm,
        )


if __name__ == "__main__":
    main()
