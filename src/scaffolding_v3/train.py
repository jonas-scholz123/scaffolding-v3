import warnings
from typing import Optional

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.metric_logger import WandbLogger
from mlbnb.paths import ExperimentPath
from mlbnb.profiler import WandbProfiler
from mlbnb.rand import seed_everything
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import (
    CheckpointOccasion,
    Config,
    init_configs,
)
from scaffolding_v3.evaluate import evaluate
from scaffolding_v3.plot.plotter import Plotter
from scaffolding_v3.util.instantiate import Experiment

init_configs()

TaskType = tuple[torch.Tensor, torch.Tensor]


@hydra.main(version_base=None, config_name="base", config_path="../config")
def main(cfg: Config) -> float:
    try:
        _configure_outputs()

        logger.debug(OmegaConf.to_yaml(cfg))

        if cfg.runtime.device == "cuda":
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


def _configure_outputs():
    load_dotenv()

    # These are all due to deprecation warnings raised within dependencies.
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="google.protobuf"
    )


class Trainer:
    cfg: Config
    model: torch.nn.Module
    optimizer: Optimizer
    train_loader: DataLoader[TaskType]
    val_loader: DataLoader[TaskType]
    test_data: DataLoader[TaskType]
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

        self._init_wandb()
        self.metric_logger = WandbLogger(
            self.cfg.output.use_wandb,
            self.state,
        )
        self.profiler = WandbProfiler(self.metric_logger)

    def _load_initial_state(self) -> TrainerState:
        start_from = self.cfg.execution.start_from

        initial_state = TrainerState(
            step=0, epoch=0, best_val_loss=np.inf, val_loss=np.inf
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
                "Checkpoint loaded, val loss: {}, epoch: {}, step: {}",
                initial_state.val_loss,
                initial_state.epoch,
                initial_state.step,
            )
        else:
            logger.info("Starting from scratch")

        if initial_state.epoch < self.cfg.execution.epochs:
            # Checkpoint is at end of epoch, add 1 for next epoch.
            initial_state.epoch += 1
            initial_state.step += 1
        else:
            logger.info(
                "Run has concluded (epoch {} / {})",
                initial_state.epoch,
                self.cfg.execution.epochs,
            )
        return initial_state

    def _init_wandb(self) -> None:
        if self.cfg.output.use_wandb:
            wandb.init(
                project=self.cfg.output.wandb_project,
                config=OmegaConf.to_container(self.cfg),  # type: ignore
                dir=self.cfg.output.out_dir,
                name=self.experiment_path.name,
                id=self.experiment_path.name,
            )

    @staticmethod
    def from_config(cfg: Config) -> "Trainer":
        exp = Experiment.from_config(cfg)

        return Trainer(
            cfg,
            exp.model,
            exp.loss_fn,
            exp.optimizer,
            exp.train_loader,
            exp.val_loader,
            exp.generator,
            exp.experiment_path,
            exp.checkpoint_manager,
            exp.scheduler,
            exp.plotter,
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

        if self.plotter:
            self.plotter.plot_prediction(self.model)

        while s.epoch <= self.cfg.execution.epochs:
            logger.info("Starting epoch {} / {}", s.epoch, self.cfg.execution.epochs)
            self.train_epoch()
            val_metrics = self.val_epoch()
            self.metric_logger.log(val_metrics)

            s.val_loss = val_metrics["val_loss"]

            self.save_checkpoint(CheckpointOccasion.LATEST)

            if s.val_loss < s.best_val_loss:
                logger.success("New best val loss: {}", s.val_loss)
                s.best_val_loss = s.val_loss
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

    def train_epoch(self) -> None:
        self.model.train()
        device = next(self.model.parameters()).device
        train_loader: tqdm[TaskType] = tqdm(
            self.train_loader, disable=not self.cfg.output.use_tqdm
        )
        dry_run = self.cfg.execution.dry_run

        if dry_run:
            # Make sure the tensor dimensions are correct.
            features, target = next(iter(train_loader))
            if features.dim() != 4:
                raise ValueError(
                    f"Expected a tensor of shape (batch_size, channels, height, width), \
                    but got {features.shape}"
                )
            if target.dim() != 1:
                msg = (
                    f"Expected a target of shape (batch_size,), but got {target.shape}"
                )
                raise ValueError(msg)

        p = self.profiler
        for features, target in p.profiled_iter("dataload", train_loader):
            with p.profile("data.to"):
                features = features.to(device)
                target = target.to(device)

            with p.profile("forward"):
                output = self.model(features)
                batch_loss = self.loss_fn(output, target)

            with p.profile("backward"):
                batch_loss.backward()
                self.metric_logger.log({"train_loss": batch_loss.item()})

            with p.profile("optimizer.step"):
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.state.step += 1

            if dry_run:
                break

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

    def val_epoch(self) -> dict[str, float]:
        return evaluate(
            self.model,
            self.loss_fn,
            self.val_loader,
            self.cfg.execution.dry_run,
        )


if __name__ == "__main__":
    main()
