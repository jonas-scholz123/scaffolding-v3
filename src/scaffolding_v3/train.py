from typing import Optional

import hydra
import torch
import torch.optim.lr_scheduler
import wandb
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.metric_logger import WandbLogger
from mlbnb.paths import ExperimentPath
from mlbnb.profiler import WandbProfiler
from mlbnb.rand import seed_everything
from omegaconf import OmegaConf
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaffolding_v3.config import Config, init_config
from scaffolding_v3.evaluate import evaluate
from scaffolding_v3.plot.plotter import Plotter
from scaffolding_v3.util.instantiate import Experiment

init_config()

TaskType = tuple[torch.Tensor, torch.Tensor]


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(cfg: Config) -> float:
    if cfg.resume:
        path = cfg.paths.output / cfg.resume
        exp = Experiment.from_path(path)
        cfg = exp.experiment_path.get_config() # type: ignore
    else:
        exp = Experiment.from_config(cfg)

    logger.debug(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.execution.seed)

    trainer = Trainer.from_experiment(exp, cfg)
    trainer.train_loop()
    if cfg.output.use_wandb:
        wandb.finish()
    return trainer.state.best_val_loss


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
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        generator: torch.Generator,
        experiment_path: ExperimentPath,
        checkpoint_manager: CheckpointManager,
        scheduler: Optional[LRScheduler],
        plotter: Optional[Plotter],
        initial_state: TrainerState,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.generator = generator
        self.experiment_path = experiment_path
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler
        self.plotter = plotter
        self.grad_scaler = GradScaler(enabled=cfg.execution.use_amp)

        self.state = initial_state

        self._init_wandb()
        self.metric_logger = WandbLogger(
            self.cfg.output.use_wandb,
            self.state,
        )
        self.profiler = WandbProfiler(self.metric_logger)

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
    def from_experiment(exp: Experiment, cfg: Config) -> "Trainer":
        return Trainer(
            cfg,
            exp.model,
            exp.optimizer,
            exp.train_loader,
            exp.val_loader,
            exp.generator,
            exp.experiment_path,
            exp.checkpoint_manager,
            exp.scheduler,
            exp.plotter if cfg.output.plot else None,
            exp.trainer_state,
        )

    def train_loop(self) -> None:
        s = self.state

        if s.epoch < self.cfg.execution.epochs:
            # Checkpoint is at end of epoch, add 1 for next epoch.
            s.epoch += 1
        else:
            logger.info("Run has concluded")
            return

        self._save_config()

        if self.cfg.output.use_wandb and self.cfg.output.log_gradients:
            wandb.watch(
                self.model,
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

            self._save_checkpoint("latest")

            if s.val_loss < s.best_val_loss:
                logger.success("New best val loss: {}", s.val_loss)
                s.best_val_loss = s.val_loss
                self._save_checkpoint("best")

            if self.scheduler:
                self.scheduler.step()

            if self.plotter:
                self.plotter.plot_prediction(self.model, s.epoch)

            s.epoch += 1
            s.best_val_loss = s.best_val_loss

        logger.success("Finished training")

    def _save_config(self) -> None:
        with open(self.experiment_path / "cfg.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    def train_epoch(self) -> None:
        self.model.train()
        device = next(self.model.parameters()).device
        train_loader: tqdm[TaskType] = tqdm(
            self.train_loader, disable=not self.cfg.output.use_tqdm
        )
        dry_run = self.cfg.execution.dry_run

        p = self.profiler
        for features, target in p.profiled_iter("dataload", train_loader):
            with p.profile("data.to"):
                features = features.to(device)
                target = target.to(device)

            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=self.cfg.execution.use_amp,
            ):
                with p.profile("forward"):
                    batch_loss = self.model(features, target)

            with p.profile("backward"):
                batch_loss.backward()
                self.metric_logger.log({"train_loss": batch_loss.item()})

            with p.profile("optimizer.step"):
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.state.step += 1

            if dry_run:
                break

    def _save_checkpoint(self, occasion: str) -> None:
        if self.cfg.output.save_checkpoints:
            self.checkpoint_manager.save_checkpoint(
                occasion,
                self.model,
                self.optimizer,
                self.generator,
                self.scheduler,
                self.state,
            )

    def val_epoch(self) -> dict[str, float]:
        with autocast(
            device_type=self.cfg.runtime.device,
            dtype=torch.float16,
            enabled=self.cfg.execution.use_amp,
        ):
            return evaluate(
                self.model,
                self.val_loader,
                self.cfg.execution.dry_run,
            )


if __name__ == "__main__":
    main()
