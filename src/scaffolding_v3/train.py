import math
import sys
from typing import Optional

import torch
import torch.multiprocessing as mp
import torch.optim.lr_scheduler
import wandb
from hydra import compose, initialize
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.iter import StepIterator
from mlbnb.metric_logger import WandbLogger
from mlbnb.paths import ExperimentPath
from mlbnb.profiler import WandbProfiler
from mlbnb.rand import seed_everything
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.distributed import destroy_process_group  # type: ignore
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from scaffolding_v3.config import Config, init_config
from scaffolding_v3.evaluate import evaluate
from scaffolding_v3.model.classification import ClassificationModule
from scaffolding_v3.plot.plotter import Plotter
from scaffolding_v3.util.ddp import ddp_setup
from scaffolding_v3.util.instantiate import Experiment

init_config()

TaskType = tuple[torch.Tensor, torch.Tensor]


def main(gpu_id: int | None, world_size: int) -> float:
    overrides = sys.argv[1:]

    # Initialise hydra:
    with initialize(config_path="../config", version_base=None):
        cfg: Config = compose(config_name="base", overrides=overrides)  # type: ignore

    multigpu = gpu_id is not None
    try:
        if multigpu:
            logger.info(f"Using multi-GPU training on {world_size} GPUs")
            ddp_setup(gpu_id, world_size)  # type: ignore

            # Only save from one process during multi-GPU training
            cfg.output.save_checkpoints &= gpu_id == 0
            cfg.output.use_tqdm &= gpu_id == 0
            cfg.output.plot &= gpu_id == 0
            cfg.output.use_wandb &= gpu_id == 0
        else:
            logger.info("Using single-device training")

        return train_loop(cfg, gpu_id)
    except Exception as e:
        raise e
    finally:
        if multigpu:
            destroy_process_group()
        if cfg.output.use_wandb:
            wandb.finish()


def train_loop(cfg: Config, gpu_id: int | None = None) -> float:
    if cfg.resume:
        path = cfg.paths.output / cfg.resume
        path = ExperimentPath.from_path(path)
        cfg = path.get_config()  # type: ignore

    exp = Experiment.from_config(cfg, gpu_id=gpu_id)

    logger.debug(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.execution.seed)

    trainer = Trainer.from_experiment(exp, cfg)
    trainer.train_loop()
    return trainer.state.best_val_loss


class Trainer:
    cfg: Config
    model: ClassificationModule
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
        model: ClassificationModule,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None,
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
        self.device = torch.device(cfg.runtime.device)

        self.state = initial_state
        if test_loader is None:
            logger.warning(
                "No test loader provided, using validation loader for final eval"
            )
            self.test_loader = val_loader
        else:
            self.test_loader = test_loader

        self._init_wandb()
        self.metric_logger = WandbLogger(
            self.cfg.output.use_wandb,
            self.state,
        )
        self.profiler = WandbProfiler(self.metric_logger)

    def _init_wandb(self) -> None:
        if self.cfg.output.use_wandb:
            run = wandb.init(
                project=self.cfg.output.wandb_project,
                config=OmegaConf.to_container(self.cfg),  # type: ignore
                dir=self.cfg.output.out_dir,
                name=self.experiment_path.name,
                id=self.experiment_path.name,
            )
            # For all metrics, use samples seen as the x-axis.
            run.define_metric("*", step_metric="samples_seen")

    @staticmethod
    def from_experiment(exp: Experiment, cfg: Config) -> "Trainer":
        return Trainer(
            cfg,
            exp.model,
            exp.optimizer,
            exp.train_loader,
            exp.val_loader,
            exp.test_loader,
            exp.generator,
            exp.experiment_path,
            exp.checkpoint_manager,
            exp.scheduler,
            exp.plotter if cfg.output.plot else None,
            exp.trainer_state,
        )

    def train_loop(self) -> None:
        if self.state.samples_seen >= self.cfg.execution.num_train_samples:
            logger.info("Run has concluded")
            return

        self._save_config()

        if self.cfg.output.use_wandb and self.cfg.output.log_gradients:
            wandb.watch(
                self.model,
                log="all",
                log_freq=self.cfg.output.gradient_log_frequency,
            )

        logger.info("Starting training")

        step_iterator = StepIterator(
            self.train_loader,
            steps=self.cfg.execution.num_train_samples,
            step_per_iter=self.cfg.data.trainloader.batch_size,
        )

        train_iter = tqdm(
            step_iterator,
            disable=not self.cfg.output.use_tqdm,
            unit="samples",
            unit_scale=self.cfg.data.trainloader.batch_size,
        )

        self.model.train()
        dry_run = self.cfg.execution.dry_run

        for batch in self.profiler.profiled_iter("dataload", train_iter):
            self._train_step(batch)

            if self._is_due(self.cfg.output.val_frequency):
                self._validation_step()

            if self._is_due(self.cfg.output.checkpoint_frequency):
                self._save_checkpoint("latest")

            if self.plotter and self._is_due(self.cfg.output.plot_frequency):
                self.plotter.plot_prediction(self.model, self.state.samples_seen)

            if dry_run:
                break

            self.state.step += 1

        logger.success("Finished training. Starting final evaluation...")
        self._final_eval()

    def _train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        p = self.profiler

        features, target = batch
        with p.profile("data.to"):
            features = features.to(self.device)
            target = target.to(self.device)

        with autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.cfg.execution.use_amp,
        ):
            with p.profile("forward"):
                batch_loss = self.model(features, target)

        with p.profile("backward"):
            batch_loss.backward()
            self.metric_logger.log({"loss": batch_loss.item()}, prefix="train")

        with p.profile("optimizer.step"):
            self.optimizer.step()
            self.optimizer.zero_grad()

        with p.profile("scheduler.step"):
            if self.scheduler:
                self.scheduler.step()

        self.state.samples_seen += features.size(0)

    def _is_due(self, frequency: int) -> bool:
        step_frequency = math.ceil(frequency / self.cfg.data.trainloader.batch_size)
        return self.state.step % step_frequency == 0

    def _validation_step(self) -> None:
        s = self.state

        self.model.eval()
        val_metrics = self.val_epoch()
        s.val_loss = val_metrics["loss"]
        self.metric_logger.log(val_metrics, prefix="val")

        if s.val_loss < s.best_val_loss:
            logger.success("New best val loss: {}", s.val_loss)
            s.best_val_loss = s.val_loss
            self._save_checkpoint("best")
        self.model.train()

    def _save_config(self) -> None:
        with open(self.experiment_path / "cfg.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

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

    def _final_eval(self) -> None:
        self.model.eval()
        final_metrics = self.val_epoch(final=True)
        self.metric_logger.log(final_metrics, prefix="test")
        logger.info("Final evaluation metrics: {}", final_metrics)

    def val_epoch(self, final: bool = False) -> dict[str, float]:
        with autocast(
            device_type=self.cfg.runtime.device,
            dtype=torch.float16,
            enabled=self.cfg.execution.use_amp,
        ):
            if final:
                dataloader = self.test_loader
            else:
                dataloader = self.val_loader

            result = evaluate(
                self.model,
                dataloader,
                self.cfg.execution.dry_run,
            )
            return result


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        main(gpu_id=None, world_size=1)
    else:
        mp.spawn(main, args=(world_size,), nprocs=world_size)
