from typing import Optional
import warnings
import sys
from loguru import logger
from dotenv import load_dotenv
from tqdm import tqdm
import wandb
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import hydra
import torch.optim.lr_scheduler
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from deepsensor.model.convnp import ConvNP
from deepsensor.train.train import set_gpu_default_device
import deepsensor.torch
from deepsensor import Task

from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import config_to_filepath
from mlbnb.rand import seed_everything
from mlbnb.metric_logger import MetricLogger

from scaffolding_v3.config import (
    SKIP_KEYS,
    CheckpointOccasion,
    Config,
    ExecutionConfig,
    load_config,
)

load_config()

@hydra.main(version_base=None, config_name="dev", config_path="../..")
def main(cfg: Config):
    try:
        _configure_outputs(cfg)

        logger.info(OmegaConf.to_yaml(cfg))

        if cfg.execution.device == "cuda":
            set_gpu_default_device()

        seed_everything(cfg.execution.seed)
        generator = torch.Generator(device=cfg.execution.device).manual_seed(
            cfg.execution.seed
        )

        logger.info("Instantiating dependencies")

        data_provider = hydra.utils.instantiate(cfg.data.data_provider)
        trainset = data_provider.get_train_data()
        valset = data_provider.get_val_data()
        collate_fn = data_provider.get_collate_fn()

        train_loader = hydra.utils.instantiate(
            cfg.data.trainloader,
            trainset,
            generator=generator,
            collate_fn=collate_fn,
        )
        val_loader = hydra.utils.instantiate(
            cfg.data.testloader,
            valset,
            generator=generator,
            collate_fn=collate_fn,
        )

        model = hydra.utils.instantiate(
            cfg.model, trainset.data_processor, trainset.task_loader
        )

        optimizer = hydra.utils.instantiate(cfg.optimizer, model.model.parameters())

        scheduler = (
            hydra.utils.instantiate(cfg.scheduler, optimizer) if cfg.scheduler else None
        )

        path = config_to_filepath(cfg, cfg.output.out_dir, SKIP_KEYS)
        logger.info("Experiment path: {}", path)
        checkpoint_manager = CheckpointManager(path)

        logger.info("Finished instantiating dependencies")

        start_epoch = initial_setup(
            cfg, checkpoint_manager, model, optimizer, generator, scheduler
        )

        train_loop(
            start_epoch,
            cfg,
            checkpoint_manager,
            model,
            optimizer,
            train_loader,
            val_loader,
            generator,
            scheduler,
        )
    except Exception as e:
        logger.exception("An error occurred during training: {}", e)
        raise e


def _configure_outputs(cfg: Config):
    load_dotenv()
    logger.configure(handlers=[{"sink": sys.stdout, "level": cfg.output.log_level}])

    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="google.protobuf"
    )

    if cfg.output.use_wandb:
        # Opt in to new wandb backend
        wandb.require("core")
        wandb.init(project="scaffolding_v3", config=OmegaConf.to_container(cfg), dir=cfg.output.out_dir)  # type: ignore


def initial_setup(
    cfg: Config,
    checkpoint_manager: CheckpointManager,
    model: ConvNP,
    optimizer: Optimizer,
    generator: torch.Generator,
    scheduler: Optional[LRScheduler],
) -> int:
    start_epoch = 1
    start_from = cfg.execution.start_from
    if start_from and checkpoint_manager.checkpoint_exists(start_from.value):
        checkpoint = checkpoint_manager.reproduce(
            start_from.value, model.model, optimizer, generator, scheduler
        )

        # Checkpoint is at end of epoch, add 1 for next epoch.
        start_epoch = checkpoint.epoch + 1
        if start_epoch < cfg.execution.epochs:
            logger.info("Checkpoint loaded, starting from epoch {}", start_epoch)
        else:
            logger.info(
                "Checkpoint loaded, run has concluded (epoch {} / {})",
                start_epoch - 1,
                cfg.execution.epochs,
            )
    return start_epoch


def train_loop(
    start_epoch: int,
    cfg: Config,
    checkpoint_manager: CheckpointManager,
    model: ConvNP,
    optimizer: Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    generator: torch.Generator,
    scheduler: Optional[LRScheduler],
):
    logger.info("Starting training")

    metric_logger = MetricLogger(cfg.output.use_wandb)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.execution.epochs + 1):
        logger.info("Starting epoch {} / {}", epoch, cfg.execution.epochs)
        train_metrics = train_step(cfg.execution, model, train_loader, optimizer)
        metric_logger.log(epoch, train_metrics)
        val_metrics = val_step(cfg.execution, model, val_loader)
        metric_logger.log(epoch, val_metrics)

        if cfg.output.save_model:
            checkpoint_manager.save_checkpoint(
                epoch,
                CheckpointOccasion.LATEST.value,
                model.model,
                optimizer,
                generator,
                scheduler,
            )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            checkpoint_manager.save_checkpoint(
                epoch,
                CheckpointOccasion.BEST.value,
                model.model,
                optimizer,
                generator,
                scheduler,
            )

        if scheduler:
            scheduler.step()

    logger.success("Finished training")


def train_step(
    train_cfg: ExecutionConfig,
    model: ConvNP,
    train_loader: DataLoader,
    optimizer: Optimizer,
) -> dict[str, float]:
    train_loss = 0.0

    model.model.train()
    for batch in tqdm(train_loader):
        if train_cfg.dry_run:
            batch = batch[:1]

        task_losses = []
        optimizer.zero_grad()

        for task in batch:
            task_losses.append(model.loss_fn(task, normalise=True))

        batch_loss = torch.mean(torch.stack(task_losses))
        batch_loss.backward()
        optimizer.step()

        train_loss += float(batch_loss.detach().cpu().numpy())

        if train_cfg.dry_run:
            break

    return {"train_loss": train_loss / len(train_loader)}


def val_step(
    cfg: ExecutionConfig,
    model: ConvNP,
    val_loader: DataLoader,
) -> dict[str, float]:
    model.model.eval()
    batch_losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader):

            if cfg.dry_run:
                batch = batch[:1]

            batch_losses.append(eval_on_batch(model, batch))

            if cfg.dry_run:
                break

    val_loss = float(np.mean(batch_losses))
    return {"val_loss": val_loss}


def eval_on_batch(model: ConvNP, batch: list[Task]) -> float:
    with torch.no_grad():
        task_losses = []
        for task in batch:
            task_losses.append(model.loss_fn(task, normalise=True))
        mean_batch_loss = torch.mean(torch.stack(task_losses))

    return float(mean_batch_loss.detach().cpu().numpy())


if __name__ == "__main__":
    main()
