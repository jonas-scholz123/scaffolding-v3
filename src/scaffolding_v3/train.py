from typing import Optional
import sys
from loguru import logger
from dotenv import load_dotenv
import wandb
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
import hydra
import torch.optim.lr_scheduler
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import config_to_filepath
from mlbnb.rand import seed_everything
from mlbnb.metric_logger import MetricLogger

from config import Config, ExecutionConfig
from scaffolding_v3.config import SKIP_KEYS, CheckpointOccasion


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config", config_path="../..")
def main(cfg: Config):
    _configure_outputs(cfg)

    logger.info(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.execution.seed)

    logger.info("Instantiating dependencies")

    data_provider = hydra.utils.instantiate(cfg.data.data_provider)
    trainset = data_provider.get_train_data()
    valset = data_provider.get_val_data()

    train_loader = hydra.utils.instantiate(
        cfg.data.trainloader, trainset, generator=torch.default_generator
    )
    val_loader = hydra.utils.instantiate(
        cfg.data.testloader, valset, generator=torch.default_generator
    )

    model = hydra.utils.instantiate(cfg.model).to(cfg.execution.device)
    loss_fn = hydra.utils.instantiate(cfg.loss)

    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    scheduler = (
        hydra.utils.instantiate(cfg.scheduler, optimizer) if cfg.scheduler else None
    )

    path = config_to_filepath(cfg, cfg.output.out_dir, SKIP_KEYS)
    logger.info("Experiment path: {}", path)

    checkpoint_manager = CheckpointManager(path)

    logger.info("Finished instantiating dependencies")

    start_epoch = initial_setup(cfg, checkpoint_manager, model, optimizer, scheduler)

    train_loop(
        start_epoch,
        cfg,
        checkpoint_manager,
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
    )


def _configure_outputs(cfg: Config):
    load_dotenv()
    logger.configure(handlers=[{"sink": sys.stdout, "level": cfg.output.log_level}])

    if cfg.output.use_wandb:
        # Opt in to new wandb backend
        wandb.require("core")
        wandb.init(project="scaffolding_v3", config=dict(cfg), dir=cfg.output.out_dir)  # type: ignore


def initial_setup(
    cfg: Config,
    checkpoint_manager: CheckpointManager,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
) -> int:
    start_epoch = 1
    start_from = cfg.execution.start_from
    if start_from and checkpoint_manager.checkpoint_exists(start_from.value):
        checkpoint = checkpoint_manager.reproduce(
            start_from.value, model, optimizer, scheduler
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
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler],
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    logger.info("Starting training")

    metric_logger = MetricLogger(cfg.output.use_wandb)

    for epoch in range(start_epoch, cfg.execution.epochs + 1):
        logger.info("Starting epoch {} / {}", epoch, cfg.execution.epochs)
        train_metrics = train_step(
            cfg.execution, model, train_loader, optimizer, loss_fn
        )
        metric_logger.log(epoch, train_metrics)
        val_metrics = val_step(model, loss_fn, cfg.execution.device, val_loader)
        metric_logger.log(epoch, val_metrics)

        if cfg.output.save_model:
            checkpoint_manager.save_checkpoint(
                epoch,
                CheckpointOccasion.LATEST.value,
                model,
                optimizer,
                scheduler,
            )

        if scheduler:
            scheduler.step()

    logger.success("Finished training")


def train_step(
    train_cfg: ExecutionConfig,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
) -> dict[str, float]:
    train_loss = 0.0

    model.train()
    for data, target in train_loader:
        # TODO: get rid of this
        data, target = data.to(train_cfg.device), target.to(train_cfg.device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

        if train_cfg.dry_run:
            break

    train_len = len(train_loader.dataset)  # type: ignore
    return {"train_loss": train_loss / train_len}


def val_step(
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
    val_loader: DataLoader,
) -> dict[str, float]:
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_len = len(val_loader.dataset)  # type: ignore
    val_loss /= val_len
    return {"val_loss": val_loss, "val_accuracy": correct / val_len}


if __name__ == "__main__":
    main()
