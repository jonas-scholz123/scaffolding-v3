from typing import Optional
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.optim.adadelta import Adadelta
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LRScheduler, StepLR
import hydra
import torch.optim.lr_scheduler
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import config_to_filepath
from mlbnb.rand import seed_everything

from models.convnet import ConvNet
from config import Config, ExecutionConfig
from scaffolding_v3.config import SKIP_KEYS, CheckpointOccasion


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    logger.info(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.execution.seed)

    logger.info("Instantiating dependencies")
    normalization = hydra.utils.instantiate(cfg.data.normalization)
    transform = transforms.Compose([transforms.ToTensor(), normalization])

    trainset = hydra.utils.instantiate(cfg.data.trainset, transform=transform)
    testset = hydra.utils.instantiate(cfg.data.testset, transform=transform)
    train_loader = hydra.utils.instantiate(
        cfg.data.trainloader, trainset, generator=torch.default_generator
    )
    test_loader = hydra.utils.instantiate(
        cfg.data.testloader, testset, generator=torch.default_generator
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
        test_loader,
    )


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
    test_loader: DataLoader,
):
    logger.info("Starting training")

    for epoch in range(start_epoch, cfg.execution.epochs + 1):
        logger.info("Starting epoch {} / {}", epoch, cfg.execution.epochs)
        train_step(cfg.execution, model, train_loader, optimizer, loss_fn, epoch)
        test_step(model, loss_fn, cfg.execution.device, test_loader)

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
    train_cfg: ExecutionConfig, model, train_loader, optimizer, loss_fn, epoch
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(train_cfg.device), target.to(train_cfg.device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if train_cfg.dry_run:
            break


def test_step(model, loss_fn, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


if __name__ == "__main__":
    main()
