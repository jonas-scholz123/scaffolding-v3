from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.adadelta import Adadelta
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler

from models.convnet import ConvNet
from config import Config, ExecutionConfig


def train(train_cfg: ExecutionConfig, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(train_cfg.device), target.to(train_cfg.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if train_cfg.dry_run:
            break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    logger.info(OmegaConf.to_yaml(cfg))

    logger.info("Instantiating dependencies")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    trainset = hydra.utils.instantiate(cfg.data.trainset, transform=transform)
    testset = hydra.utils.instantiate(cfg.data.testset, transform=transform)
    train_loader = hydra.utils.instantiate(cfg.data.trainloader, trainset)
    test_loader = hydra.utils.instantiate(cfg.data.testloader, testset)

    model = hydra.utils.instantiate(cfg.model).to(cfg.execution.device)

    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    scheduler = (
        hydra.utils.instantiate(cfg.scheduler, optimizer) if cfg.scheduler else None
    )

    logger.info("Finished instantiating dependencies")

    logger.info("Starting training")
    for epoch in range(1, cfg.execution.epochs + 1):
        logger.info(f"Starting epoch {epoch}")
        train(cfg.execution, model, train_loader, optimizer, epoch)
        test(model, cfg.execution.device, test_loader)
        if scheduler:
            scheduler.step()
        logger.info(f"Finished epoch {epoch}")

    logger.info("Finished training")
    if cfg.output.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    logger.info("Saved model to mnist_cnn.pt")


if __name__ == "__main__":
    main()
