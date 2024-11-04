import sys
import warnings
from typing import Optional

import deepsensor.torch  # noqa
import hydra
import numpy as np
import torch
import torch.optim.lr_scheduler
import wandb
from deepsensor.model.convnp import ConvNP
from deepsensor.train.train import set_gpu_default_device
from dotenv import load_dotenv
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.metric_logger import MetricLogger
from mlbnb.paths import ExperimentPath
from mlbnb.rand import seed_everything
from mlbnb.types import Split
from omegaconf import OmegaConf
from sqlalchemy.exc import MovedIn20Warning
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
from scaffolding_v3.data.dataprovider import DataProvider, DeepSensorDataset
from scaffolding_v3.data.dataset import make_dataset
from scaffolding_v3.data.dwd import get_data_processor
from scaffolding_v3.evaluate import evaluate
from scaffolding_v3.plot.plotter import Plotter

load_config()


@hydra.main(version_base=None, config_name="dev", config_path="")
def main(cfg: Config):
    try:
        _configure_outputs(cfg)

        logger.debug(OmegaConf.to_yaml(cfg))

        if cfg.execution.device == "cuda":
            set_gpu_default_device()

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
    logger.configure(handlers=[{"sink": sys.stdout, "level": cfg.output.log_level}])

    # These are all due to deprecation warnings raised within dependencies.
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="google.protobuf"
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="deepsensor")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lab")
    warnings.filterwarnings("ignore", category=MovedIn20Warning)

    if cfg.output.use_wandb:
        wandb.init(
            project="scaffolding_v3",
            config=OmegaConf.to_container(cfg),  # type: ignore
            dir=cfg.output.out_dir,
        )  # type: ignore


class Trainer:
    cfg: Config
    model: ConvNP
    optimizer: Optimizer
    train_loader: DataLoader
    val_loader: DataLoader
    test_data: DeepSensorDataset
    generator: torch.Generator
    experiment_path: ExperimentPath
    checkpoint_manager: CheckpointManager
    scheduler: Optional[LRScheduler]
    plotter: Optional[Plotter]
    state: TrainerState

    def __init__(
        self,
        cfg: Config,
        model: ConvNP,
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

        if self.cfg.execution.use_pretrained:
            pretrained_model_path = self.cfg.paths.pretrained_model_path
            try:
                checkpoint = CheckpointManager.load_checkpoint_from_path(
                    pretrained_model_path
                )
                self.model.model.load_state_dict(checkpoint.model_state)
                logger.info(
                    "Pretrained model loaded from path {}, starting from pretrained.",
                    pretrained_model_path,
                )
            except FileNotFoundError:
                logger.warning(
                    "Pretrained model path {} does not exist, starting from scratch.",
                    pretrained_model_path,
                )
            except Exception as e:
                logger.warning(
                    "Error loading pretrained model from path {}, starting from scratch: {}",
                    pretrained_model_path,
                    e,
                )

        if start_from and self.checkpoint_manager.checkpoint_exists(start_from.value):
            self.checkpoint_manager.reproduce(
                start_from.value,
                self.model.model,
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

        logger.info("Instantiating dependencies")

        data_processor = get_data_processor(cfg.paths)

        # Create the primary data source (ERA5/DWD)
        data_provider: DataProvider = hydra.utils.instantiate(cfg.data.data_provider)

        trainset = make_dataset(
            cfg.data, cfg.paths, data_provider, Split.TRAIN, data_processor
        )
        valset = make_dataset(
            cfg.data, cfg.paths, data_provider, Split.VAL, data_processor
        )

        train_loader: DataLoader = hydra.utils.instantiate(
            cfg.data.trainloader,
            trainset,
            generator=generator,
            collate_fn=lambda x: x,
        )
        val_loader: DataLoader = hydra.utils.instantiate(
            cfg.data.testloader,
            valset,
            generator=generator,
            collate_fn=lambda x: x,
        )

        model = hydra.utils.instantiate(cfg.model, data_processor, trainset.task_loader)

        optimizer = hydra.utils.instantiate(cfg.optimizer, model.model.parameters())

        scheduler = (
            hydra.utils.instantiate(cfg.scheduler, optimizer) if cfg.scheduler else None
        )

        experiment_path = ExperimentPath.from_config(cfg, cfg.paths.output, SKIP_KEYS)

        logger.info("Experiment path: {}", str(experiment_path))
        checkpoint_manager = CheckpointManager(experiment_path)

        test_data = data_provider.get_test_data()

        if cfg.output.plot:
            plotter = Plotter(cfg, data_processor, test_data, experiment_path)
        else:
            plotter = None

        logger.info("Finished instantiating dependencies")

        return Trainer(
            cfg,
            model,
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

        logger.info("Starting training")

        metric_logger = MetricLogger(self.cfg.output.use_wandb)

        if self.plotter:
            self.plotter.plot_task(self.train_loader.dataset[0])
            self.plotter.plot_context_encoding(self.model, self.train_loader.dataset[0])

        while s.epoch <= self.cfg.execution.epochs + 1:
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

        self.model.model.train()
        for batch in tqdm(self.train_loader):
            if self.cfg.execution.dry_run:
                batch = batch[:1]

            task_losses = []
            self.optimizer.zero_grad()

            for task in batch:
                task_losses.append(self.model.loss_fn(task, normalise=True))

            batch_loss = torch.mean(torch.stack(task_losses))
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
                self.model.model,
                self.optimizer,
                self.generator,
                self.scheduler,
                self.state,
            )

    def val_step(self) -> dict[str, float]:
        return evaluate(self.model, self.val_loader, self.cfg.execution.dry_run)


if __name__ == "__main__":
    main()
