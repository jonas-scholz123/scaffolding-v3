import sys

import deepsensor.torch  # noqa
import numpy as np
import pandas as pd
import torch
from deepsensor import Task
from deepsensor.model.convnp import ConvNP
from deepsensor.train.train import set_gpu_default_device
from hydra.utils import instantiate
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.paths import ExperimentPath, get_experiment_paths
from mlbnb.types import Split
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from scaffolding_v3.config import SKIP_KEYS, Config, Paths
from scaffolding_v3.data.dataset import make_dataset
from scaffolding_v3.data.dwd import get_data_processor

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])


def evaluate_all() -> pd.DataFrame:
    df = make_eval_df(paths, load_eval_df())
    df = evaluate_remaining(df)
    save_df(df)
    return df


def load_eval_df() -> pd.DataFrame:
    if not Paths.evaluation.exists():
        # Include these columns to simplify the code later
        df = pd.DataFrame(columns=["evaluated", "path", "epoch"])  # type: ignore
    else:
        df = pd.read_csv(Paths.evaluation)
    return df.set_index("path")


def make_eval_df(
    paths: list[ExperimentPath],
    initial_df: pd.DataFrame,
) -> pd.DataFrame:
    dfs = [initial_df]

    for path in tqdm(paths):
        config: Config = path.get_config()  # type: ignore

        df = config_to_df(config)

        trainer_state = get_trainer_state(path)

        if is_already_evaluated(initial_df, path, trainer_state.epoch):
            continue

        df["path"] = path.root
        df["epoch"] = trainer_state.epoch
        df["val_loss"] = trainer_state.best_val_loss
        df = df.set_index("path")
        dfs.append(df)
    df = pd.concat(dfs)
    df["evaluated"] = df["evaluated"].astype(bool)
    df = df.sort_values("val_loss", ascending=True)

    return df


def evaluate_remaining(df: pd.DataFrame) -> pd.DataFrame:
    for path_str in tqdm(df[~df["evaluated"]].index):
        path = ExperimentPath(path_str)  # type: ignore
        config: Config = path.get_config()  # type: ignore

        for metric_name, metric in compute_test_metrics(config).items():
            df.loc[path_str, f"test_{metric_name}"] = metric
        df.loc[path_str, "evaluated"] = True
        save_df(df)
    return df


def compute_test_metrics(cfg: Config) -> dict[str, float]:
    if cfg.execution.device == "cuda":
        set_gpu_default_device()

    generator = torch.Generator(device=cfg.execution.device).manual_seed(
        cfg.execution.seed
    )
    data_processor = get_data_processor(cfg.paths)
    data_provider = instantiate(cfg.data.data_provider)
    testset = make_dataset(
        cfg.data, cfg.paths, data_provider, Split.TEST, data_processor
    )

    test_loader: DataLoader = instantiate(
        cfg.data.testloader,
        testset,
        generator=generator,
        collate_fn=lambda x: x,
    )

    model = instantiate(cfg.model, data_processor, testset.task_loader)

    return evaluate(model, test_loader, False)


def evaluate(model: ConvNP, dataloader: DataLoader, dry_run: bool) -> dict[str, float]:
    model.model.eval()
    batch_losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if dry_run:
                batch = batch[:1]

            batch_losses.append(eval_on_batch(model, batch))

            if dry_run:
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


def config_to_df(config: Config) -> pd.DataFrame:
    config_dict = OmegaConf.to_container(config, resolve=True)
    flat_dict = pd.json_normalize(config_dict)  # type: ignore
    df = pd.DataFrame(flat_dict)
    df["evaluated"] = False
    skip_cols = []
    for col in df.columns:
        for keys in col.split("."):
            if keys in SKIP_KEYS:
                skip_cols.append(col)
                break
    df = df.drop(columns=skip_cols)
    return df


def get_trainer_state(path: ExperimentPath) -> TrainerState:
    checkpoint_manager = CheckpointManager(path)
    checkpoint = checkpoint_manager.load_checkpoint("best")
    if checkpoint.other_state is None:
        raise ValueError(f"No other state found in checkpoint at path {path}")
    return TrainerState.from_dict(checkpoint.other_state)


def is_already_evaluated(
    initial_df: pd.DataFrame, path: ExperimentPath, epoch: int
) -> bool:
    path_str = str(path)  # noqa
    matching_rows = initial_df.query(
        "path == @path_str and epoch == @epoch and evaluated"
    )

    return not matching_rows.empty


def save_df(df: pd.DataFrame) -> None:
    df = df.reset_index()
    df.to_csv(Paths.evaluation, index=False)


def era5_filter(cfg: Config) -> bool:
    return cfg.data.data_provider._target_.split(".")[-1] == "Era5DataProvider"  # type: ignore


def drop_boring_cols(df: pd.DataFrame):
    """Drop columns that are the same for all experiments"""
    droppable = [col for col in df.columns if len(df[col].unique()) == 1]
    df = df.drop(columns=droppable)
    return df


paths = get_experiment_paths(
    Paths.output,
    filter=era5_filter,
)

if __name__ == "__main__":
    df = evaluate_all()
    print(drop_boring_cols(df))
