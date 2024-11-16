import sys
from pathlib import Path
from typing import Iterable

import deepsensor.torch  # noqa
import hydra
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

from scaffolding_v3.config import SKIP_KEYS, Config, Paths, load_config
from scaffolding_v3.data.dataset import make_dataset
from scaffolding_v3.data.dwd import get_data_processor

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])

load_config()


@hydra.main(version_base=None, config_name="dev", config_path="")
def main(eval_cfg: Config) -> None:
    df = evaluate_all(eval_cfg)
    print(drop_boring_cols(df))


def evaluate_all(eval_cfg: Config) -> pd.DataFrame:
    # If dry run requested, only evaluate dry run experiments and vice versa
    def dry_run_filter(cfg: Config) -> bool:
        return cfg.execution.dry_run == eval_cfg.execution.dry_run

    logger.info("Initializing evaluation dataframe")
    paths = get_experiment_paths(Paths.output, dry_run_filter)
    df = make_eval_df(paths, load_eval_df(_extract_data_provider_name(eval_cfg)))
    logger.info("Evaluate unevaluated experiments")
    df = evaluate_remaining(df, eval_cfg)
    return df


def load_eval_df(data_provider_name: str) -> pd.DataFrame:
    path = _get_csv_fpath(data_provider_name)
    if not path.exists():
        # Include these columns to simplify the code later
        df = pd.DataFrame(columns=["evaluated", "path", "epoch"])  # type: ignore
    else:
        df = pd.read_csv(path)
    return df.set_index("path")


def _get_csv_fpath(data_provider_name: str) -> Path:
    return Paths.output / f"evaluation_{data_provider_name}.csv"


def _extract_data_provider_name(cfg: Config) -> str:
    return cfg.data.data_provider._target_.split(".")[-1]  # type: ignore


def make_eval_df(
    paths: list[ExperimentPath],
    initial_df: pd.DataFrame,
) -> pd.DataFrame:
    dfs = [initial_df]

    for path in tqdm(paths):
        experiment_cfg: Config = path.get_config()  # type: ignore

        df = config_to_df(experiment_cfg)

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


def evaluate_remaining(df: pd.DataFrame, eval_cfg: Config) -> pd.DataFrame:
    if df[~df["evaluated"]].empty:
        logger.info("All experiments have been evaluated")
        return df

    if eval_cfg.execution.device == "cuda":
        set_gpu_default_device()

    generator = torch.Generator(device=eval_cfg.execution.device).manual_seed(
        eval_cfg.execution.seed
    )
    data_processor = get_data_processor(eval_cfg.paths)
    data_provider = instantiate(eval_cfg.data.data_provider)
    testset = make_dataset(
        eval_cfg.data, eval_cfg.paths, data_provider, Split.TEST, data_processor
    )

    test_loader: DataLoader = instantiate(
        eval_cfg.data.testloader,
        testset,
        generator=generator,
        collate_fn=lambda x: x,
    )

    # Load all data into memory
    eval_data = list(test_loader)

    for path_str in tqdm(df[~df["evaluated"]].index):
        logger.info(f"Evaluating {path_str}")
        path = ExperimentPath(path_str)  # type: ignore
        experiment_cfg: Config = path.get_config()  # type: ignore
        checkpoint_manager = CheckpointManager(path)
        model: ConvNP = instantiate(
            experiment_cfg.model, data_processor, testset.task_loader
        )
        checkpoint_manager.reproduce_model(model.model, "best")

        test_metrics = evaluate(model, eval_data, False)

        for metric_name, metric in test_metrics.items():
            df.loc[path_str, f"test_{metric_name}"] = metric
        df.loc[path_str, "evaluated"] = True
        if not eval_cfg.execution.dry_run:
            save_df(df, _extract_data_provider_name(eval_cfg))
    return df


def evaluate(model: ConvNP, eval_data: Iterable, dry_run: bool) -> dict[str, float]:
    model.model.eval()
    batch_losses = []
    with torch.no_grad():
        for batch in tqdm(eval_data):
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


def save_df(df: pd.DataFrame, data_name: str) -> None:
    df = df.reset_index()
    path = _get_csv_fpath(data_name)
    df.to_csv(path, index=False)


def era5_filter(cfg: Config) -> bool:
    return cfg.data.data_provider._target_.split(".")[-1] == "DwdDataProvider"  # type: ignore


def drop_boring_cols(df: pd.DataFrame):
    """Drop columns that are the same for all experiments"""
    droppable = [col for col in df.columns if len(df[col].unique()) == 1]
    droppable += ["data.data_provider.train_range", "data.data_provider.test_range"]
    logger.warning(f"Dropping columns {droppable}")
    df = df.drop(columns=droppable)
    return df


if __name__ == "__main__":
    main()
