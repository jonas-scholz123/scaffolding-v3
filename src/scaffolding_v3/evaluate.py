import sys
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.paths import ExperimentPath, get_experiment_paths
from mlbnb.types import Split
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from config.config import SKIP_KEYS, Config, Paths, init_configs
from scaffolding_v3.data.data import make_dataset
from scaffolding_v3.util.config_filter import DryRunFilter

logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])

init_configs()


@hydra.main(version_base=None, config_name="base", config_path="../config")
def main(eval_cfg: Config) -> None:
    if eval_cfg.runtime.device == "cuda":
        torch.set_default_device("cuda")

    df = evaluate_all(eval_cfg)
    print(drop_boring_cols(df))


def evaluate_all(eval_cfg: Config) -> pd.DataFrame:
    logger.info("Initializing evaluation dataframe")
    # If dry run requested, only evaluate dry run experiments and vice versa
    paths = get_experiment_paths(
        eval_cfg.paths.output, DryRunFilter(eval_cfg.execution.dry_run)
    )

    eval_df = load_eval_df(eval_cfg.paths, _extract_data_provider_name(eval_cfg))
    df = make_eval_df(paths, eval_df)
    if df.empty:
        logger.warning("No experiments to evaluate, exiting")
        return df
    logger.info("Evaluate unevaluated experiments")

    df = evaluate_remaining(df, eval_cfg)
    return df


def load_eval_df(paths: Paths, data_provider_name: str) -> pd.DataFrame:
    path = _get_csv_fpath(paths, data_provider_name)
    if not path.exists():
        # Include these columns to simplify the code later
        df = pd.DataFrame(columns=["evaluated", "path", "epoch"])  # type: ignore
    else:
        df = pd.read_csv(path)
    return df.set_index("path")


def _get_csv_fpath(paths: Paths, data_provider_name: str) -> Path:
    return paths.output / f"evaluation_{data_provider_name}.csv"


def _extract_data_provider_name(cfg: Config) -> str:
    return cfg.data.dataset._target_.split(".")[-1]  # type: ignore


def make_eval_df(
    paths: list[ExperimentPath],
    initial_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(paths) == 0:
        return initial_df
    dfs = [initial_df]

    for path in tqdm(paths):
        experiment_cfg: Config = path.get_config()  # type: ignore

        df = config_to_df(experiment_cfg)

        trainer_state = get_trainer_state(path)

        if is_already_evaluated(initial_df, path, trainer_state.epoch):
            continue

        df["path"] = path.name
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

    device = eval_cfg.runtime.device
    generator = torch.Generator(device=device).manual_seed(eval_cfg.execution.seed)
    testset = make_dataset(eval_cfg.data, Split.TEST, generator)

    test_loader: DataLoader = instantiate(
        eval_cfg.data.testloader,
        testset,
        generator=generator,
    )

    for path_str in tqdm(df[~df["evaluated"]].index):
        logger.info(f"Evaluating {path_str}")
        path = ExperimentPath(eval_cfg.paths.output, path_str)
        experiment_cfg: Config = path.get_config()  # type: ignore
        checkpoint_manager = CheckpointManager(path)

        in_channels = experiment_cfg.data.in_channels
        num_classes = experiment_cfg.data.num_classes
        sidelength = experiment_cfg.data.sidelength

        model: nn.Module = instantiate(
            experiment_cfg.model,
            in_channels=in_channels,
            num_classes=num_classes,
            sidelength=sidelength,
        ).to(device)
        loss_fn: nn.Module = instantiate(experiment_cfg.loss).to(device)
        checkpoint_manager.reproduce_model(model, "best")

        test_metrics = evaluate(model, loss_fn, test_loader, False)

        for metric_name, metric in test_metrics.items():
            df.loc[path_str, f"test_{metric_name}"] = metric
        df.loc[path_str, "evaluated"] = True
        if not eval_cfg.execution.dry_run:
            save_df(df, _extract_data_provider_name(eval_cfg), eval_cfg.paths)
    return df


def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    val_loader: DataLoader,
    dry_run: bool = False,
) -> dict[str, float]:
    model.eval()
    val_loss = 0
    correct = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if dry_run:
                break

    val_len = len(val_loader.dataset)  # type: ignore
    val_loss /= val_len
    return {"val_loss": val_loss, "val_accuracy": correct / val_len}


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


def save_df(df: pd.DataFrame, data_name: str, paths: Paths) -> None:
    df = df.reset_index()
    path = _get_csv_fpath(paths, data_name)
    df.to_csv(path, index=False)


def drop_boring_cols(df: pd.DataFrame):
    """Drop columns that are the same for all experiments"""
    # As strings to deal with unhahsable types.
    droppable = [col for col in df.columns if len(df[col].astype(str).unique()) == 1]
    df = df.drop(columns=droppable)
    return df


if __name__ == "__main__":
    main()
