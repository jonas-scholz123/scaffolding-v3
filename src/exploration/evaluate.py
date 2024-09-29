# %%
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from deepsensor.train.train import set_gpu_default_device
from hydra.utils import instantiate
from loguru import logger
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from mlbnb.types import Split
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from scaffolding_v3.config import SKIP_KEYS, Config, Paths
from scaffolding_v3.data.dataset import make_dataset
from scaffolding_v3.data.dwd import get_data_processor
from scaffolding_v3.train import TrainerState, evaluate


def get_experiment_paths(root: Path, filter: Callable[[Config], bool] = lambda x: True):
    paths = root.rglob("*/checkpoints/best.pt")
    paths = [ExperimentPath(path.parent.parent) for path in paths]
    paths = [path for path in paths if filter(path.get_config())]
    logger.info("Found {} matching path(s)", len(paths))
    return paths


def era5_filter(cfg: Config) -> bool:
    return cfg.data.data_provider._target_.split(".")[-1] == "Era5DataProvider"  # type: ignore


paths = get_experiment_paths(
    Paths.output,
    filter=era5_filter,
)


# TODO: Split this up into smaller functions:
# 1. Load but don't yet evaluate the data
# 2. Sort by validation loss to evaluate the best models first
# 3. Evaluate the models


# TODO: Make sure this is deterministic, write tests
def make_eval_df(
    paths: list[ExperimentPath],
    initial_df: pd.DataFrame,
) -> pd.DataFrame:
    dfs = [initial_df]

    for path in tqdm(paths):
        checkpoint_manager = CheckpointManager(path)
        config: Config = path.get_config()  # type: ignore

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

        checkpoint = checkpoint_manager.load_checkpoint("best")
        if checkpoint.other_state is None:
            logger.warning("No other state found in checkpoint {}", path)
            continue
        trainer_state = TrainerState.from_dict(checkpoint.other_state)

        path_str = str(path.root)
        matching_rows = initial_df.query(
            "path == @path_str and epoch == @trainer_state.epoch and evaluated"
        )

        already_evaluated = not matching_rows.empty
        if already_evaluated:
            continue

        df["path"] = path.root
        df["epoch"] = trainer_state.epoch
        df["val_loss"] = trainer_state.best_val_loss
        df = df.set_index("path")
        dfs.append(df)
    df = pd.concat(dfs)
    df["evaluated"] = df["evaluated"].astype(bool)
    df = df.sort_values("val_loss", ascending=True)

    for path_str in df[~df["evaluated"]].index:
        path = ExperimentPath(Path(path_str))  # type: ignore
        config: Config = path.get_config()  # type: ignore

        for metric_name, metric in compute_test_metrics(config).items():
            df.loc[path_str, f"test_{metric_name}"] = metric
        df.loc[path_str, "evaluated"] = True

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


def evaluate_all() -> pd.DataFrame:
    df = make_eval_df(paths, load_eval_df())
    df = df.reset_index()
    df.to_csv(Paths.evaluation, index=False)
    return df


def load_eval_df() -> pd.DataFrame:
    if not Paths.evaluation.exists():
        # Include these columns to simplify the code later
        df = pd.DataFrame(columns=["evaluated", "path", "epoch"])  # type: ignore
    else:
        df = pd.read_csv(Paths.evaluation)
    return df.set_index("path")


evaluate_all()
