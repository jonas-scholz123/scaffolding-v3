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


def make_eval_df(paths: list[ExperimentPath]) -> pd.DataFrame:
    dfs = []
    for path in tqdm(paths):
        checkpoint_manager = CheckpointManager(path)
        config: Config = path.get_config()  # type: ignore

        config_dict = OmegaConf.to_container(config, resolve=True)
        flat_dict = pd.json_normalize(config_dict)  # type: ignore
        df = pd.DataFrame(flat_dict)
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

        df["path"] = checkpoint_manager.dir
        df["epoch"] = trainer_state.epoch
        df["val_loss"] = trainer_state.best_val_loss
        for metric_name, metric in compute_test_metrics(config).items():
            df[f"test_{metric_name}"] = metric
        dfs.append(df)

    if not dfs:
        logger.warning("No data found")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


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
    df = make_eval_df(paths)
    df.to_csv(Paths.output / "evaluation.csv", index=False)
    return df


evaluate_all()
