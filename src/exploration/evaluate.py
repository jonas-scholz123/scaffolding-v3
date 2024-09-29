# %%
from dataclasses import asdict
from mlbnb.paths import ExperimentPath
from omegaconf import OmegaConf
from mlbnb.checkpoint import CheckpointManager
from scaffolding_v3.config import Config, Paths, SKIP_KEYS
from loguru import logger
import pandas as pd

from scaffolding_v3.train import TrainerState

paths = Paths()
paths = paths.output.rglob("*/checkpoints/best.pt")
paths = [ExperimentPath(path.parent.parent) for path in paths]
logger.info("Found {} matching path(s)", len(paths))

dfs = []
for path in paths:
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

    df["epoch"] = trainer_state.epoch
    df["val_loss"] = trainer_state.best_val_loss
    dfs.append(df)

df = pd.concat(dfs)
df
