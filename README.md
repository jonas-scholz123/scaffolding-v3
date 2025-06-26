# Scaffolding

This repository provides an opinionated scaffolding for PyTorch projects. It is designed to be a robust starting point for turning early-stage Jupyter notebook experiments into a more structured, reproducible, and scalable format.

The core philosophy is to provide a clean, well-structured foundation that includes best practices for configuration, experiment tracking, and model development, allowing you to focus on the experiment itself, without having to learn a new opaque framework like lightning.

It uses [the mlbnb library](https://pypi.org/project/mlbnb/#description) under the hood for some of the tooling.

## Core Features

- **Dev Mode for fast debugging**: There's nothing worse than missing a bug that only happens after a training epoch, potentially losing you hours. Scaffolding has a `dev` mode, that shortcuts through the training loop so you can be sure that your script won't crash.
- **Configuration with Hydra**: Leverages `hydra` for flexible and clean configuration management. This allows for easy swapping of models, datasets, and training parameters via YAML files.
- **Experiment Tracking with W&B**: Integrates with Weights & Biases (`wandb`) for logging metrics, tracking experiments, and visualizing results.
- **A dedicated experiment directory for each run**, where checkpoints, plots and logs are saved
- **Checkpointing**: Includes a `CheckpointManager` to save and load model checkpoints, optimizer states, and training progress. This allows for resuming training and saving the best-performing models.
- **Structured Logging**: Uses `loguru` for clear and configurable logging.
- **Data Loading**: A simple `make_dataset` factory to instantiate datasets for different splits (`train`, `val`, `test`).
- **Tooling for Exploration**: Provides utilities for plotting predictions and exploring model behavior.

## Project Structure

```
.
├── src
│   ├── config/                 # Hydra configuration files
│   ├── scaffolding_v3
│   │   ├── data/               # Data loading and dataset definitions
│   │   ├── model/              # Model definitions
│   │   ├── plot/               # Plotting utilities
│   │   ├── util/               # Helper utilities
│   │   ├── config.py           # Configuration dataclasses
│   │   ├── evaluate.py         # Evaluation loop
│   │   └── train.py            # Main training script
│   └── scripts/                # Standalone scripts for analysis
├── tests/                      # Tests
└── README.md
```

## Getting Started

### Installation

We thoroughly reccommend using [pdm](https://pdm-project.org/en/latest/) as your dependency management tool, and scaffolding is set up around it.

To install pdm (Linux/Mac):

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

Then:

```bash
pdm venv create
source .venv/bin/activate
pdm install
```

Finally, for optional but recommended W&B integration, set your API key:

```
WANDB_API_KEY=<Your API Key>
```

### Training a Model

The main entry point for training is `src/scaffolding_v3/train.py`. You can run it using `hydra`'s command-line interface.

To ensure your training script runs, you can use the `mode=dev` override, which fast-forwards through the training code.

```bash
python src/scaffolding_v3/train.py data=mnist mode=dev
```

If that looks good, you can train the model by using `mode=prod`:

```bash
python src/scaffolding_v3/train.py data=mnist mode=prod
```

To override configuration parameters, for example to use a different dataset:

```bash
python src/scaffolding_v3/train.py data=cifar10 mode=dev
```

### Resuming Training

Each run is automatically checkpointed (including RNG state) in the `_results/<experiment_name>` directory. You can continue a previous run via the `resume` override:

```bash
python src/scaffolding_v3/train.py resume=<experiment_name>
```

# How should you use Scaffolding?

Scaffolding tries to balance the iteration speed you get out of jupyter notebooks with the robustness you want out of long-term ML codebases, without you having to translate your script into a whole new framework like "pytorch lightning", or writing the boilerplate code yourself. The codebase is meant to be edited by you and adjusted to fit your experiments. The ideal workflow looks something like this:

1. Experiment with your new research idea (e.g. "bird call reinforcement learning, BCRL"), e.g. in a jupyter notebook or wherever you run your quick-and-dirty experiments
1. Clone the scaffolding repo
1. Replace all "scaffolding-v3" --> "bcrl" to make the codebase yours
1. Copy your bird call dataset into data/birdcall.py
1. Copy your RL model into model/rl_model.py
1. Copy the magic numbers from your jupyter notebooks into the `base.yaml` config, for example `lr=1e-4`
1. Edit the `config.py` file to contain the values you need
1. Edit base.yaml to instantiate your new dependencies, like the `birdcall` dataset and the `rl model` you created. Those should replace the existing models/datasets etc that you no longer need.
1. Delete any of the models/data/code directories that you don't need.
1. Run your new code with `python src/scaffolding_v3/train.py data=birdcall mode=dev`, and adjust the `train.py` script until it runs
1. Once your code runs, launch a real training run by replacing `mode=dev` with `mode=prod`.
1. Enjoy for free: wandb logging of metrics and configs, validation plots, checkpointing, resumption of your run (with correct wandb handling), timing/profiling of your code, mixed precision training, and a well structured codebase.

When you want to test or debug your model, and load the best checkpoint, it's as easy as making a script in `src/scripts/emulate_birdcall.py`:

```python
from scaffolding_v3.util.instantiate import Experiment
# This should be your experiment path
path = "_output/2025-06-25_13-59_gentle_aardvark"
exp = Experiment.from_path(path, checkpoint="best")

x, y = next(iter(exp.val_loader))

pred = exp.model(x)
```
