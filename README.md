# scaffolding-v3

Create a .env file with the following contents:

```
WANDB_API_KEY=<Your API Key>
```

## Running the code

### Command line config groups
```
-cn in {"hyperparam_opt", "sweep", "train"}
mode in {"dev", "prod"},
data in {"cifar10", "mnist"},
runner in {"default", "parallel"}
```

#### ConfigName
`-cn` option:
 - `hyperparam_opt`: Determines ideal hyperparams, can be easily extended to other hyperparameters.
 - `sweep`: Computes multiple training runs with different seeds.
 - `train`: Trains the model with the given configuration.

#### Mode
`dev` is a dry run for testing that everything is working as expected. Only a small amount of data is loaded, and the model training is broken after one iteration. Results are not sent to wandb.

E.g.
```
python src/scaffolding_v3/train.py mode=dev
```

for a dry run and
```
python src/scaffolding_v3/train.py mode=prod
```
for a real training run.

#### Data
Determines which data source to use. Options are `cifar10` and `mnist`.
 - `cifar10`: Use the CIFAR-10 dataset.
 - `mnist`: Use the MNIST dataset.

#### Runner
Determines parallelization strategy. Options are `default` and `parallel`.
 - `default`: Use the default runner, launching one process (+N dataloader processes)
 - `parallel`: Use the submitit parallel runner, launching N processes (+0 dataloader processes)
