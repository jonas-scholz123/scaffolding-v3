# scaffolding-v3

Create a .env file with the following contents:

```
WANDB_API_KEY=<Your API Key>
```

#### Fix broken dependency:
```
sed -i 's/np\.math/np/g' .venv/lib/python3.12/site-packages/fdm/fdm.py
sed -i 's/np\.factorial/math.factorial/g' .venv/lib/python3.12/site-packages/fdm/fdm.py
sed -i '1s/^/import math\n/' .venv/lib/python3.12/site-packages/fdm/fdm.py
```

#### Non-python dependencies
Eccods (linux):
```
sudo apt install libeccodes-dev
```

Eccodes (mac):
```
brew install eccodes
```

#### ERA5 Data
To use ERA5 data, you will need to register for an account. Follow the steps [here](https://cds.climate.copernicus.eu/api-how-to#use-the-cds-api-client-for-data-access)
Once you have an API key, in the .env file, set:

```
CDS_API_KEY=<Your API Key>
```

## Running the code

### Running any process in background

Use nohup to run any process in the background.
This will allow the process to continue running even after you close the terminal
or terminate the ssh session.
```
CMD="python src/scaffolding_v3/train.py -cn prod"
nohup ${CMD} output.use_tqdm=False > _output/nohup.out 2>&1 &
```

### Command line config groups
```
-cn in {"hyperparam_opt", "pretrain", "finetune"}
mode in {"dev", "prod"},
data in {"sim", "real"},
runner in {"default", "parallel"}
```

#### ConfigName
`-cn` option:
 - `hyperparam_opt`: Determines the ideal learning rate, can be easily extended to other hyperparameters.
 - `pretrain`: Pretrain the model from scratch (defaults to sim data).
 - `finetune`: Finetune the model from a pretrained model (defaults to sim -> real)

#### Mode
`dev` is a dry run for testing that everything is working as expected. Only a small amount of data is loaded, and the model training is broken after one iteration. Results are not sent to wandb.

E.g.
```
python src/scaffolding_v3/train.py -cn hyperparam_opt mode=dev
```

for a dry run and
```
python src/scaffolding_v3/train.py -cn hyperparam_opt mode=prod
```
for a real training run.

#### Data
Determines which data source to use. Options are `sim` and `real`.
 - `sim`: Use gridded ERA5 weather data over Germany.
 - `real`: Use real weather data from the German weather service (DWD).

#### Runner
Determines parallelization strategy. Options are `default` and `parallel`.
 - `default`: Use the default runner, launching one process (+N dataloader processes)
 - `parallel`: Use the submitit parallel runner, launching N processes (+0 dataloader processes)
