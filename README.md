# scaffolding-v3

Create a .env file with the following contents:

```
WANDB_API_KEY=<Your API Key>
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

#### Training:
Dry run:
```
python src/scaffolding_v3/train.py -cn dev
```
Actual run:
```
python src/scaffolding_v3/train.py -cn prod
```

Sim data:
```
python src/scaffolding_v3/train.py -cn prod data=sim
```

Real:
```
python src/scaffolding_v3/train.py -cn prod data=real
```

Sim2Real:
```
python src/scaffolding_v3/train.py -cn prod-finetune
```

#### Running any process in background

Use nohup to run any process in the background.
This will allow the process to continue running even after you close the terminal
or terminate the ssh session.
```
CMD="python src/scaffolding_v3/train.py -cn prod"
nohup ${CMD} output.use_tqdm=False > _output/nohup.out 2>&1 &
```
