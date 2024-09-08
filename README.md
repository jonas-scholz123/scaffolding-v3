# scaffolding-v3

Create a .env file with the following contents:

```
WANDB_API_KEY=<Your API Key>
```

#### Non-python dependencies
Eccods:
```
sudo apt install libeccodes-dev
```

#### ERA5 Data
To use ERA5 data, you will need to register for an account. Follow the steps [here](https://cds.climate.copernicus.eu/api-how-to#use-the-cds-api-client-for-data-access)
Once you have an API key, in the .env file, set:

```
CDS_API_KEY=<Your API Key>
```
