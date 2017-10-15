# TOX21 benchmark

This folder contains scripts to run TOX21(Task = 12, Molecule = 8014) benchmark test. Build models for each task. Used 75% data to perform stratified 3-fold cross-validation, 25% data as final testset. It trains 4 layer one models that use ROCAUC as stopping metric and 4 layer two models where 2 of them use Enrichment Factor@0.01(EFR1) and 2 of them use ROCAUC as stopping metrics.

## Model detail

Abbreviation:  
`GbtreeLogistic`: Gradient boosted tree with binary logistic as objective  
`GblinearLogistic`: Gradient boosted linear with binary logistic as objective    

### Layer one models

#### Stopping metric = `ROCAUC`

* Datasets: Features = ecfp1024 fingerprint, Label = Binary label  
Model: `GbtreeLogistic`, `GblinearLogistic`
* Datasets: Features = MACCSkeys167 fingerprint, Label = Binary label  
Model: `GbtreeLogistic`, `GblinearLogistic`

### Layer two models

#### Stopping metric = `ROCAUC`

* Datasets: Features = holdout(out of fold) predictions from layer1 models,  
            Label = Binary label  
Model: `GbtreeLogistic`, `GblinearLogistic`

#### Stopping metric = `EFR1`

* Datasets: Features = holdout(out of fold) predictions from layer1 models,  
            Label = Binary label  
Model: `GbtreeLogistic`, `GblinearLogistic`

## Usage

1. `cd` into the tox21_run directory:  
   ```bash
   cd ./tox21_run/
   ```

2. Give permission to execute bash script `./tox21_xgboost_models.sh`:  
   ```bash
   chmod 755 ./tox21_xgboost_models.sh
   ```

3. Execute bash script.
   Arg1: dir/to/lightchem/example/tox21/  
   Arg2: dir/to/store/result/
   Arg3: tox21_TargetName.csv

   ```bash
   ./tox21_xgboost_models.sh .. ../tox21_results ./tox21_TargetName.csv
   ```

## Results

The dataframes contains 3-fold cross-validation and final test results  
of tox21 are stored in tox21_cv_result.csv and tox21_test_result.csv,
inside ../tox21_results folder.
