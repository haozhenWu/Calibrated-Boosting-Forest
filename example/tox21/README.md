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

1. Copy the `run` folder outside lightchem:  
   ```bash
   cp -r ./tox21_run dir/outside/lightchem/
   ```

2. `cd` into the directory:  
   ```bash
   cd dir/outside/lightchem/
   ```

3. Give permission to execute bash script `./tox21_xgboost_models.sh`:  
   ```bash
   chmod 755 ./tox21_xgboost_models.sh
   ```

4. Execute bash script.  
    Argument1: dir/to/lightchem/example/tox21/  
    e.g. ~/lightchem/example/tox21/  

    Argument2: dir/to/store/result/  
    (You can just use current dir, which is tox21_run folder)  
    e.g. ./  

    Argument3: tox21_TargetName.csv (A csv file containing target names, e.g. 'NR-AR')  
    Used to loop over each target.  
    ```bash
    ./tox21_xgboost_models.sh arg1 arg2 arg3
    ```

## Results

The dataframes contains 3-fold cross-validation and final test results  
of tox21 are stored in tox21_cv_result.csv and tox21_test_result.csv.
