# MUV benchmark

This folder contains scripts to run MUV(Task = 17, Molecule = 93127) benchmark test. Build models for each task. Used 75% data to perform stratified 3-fold cross-validation, 25% data as final testset. It trains 4 layer one models that use ROCAUC as stopping metric and 4 layer two models where 2 of them use Enrichment Factor@0.01(EFR1) and 2 of them use ROCAUC as stopping metrics.

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
   cp -r ./muv_run dir/outside/lightchem/
   ```

2. `cd` into the directory:  
   ```bash
   cd dir/outside/lightchem/muv_run/
   ```

3. Give permission to execute bash script `./muv_xgboost_models.sh`:  
   ```bash
   chmod 755 ./muv_xgboost_models.sh
   ```

4. Execute bash script.  
    Argument1: dir/to/lightchem/example/muv/  
    e.g. ~/lightchem/example/muv/  

    Argument2: dir/to/store/result/  
    (You can just use current dir, which is muv_run folder)  
    e.g. ./  

    Argument3: muv_TargetName.csv (A csv file containing target names, e.g. 'MUV-466')  
    Used to loop over each target.  
    ```bash
    ./muv_xgboost_models.sh arg1 arg2 arg3
    ```

## Results

The dataframes contains 3-fold cross-validation and final test results  
of muv are stored in muv_cv_result.csv and muv_test_result.csv.
