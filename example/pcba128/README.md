# PCBA128 benchmark

This folder contains scripts to run pcba128(PCBA, task = 128) benchmark test.    
Used 75% data to perform stratified 3-fold cross-validation,  
25% data as final testset.  
It trains 4 layer one models that use ROCAUC as stopping metric and 4 layer two  
models where 2 of them use Enrichment Factor@0.01(EFR1) and 2 of them use ROCAUC  
as stopping metrics.

## Model detail

Abbreviation:  
`GbtreeLogistic`: Gradient boosted tree with binary logistic as objective  
`GbtreeRegression`: Gradient boosted tree with linear regression as objective  
`GblinearLogistic`: Gradient boosted linear with binary logistic as objective    

### Layer one models

#### Stopping metric = `ROCAUC`

* Datasets: Features = ecfp1024 fingerprint, Label = Binary label  
Model: `GbtreeLogistic`
* Datasets: Features = ecfp1024 fingerprint, Label = Continuous label
Model: `GbtreeRegression`
* Datasets: Features = MACCSkeys167 fingerprint, Label = Binary label
Model: `GbtreeLogistic`
* Datasets: Features = MACCSkeys167 fingerprint, Label = Continuous label
Model: `GbtreeRegression`

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
   scp ./pcba128_run dir/outside/lightchem/
   ```

2. `cd` into the directory:  
   ```bash
   cd dir/outside/lightchem/pcba128_run/
   ```

3. Give permission to execute bash script `./pcba128_xgboost_models.sh`:  
   ```bash
   chmod 755 ./pcba128_xgboost_models.sh
   ```

4. Execute bash script.  
    Argument1: dir/to/lightchem/example/pcba128/  
    e.g. ~/lightchem/example/pcba128/  

    Argument2: dir/to/store/result/  
    (You can just use current dir, which is pcba128_run folder)  
    e.g. ./  

    Argument3: aid_list.csv (A csv file containing target names, e.g. 'aid411')  
    Used to loop over each target.  
    ```bash
    ./pcba128_xgboost_models.sh arg1 arg2 arg3
    ```

## Results

The dataframes contains 3-fold cross-validation and final test results  
of pcba128 are stored in pcba128_cv_result.csv and pcba128_test_result.csv.
