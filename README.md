# LightChem

LightChem provides high level machine-learning interface that served to used by researcher without deep machine-learning background. It aims to offer convenient exploration for researcher interested in machine-learning for drug discovery. LightChem is a package created by Haozhen Wu from [Small Molecule Screening Facility](http://www.uwhealth.org/uw-carbone-cancer-center/for-researchers/shared-resources/smsf/small-molecule-screening/27197)  
at University of Wisconsin-Madison.  

## Key features:  

* [XGBoost](https://github.com/dmlc/xgboost) backend
* Parallel computing    
* Supports tree based models and regression models  
* Interface to use ensemble( stacking ) models  
* Support multiple evaluation metrics  
* Common featurization methods to transform molecules into fingerprint  
* Competitive benchmark results for well-known public datasets  

### Table of contents:

* [Example](https://github.com/haozhenWu/lightchem/tree/master/example)
    * [MUV](https://github.com/haozhenWu/lightchem/tree/master/example/muv)
    * [TOX21](https://github.com/haozhenWu/lightchem/tree/master/example/tox21)
    * [PCBA128](https://github.com/haozhenWu/lightchem/tree/master/example/pcba128)  
* [Datasets](https://github.com/haozhenWu/lightchem/tree/master/datasets)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Benchmark Results](#benchmark-results)
* [FAQ](#faq)
* [RoadMap](https://github.com/haozhenWu/lightchem/issues/1)
* [Reference](#reference)

## Dependencies:

* [sklearn](http://scikit-learn.org/stable/index.html)  version = 0.18.1
* [XGBoost](https://xgboost.readthedocs.io/en/latest/) version = 0.6  
* [numpy](http://www.numpy.org/) version = 1.11.1  
* [scipy](https://www.scipy.org/) version = 0.18.1  
* [pandas](http://pandas.pydata.org/) version = 0.18.1   
* [rdkit](http://www.rdkit.org/) version = 2015.09.1



## Installation

We recommend you to use Anaconda for convenient installing packages. Right now, LightChem has been tested for Python 2.7 under OS X and linux Ubuntu Server 16.04.   

1. Download 64-bit Python 2.7 version of Anaconda for linux/OS X [here](https://www.continuum.io/downloads) and follow the instruction. After you installed Anaconda, you will have most of the dependencies ready.  

2. Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) if do not have:  
   Linux Ubuntu:    
   ```bash
   sudo yum install git-all
   ```

3. Install `sklearn`:  
   ```bash
   conda install scikit-learn=0.18
   ```

4. Install `xgboost` [documentation](https://xgboost.readthedocs.io/en/latest/build.html)  
   Linux Ubuntu:  Build xgboost shared library  
   ```bash
   git clone --recursive https://github.com/dmlc/xgboost
   cd xgboost; make -j4
   ```
   Or if OSX: Build xgboost shared library  
   ```bash
   git clone --recursive https://github.com/dmlc/xgboost
   cd xgboost; cp make/minimum.mk ./config.mk; make -j4
   ```
   After shared library is built, build xgboost python package  
   ```bash
   cd python-package; python setup.py develop --user
   ```

5. Install [rdkit](http://www.rdkit.org/docs/Install.html)  Note: `rdkit` is only used to transform SMILE string into fingerprint.  
   ```bash
   conda install -c omnia rdkit
   ```

6. Clone the `lightchem` github repository:  
   ```bash
   git clone https://github.com/haozhenWu/lightchem.git
   ```
   `cd` into `lightchem` directory and execute  
   ```
   pip install -e .
   ```

## Benchmark Results

Model Abbreviation:  
`GbtreeLogistic`: Gradient boosted tree with binary logistic as objective  
`GbtreeRegression`: Gradient boosted tree with linear regression as objective  
`GblinearLogistic`: Gradient boosted linear with binary logistic as objective    
`GblinearRegression`: Gradient boosted linear with linear regression as objective  

*Evaluation results report below are mean/median of k-fold cross-validation results among all targets in a given dataset.*

### Concatenated Version:  
Molecules of targets from same dataset are concatenated together into a single big dataframe, which leads to many missing values in the labels. Molecules without a lab result for a given target assume having a negative label. This approach might benefit multi-task learning.    
LightChem so far only supports single-task learning, which builds separate models for each target. Will have Self-contained molecules version later.

### Classification  

Stratified Split   
Cross Validation: 3 folds, 75% of the data    
Test: Remaining 25% of the data   

#### ROC-AUC   
Cross Validation

|Dataset |Layer |Label  |Feature |Model            |Mean            |Median              |
|--------|------|-------|--------|-----------------|-------------------|-----------------|
|tox21   |Layer1|Binary |ECFP1024|GbtreeLogistic   |0.783+-0.022       |0.779+-0.017     |
|        |      |       |        |GblinearLogistic |0.729+-0.021       |0.747+-0.012     |
|        |      |       |MACCSkeys |GbtreeLogistic |0.798+-0.019    |0.801+-0.016     |         
|        |      |       |          |GblinearLogistic |0.766+-0.02   |0.776+-0.023     |
|        |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |0.790+-0.018 |0.794+-0.014 |
|        |      |       |                           |GblinearLogistic |**0.809**+-0.020 |**0.806**+-0.017 |
|PCBA128 |Layer1|Binary |ECFP1024 |GbtreeLogistic |0.843+-0.012 |0.857+-0.007 |
|        |      |       |MACCSkeys |GbtreeLogistic |0.835+-0.012 |0.847+-0.008 |
|        |      |Continuous |ECFP1024 |GbtreeRegression |0.863+-0.013 |0.882+-0.008 |
|        |      |           |MACCSkeys |GbtreeRegression |0.852+-0.013 |0.865+-0.009 |
|        |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |**0.880**+-0.011 |**0.894**+-0.007 |
|        |      |       |                           |GblinearLogistic |0.879+-0.015 |0.893+-0.006 |
|muv     |Layer1|Binary |ECFP1024|GbtreeLogistic   |0.799+-0.053       |0.793+-0.048     |
|        |      |       |        |GblinearLogistic |0.756+-0.055       |0.805+-0.044     |
|        |      |       |MACCSkeys |GbtreeLogistic |**0.810**+-0.049    |**0.820**+-0.046     |         
|        |      |       |          |GblinearLogistic |0.755+-0.054   |0.756+-0.059     |
|        |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |0.787+-0.053 |0.811+-0.053 |
|        |      |       |                           |GblinearLogistic |0.687+-0.155 |0.669+-0.109 |

Test:

|Dataset |Layer |Label  |Feature |Model            |Mean         |Median              |
|--------|------|-------|--------|-----------------|-------------------|--------------|
|tox21   |Layer1|Binary |ECFP1024|GbtreeLogistic   |0.776    |0.748     |
|        |      |       |        |GblinearLogistic |0.734    |0.738     |
|        |      |       |MACCSkeys |GbtreeLogistic |0.805    |**0.797**     |         
|        |      |       |          |GblinearLogistic |0.765   |0.769     |
|        |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |0.795 |0.783 |
|        |      |       |                           |GblinearLogistic |**0.808** |0.788 |
|PCBA128 |Layer1|Binary |ECFP1024 |GbtreeLogistic |0.850 |0.866 |
|        |      |       |MACCSkeys |GbtreeLogistic |0.843 |0.857 |
|        |      |Continuous |ECFP1024 |GbtreeRegression |0.872 |0.890 |
|        |      |           |MACCSkeys |GbtreeRegression |0.856 |0.868 |
|        |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |0.883 |**0.903** |
|        |      |       |                           |GblinearLogistic |**0.884** |0.901 |
|muv     |Layer1|Binary |ECFP1024|GbtreeLogistic   |0.744    |0.712     |
|        |      |       |        |GblinearLogistic |0.671    |0.681     |
|        |      |       |MACCSkeys |GbtreeLogistic |**0.774** |**0.824** |         
|        |      |       |          |GblinearLogistic |0.751   |0.782     |
|        |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |0.663 |0.689 |
|        |      |       |                           |GblinearLogistic |0.552 |0.605 |

#### EFR1   
Cross Validation

|Dataset |Layer |Label  |Feature |Model            |Mean         |Median                 |
|--------|------|-------|--------|-----------------|-------------------|-----------------|
|tox21   |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |**15.724**+-1.103 |**18.408**+-0.726 |
|        |      |       |                           |GblinearLogistic |15.162+-1.702 |18.352+-1.122 |
|PCBA128 |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |**33.873**+-2.736 |32.822+-1.682 |
|        |      |       |                           |GblinearLogistic |32.665+-3.816 |**33.053**+-1.869 |
|muv     |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |**21.580**+-9.621 |**18.452**+-7.192 |
|        |      |       |                           |GblinearLogistic |7.178+-5.489 |0+-0 |

Test:

|Dataset |Layer |Label  |Feature |Model            |Mean         |Median                 |
|--------|------|-------|--------|-----------------|-------------------|-----------------|
|tox21   |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |**15.612** |**12.360**  |
|        |      |       |                           |GblinearLogistic |14.774 |11.842 |
|PCBA128 |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |**34.413** |**33.333**  |
|        |      |       |                           |GblinearLogistic |33.703 |32.727 |
|muv     |Layer2|Binary |Layer1 holdout predictions |GbtreeLogistic |**11.905** |0  |
|        |      |       |                           |GblinearLogistic |0 |0 |



## FAQ  

1. When I import lightchem, the following error shows up `version GLIBCXX_3.4.20 not found`:   
   Try:  
   ```bash   
   conda install libgcc
   ```  
   [Source](http://askubuntu.com/questions/575505/glibcxx-3-4-20-not-found-how-to-fix-this-error)

## Reference

1. [DeepChem] (https://github.com/deepchem/deepchem): Deep-learning models for Drug Discovery and Quantum Chemistry
