# Calibrated Boosting-Forest

[![Build Status](https://travis-ci.org/haozhenWu/lightchem.svg?branch=master)](https://travis-ci.org/haozhenWu/lightchem)

Calibrated Boosting-Forest (CBF) is an integrative technique that leverages both continuous and binary labels and output calibrated posterior probabilities. It is originally designed for ligand-based virtual screening and can be extended to
other applications.
Calibrated Boosting-Forest is a package created by Haozhen Wu from [Small Molecule Screening Facility](http://www.uwhealth.org/uw-carbone-cancer-center/for-researchers/shared-resources/smsf/small-molecule-screening/27197)  
at University of Wisconsin-Madison.  

## Key features:  

* Take both continuous and binary labels as input
* Superior ranking power over individual regression or classification model  
* Output well calibrated posterior probabilities
* Streamlined hyper-parameter tuning stage
* Support multiple evaluation metrics  
* Competitive benchmark results for well-known public datasets  
* [XGBoost](https://github.com/dmlc/xgboost) backend

### Table of contents:

* [Example](https://github.com/haozhenWu/lightchem/tree/master/example)
    * [MUV](https://github.com/haozhenWu/lightchem/tree/master/example/muv)
    * [TOX21](https://github.com/haozhenWu/lightchem/tree/master/example/tox21)
    * [PCBA128](https://github.com/haozhenWu/lightchem/tree/master/example/pcba128)  
* [Datasets](https://github.com/haozhenWu/lightchem/tree/master/datasets)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Testing](#testing)
* [FAQ](#faq)
* [RoadMap](https://github.com/haozhenWu/lightchem/issues/1)
* [Reference](#reference)

## Dependencies:

* [scikit-learn](http://scikit-learn.org/stable/index.html)  version = 0.18.1
* [XGBoost](https://xgboost.readthedocs.io/en/latest/) version = 0.6  
* [numpy](http://www.numpy.org/) version = 1.11.1  
* [scipy](https://www.scipy.org/) version = 0.18.1  
* [pandas](http://pandas.pydata.org/) version = 0.18.1   
* [rdkit](http://www.rdkit.org/) version = 2015.09.1
* [pytest](http://doc.pytest.org/) (optional)



## Installation

We recommend you to use Anaconda for convenient installing packages. Right now, LightChem has been tested for Python 2.7 under OS X and linux Ubuntu Server 16.04.   

1. Download 64-bit Python 2.7 version of Anaconda for linux/OS X [here](https://www.continuum.io/downloads) and follow the instruction. After you installed Anaconda, you will have most of the dependencies ready.  

2. Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) if do not have:  
   Linux Ubuntu:    
   ```bash
   sudo yum install git-all
   ```

3. Install `scikit-learn`:  
   ```bash
   conda install scikit-learn=0.18
   ```

4. # Install conda distribution of xgboost
   ```bash
   conda install --yes -c conda-forge xgboost=0.6a2
   ```

5. Install [rdkit](http://www.rdkit.org/docs/Install.html)  Note: `rdkit` is only used to transform SMILE string into fingerprint.  
   ```bash
   conda install -c omnia rdkit
   ```

6. Clone the `Calibrated-Boosting-Forest` github repository:  
   ```bash
   git clone https://github.com/haozhenWu/Calibrated-Boosting-Forest.git
   ```
   `cd` into `lightchem` directory and execute  
   ```
   pip install -e .
   ```


## Testing

To test that the dependencies have been installed correctly, simply enter `pytest`
in the lightchem directory.  This requires the optional `pytest` Python package.
The current tests 1.confirm that the required dependencies exist and can be
imported, 2.confirm the model performance results of one target MUV-466 fall into
expected ranges.

## FAQ  

1. When I import lightchem, the following error shows up `version GLIBCXX_3.4.20 not found`:   
   Try:  
   ```bash   
   conda install libgcc
   ```  
   [Source](http://askubuntu.com/questions/575505/glibcxx-3-4-20-not-found-how-to-fix-this-error)

## Reference

1. [DeepChem] (https://github.com/deepchem/deepchem): Deep-learning models for Drug Discovery and Quantum Chemistry
