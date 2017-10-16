This folder contains scripts of running keck_pria datasets.   
General workflow:  
1. Script keck_use_FP_stage1.py train Calibrated Boosting-Forest models and generate
cross validation scores based on pre-splited fixed 5 folds.   
Each time use 3 folds as train, 1 fold as valid, and 1 fold as test. Output --> out.txt and summary.txt   

2. Script keck_use_FP_stage2.py train Calibrated Boosting-Forest models based on    
4 folds as train, 1 folds as valid. The test data is LC4 which we do not know the true label at that time.

Scripts keck_dataset.py, generate_features.py use DeepChem's featurizer.   
Experimenting adding ConvGraph features. (Need DeepChem installed.)
