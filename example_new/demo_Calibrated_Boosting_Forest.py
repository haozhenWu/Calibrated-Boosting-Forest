import sys
import pandas as pd
import numpy as np
import os
from lightchem.ensemble.virtualScreening_models import *
from lightchem.eval.compute_eval import compute_roc_auc

# Read demo data.
file_dir = "./demo_data/demo_pcba1030.csv.gzip"
pcba_1030 = pd.read_csv(file_dir, compression="gzip")

# create train/test data.
pos_index = np.where(pcba_1030.loc[:,'aid1030_binary'] == 1)[0]
neg_index = np.where(pcba_1030.loc[:,'aid1030_binary'] != 1)[0]
train_pos_index = pos_index[1:len(pos_index)/2]
test_pos_index = pos_index[len(pos_index)/2+1:len(pos_index)]
train_neg_index = neg_index[1:len(neg_index)/4]
test_neg_index = neg_index[len(neg_index)/4+1:len(neg_index)/2]
train_index = list(train_pos_index) + list(train_neg_index)
test_index = list(test_pos_index) + list(test_neg_index)
train_data = pcba_1030.iloc[train_index]
test_data = pcba_1030.iloc[test_index]
print train_data.shape
print test_data.shape

# Create CalibratedBoostingForest input format.
training_info = []
testing_info = []
label_name_list = ['aid1030_binary', 'aid1030_logAC50']
training_info.append((train_data, label_name_list))
testing_info.append((test_data, None))
# Depending on your compute resource, try increase H and see the performance difference.
H = 1
num_gbtree = [H,H]
num_gblinear = [H,H]
threshold = train_data.loc[train_data.loc[:,label_name_list[0]]==1,label_name_list[1]].min()
# Internally convert continuous label to binary based on threshold.
# This is used to compate cv scores of regression models on ROC-AUC.
eval_name = 'ROCAUC' + "_" + str(threshold-1)
model = CalibratedBoostingForest(training_info,
                                 eval_name,
                                 fold_info = 3,
                                 createTestset = False,
                                 num_gblinear = num_gblinear,
                                 num_gbtree = num_gbtree,
                                 layer2_modeltype = ['GblinearLogistic'],
                                 nthread = 20)
model.train()

# CBF performance
cv_result = model.training_result()
all_results = model.detail_result()
print 'Predict test data'
y_pred_on_test = model.predict(testing_info)
y_pred_on_train = model.predict(training_info)
y_test = np.array(test_data['aid1030_binary'])
y_train = np.array(training_info[0][0]['aid1030_binary'])
validation_info = model.get_validation_info()
test_score = compute_roc_auc(y_test, y_pred_on_test)
train_score = compute_roc_auc(y_train, y_pred_on_train)
print "Calibrated Boosting-Forest performance\n"
print "CBF cross-validation score:\n", cv_result
print "all models cross-validation scores:\n", all_results
print "CBF Training score:\n", train_score
print "CBF Testing score:\n", test_score
for val in validation_info:
    print "CBF Validation folds scores:\n", compute_roc_auc(val.label, val.validation_pred)

# Set model to use best layer1 GBM
model.set_final_model("layer1")
cv_result = model.training_result()
all_results = model.detail_result()
print 'Predict test data'
y_pred_on_test = model.predict(testing_info)
y_pred_on_train = model.predict(training_info)
validation_info = model.get_validation_info()
test_score = compute_roc_auc(y_test, y_pred_on_test)
train_score = compute_roc_auc(y_train, y_pred_on_train)
print "Best layer1 GBM performance\n"
print "Best GBM cross-validation score:\n", cv_result
print "all models cross-validation scores:\n", all_results
print "Best GBM Training score:\n", train_score
print "Best GBM Testing score:\n", test_score
for val in validation_info:
    print "Best GBM Validation folds scores:\n", compute_roc_auc(val.label, val.validation_pred)
