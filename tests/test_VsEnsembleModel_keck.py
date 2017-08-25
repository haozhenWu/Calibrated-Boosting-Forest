'''
Use MUV-466 to test ensemble model VsEnsembleModel_keck (Rename to Calibrated Boosting-Forest later)
'''

import sys
import pandas as pd
import numpy as np
import os
from lightchem.ensemble.virtualScreening_models import *
from lightchem.eval.compute_eval import compute_roc_auc
import tempfile
import filecmp

target_name = 'MUV-466'
dataset_name = 'muv'

def rmse(series):
    '''
    Root mean square error
    series: pandas.core.series.Series
    '''
    return np.sqrt(np.sum(np.square(series))/len(series))

def test_VsEnsembleModel_keck():
    SEED = 2016
    current_dir = os.path.dirname(os.path.realpath(__file__))
    result_dir = tempfile.mkdtemp()
    setting_list = []
    file_dir = os.path.join(current_dir,
                            "./test_datasets/muv_sample/muv466_ecfp.csv.zip")
    muv = pd.read_csv(file_dir)
    # Manually create a column with continuous label
    muv.loc[:,'cont_label'] = muv.loc[:,'MUV-466']
    index = np.where(muv.loc[:,'cont_label'] == 1)[0]
    muv.loc[index[0:15], 'cont_label'] = 50
    muv.loc[index[15:27], 'cont_label'] = 100

    # Use portion of data to create train/test sets.
    train_index = list(np.where(muv.loc[:,'MUV-466'] == 1)[0][0:20])
    test_index = list(np.where(muv.loc[:,'MUV-466'] == 1)[0][0:27])
    train_index.extend(list(np.where(muv.loc[:,'MUV-466'] == 0)[0][0:1000]))
    test_index.extend(list(np.where(muv.loc[:,'MUV-466'] == 0)[0][2000:3000]))
    train_index = np.unique(train_index)
    test_index = np.unique(test_index)
    train_data = muv.iloc[train_index]
    test_data = muv.iloc[test_index]
    # Create VsEnsembleModel_keck
    training_info = []
    testing_info = []
    label_name_list = ['MUV-466', 'cont_label']
    training_info.append((train_data, label_name_list))
    testing_info.append((test_data, None))
    num_gbtree = [2,2]
    num_gblinear = [2,2]
    eval_name = 'ROCAUC' + "_" + str(0)
    model = VsEnsembleModel_keck(training_info,
                                 eval_name,
                                 fold_info = 3,
                                 createTestset = False,
                                 num_gblinear = num_gblinear,
                                 num_gbtree = num_gbtree,
                                 layer2_modeltype = ['GblinearLogistic'],
                                 nthread = 1)
    model.train()

    my_final_model_list = ['layer2', 'layer1']
    for my_final_model in my_final_model_list:
        model.set_final_model(my_final_model)
        cv_result = model.training_result()
        all_results = model.detail_result()
        cv_result.to_csv(os.path.join(result_dir, my_final_model + '_CBF_cv_result.csv'))
        all_results.to_csv(os.path.join(result_dir, my_final_model + '_CBF_all_result.csv'))

        old = pd.read_csv(os.path.join(current_dir,
        "./test_datasets/muv_sample/" + my_final_model + "_CBF_cv_result.csv"), index_col=0)
        temp_combine = pd.DataFrame({'old' : old.iloc[:,0],'new':cv_result.iloc[:,0]})
        print rmse(temp_combine.new - temp_combine.old)
        assert rmse(temp_combine.new - temp_combine.old) < 0.01

        old = pd.read_csv(os.path.join(current_dir,
        "./test_datasets/muv_sample/" + my_final_model + "_CBF_all_result.csv"), index_col=0)
        temp_combine = pd.DataFrame({'old' : old.iloc[:,0],'new':all_results.iloc[:,0]})
        print rmse(temp_combine.new - temp_combine.old)
        assert rmse(temp_combine.new - temp_combine.old) < 0.01

        #----------- Predict testset
        print 'Predict test data'
        y_pred_on_test = model.predict(testing_info)
        y_pred_on_train = model.predict(training_info)
        y_test = np.array(test_data['MUV-466'])
        y_train = np.array(training_info[0][0]['MUV-466'])
        validation_info = model.get_validation_info()

        test_score = compute_roc_auc(y_test, y_pred_on_test)
        assert test_score > 0.89
        train_score = compute_roc_auc(y_train, y_pred_on_train)
        assert train_score > 0.97
        for val in validation_info:
            assert compute_roc_auc(val.label, val.validation_pred) > 0.7
