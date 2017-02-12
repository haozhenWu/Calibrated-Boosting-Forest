'''
Test each function used in  MUV example script.
'''

import sys
from sklearn import metrics
from lightchem.load import load
from lightchem.fold import fold
from lightchem.data import xgb_data
from lightchem.eval import xgb_eval
from lightchem.eval import eval_testset
from lightchem.model import first_layer_model
from lightchem.model import second_layer_model
import pandas as pd
import numpy as np
import os
import time
import tempfile
import filecmp
import xgboost

target_name = 'MUV-466'
#dir_to_store_result
dataset_name = 'muv'

def test_muv_function():
    SEED = 2016
    #----------------------------------- Build first layer data
    print '{}: loading data'.format(target_name)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    setting_list = []
    # binary ecfp1024
    file_dir = os.path.join(current_dir,
                            "./muv466_ecfp.csv")
    data_name = 'ecfp'
    label_colname = target_name # label column name of one target
    model_name_to_use = ['GbtreeLogistic','GblinearLogistic'] # Define model to use
    temp_data = load.readData(file_dir,label_colname)
    temp_data.read()
    X_data = temp_data.features()
    # check training feature dimension
    assert X_data.shape == (93127,1024)
    # check label dimension
    y_data = temp_data.label()
    assert y_data.shape == (93127,)
    # check label positive number
    assert y_data.sum() == 27
    myfold = fold.fold(X_data,y_data,4,SEED)
    myfold = myfold.generate_skfolds()
    result_dir = tempfile.mkdtemp()
    myfold.to_csv(os.path.join(result_dir,'fold_all.csv'))
    # check whether stratified 4 folds are the same
    assert filecmp.cmp(os.path.join(result_dir,'fold_all.csv'),
    "./test_datasets/muv_sample/muv466_folds_all.csv")
    data = xgb_data.xgbData(myfold,X_data,y_data)
    data.build()
    # check whether training and test data objects are xgboost.core.DMatrix
    assert isinstance(data.get_dtest(),xgboost.core.DMatrix)
    assert all([isinstance(data.get_dtrain(i)[0],xgboost.core.DMatrix) for i in range(3)])
    # check positive number of training data and test data.
    assert data.get_holdoutLabel().sum() == 21
    assert data.get_testLabel().sum() == 6
    # check whether train folds are the same
    data.get_train_fold().to_csv(os.path.join(result_dir,'fold_train.csv'))
    assert filecmp.cmp(os.path.join(result_dir,'fold_train.csv'),
    "./test_datasets/muv_sample/muv466_folds_train.csv")
    # check number of training folds
    assert data.numberOfTrainFold() == 3
    setting_list.append({'data_name':data_name,'model_type':model_name_to_use,
                        'data':data})


    # binary MACCSkeys
    file_dir = os.path.join(current_dir,
                            "./test_datasets/muv_sample/muv466_macckey.csv")
    data_name = 'macckey'
    label_colname = target_name # label column name of one target
    model_name_to_use = ['GbtreeLogistic','GblinearLogistic']
    temp_data = load.readData(file_dir,label_colname)
    temp_data.read()
    X_data = temp_data.features()
    y_data = temp_data.label()
    data = xgb_data.xgbData(myfold,X_data,y_data)
    data.build()
    setting_list.append({'data_name':data_name,'model_type':model_name_to_use,
                        'data':data})

    #---------------------------------first layer models ----------
    # 4 layer1 models based on ecfp,MACCSkeys data
    # gbtree
    print '{}: building first layer models'.format(target_name)
    layer1_model_list = []
    evaluation_metric_name = 'ROCAUC'
    for data_dict in setting_list:
        for model_type in data_dict['model_type']:
            unique_name = 'layer1_' + data_dict['data_name'] + '_' + model_type + '_' + evaluation_metric_name
            model = first_layer_model.firstLayerModel(data_dict['data'],
                    evaluation_metric_name,model_type,unique_name)
            # Retrieve default parameter and change default seed.
            default_param,default_MAXIMIZE,default_STOPPING_ROUND = model.get_param()
            default_param['seed'] = SEED
            # Default parameters overfit muv dataset, use more conservative param.
            if model_type == 'GbtreeLogistic':
                default_param['eta'] = 0.03
                default_param['max_depth'] = 5
                default_param['colsample_bytree'] = 0.5
                default_param['min_child_weight'] = 2
            elif model_type == 'GblinearLogistic':
                default_param['eta'] = 0.1

            model.update_param(default_param,default_MAXIMIZE,default_STOPPING_ROUND)
            model.xgb_cv()
            model.generate_holdout_pred()
            layer1_model_list.append(model)

    # check whether second decimal of cv scores are the same.
    cv_result = pd.concat([layer1_model_list[0].cv_score_df(),
                            layer1_model_list[1].cv_score_df(),
                            layer1_model_list[2].cv_score_df(),
                            layer1_model_list[3].cv_score_df()])
    cv_result = np.round(cv_result,2)
    cv_result.to_csv(os.path.join(result_dir,'firstlayerModel_cvScore.csv'))
    assert filecmp.cmp(os.path.join(result_dir,'firstlayerModel_cvScore.csv'),
    "./test_datasets/muv_sample/muv466_firstlayerModel_cvScore.csv")

    # check whether holdout results of first layer model are same, round to THIRD decimal.
    holdout_result = pd.DataFrame({layer1_model_list[0].name : layer1_model_list[0].get_holdout(),
                                    layer1_model_list[1].name : layer1_model_list[1].get_holdout(),
                                    layer1_model_list[2].name : layer1_model_list[2].get_holdout(),
                                    layer1_model_list[3].name : layer1_model_list[3].get_holdout()})
    holdout_result = np.round(holdout_result,3)
    holdout_result.to_csv(os.path.join(result_dir,'firstlayerModel_holdout.csv'))
    assert filecmp.cmp(os.path.join(result_dir,'firstlayerModel_holdout.csv'),
    "./test_datasets/muv_sample/muv466_firstlayerModel_holdout.csv")


    #------------------------------------second layer models
    # use label from binary data to train layer2 models
    #layer1_model_list
    print '{}: building second layer models'.format(target_name)
    layer2_label_data = setting_list[0]['data'] # layer1 data object containing the label for layer2 model
    layer2_model_list = []
    layer2_modeltype = ['GbtreeLogistic','GblinearLogistic']
    layer2_evaluation_metric_name = ['ROCAUC','EFR1']
    for evaluation_metric_name in layer2_evaluation_metric_name:
        for model_type in layer2_modeltype:
            unique_name = 'layer2' + '_' + model_type + '_' + evaluation_metric_name
            l2model = second_layer_model.secondLayerModel(layer2_label_data,layer1_model_list,
                        evaluation_metric_name,model_type,unique_name)
            l2model.second_layer_data()
            # Retrieve default parameter and change default seed.
            default_param,default_MAXIMIZE,default_STOPPING_ROUND = l2model.get_param()
            default_param['seed'] = SEED
            # Default parameters overfit muv dataset, use more conservative param.
            if model_type == 'GbtreeLogistic':
                default_param['eta'] = 0.06
                default_param['max_depth'] = 5
                default_param['colsample_bytree'] = 0.5
                default_param['min_child_weight'] = 2
            elif model_type == 'GblinearLogistic':
                default_param['eta'] = 0.12

            l2model.update_param(default_param,default_MAXIMIZE,default_STOPPING_ROUND)
            l2model.xgb_cv()
            layer2_model_list.append(l2model)

    # check whether second decimal of cv scores are the same.
    cv_result = pd.concat([layer2_model_list[0].cv_score_df(),
                            layer2_model_list[1].cv_score_df(),
                            layer2_model_list[2].cv_score_df(),
                            layer2_model_list[3].cv_score_df()])
    cv_result = np.round(cv_result,2)
    cv_result.to_csv(os.path.join(result_dir,'secondlayerModel_cvScore.csv'))
    assert filecmp.cmp(os.path.join(result_dir,'secondlayerModel_cvScore.csv'),
    "./test_datasets/muv_sample/muv466_secondlayerModel_cvScore.csv")


    #------------------------------------ evaluate model performance on test data
    # prepare test data, retrive from layer1 data
    print '{}: evaluating test set'.format(target_name)
    list_TestData = []
    for data_dict in setting_list:
        for model_type in data_dict['model_type']:
            list_TestData.append(data_dict['data'].get_dtest())

    test_label = layer2_label_data.get_testLabel()

    test_result_list = []
    i = 0
    for evaluation_metric_name in layer2_evaluation_metric_name:
        for model_type in layer2_modeltype:
            test_result = eval_testset.eval_testset(layer2_model_list[i],
                                                    list_TestData,test_label,
                                                    evaluation_metric_name)
            test_result_list.append(test_result)
            i += 1

    # collect test result
    result = pd.concat(test_result_list,axis = 0,ignore_index=False)
    result = np.round(result,3)
    result.to_csv(os.path.join(result_dir,'testResult_all.csv'))
    assert filecmp.cmp(os.path.join(result_dir,'testResult_all.csv'),
    "./test_datasets/muv_sample/muv466_testResult_all.csv")
