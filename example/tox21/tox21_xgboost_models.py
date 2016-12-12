"""
Scripts to test tox21.
Datasets used:
fingerprint = ecfp1024, Label = Binary label
fingerprint = MACCSkeys167, Label = Binary label

Model used:
Gbtree
"""

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
#
import time


target_name = sys.argv[1]
dir_to_store_result = sys.argv[2]
dataset_name = 'tox21'

dir_list = [dir_to_store_result + 'each_target_cv_result',
            dir_to_store_result + 'each_target_test_result']

for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)


# Create folder to store results.

if __name__ == "__main__":
    start_time = time.time()
    #----------------------------------- Build first layer data
    current_dir = os.path.dirname(os.path.realpath(__file__))
    setting_list = []
    # binary ecfp1024
    file_dir = os.path.join(current_dir,
                            "../../datasets/tox21/classification/tox21_BinaryLabel_ecfp1024.csv.zip")
    data_name = 'binaryECFP'
    label_colname = target_name # label column name of one target
    model_name_to_use = ['GbtreeLogistic'] # Define model to use
    temp_data = load.readData(file_dir,label_colname)
    temp_data.read()
    X_data = temp_data.features()
    y_data = temp_data.label()
    myfold = fold.fold(X_data,y_data,4)
    myfold = myfold.generate_skfolds()
    data = xgb_data.xgbData(myfold,X_data,y_data)
    data.build()
    setting_list.append({'data_name':data_name,'model_type':model_name_to_use,
                        'data':data})


    # binary MACCSkeys
    file_dir = os.path.join(current_dir,
                            "../../datasets/tox21/classification/tox21_BinaryLabel_MACCSkey167.csv.zip")
    data_name = 'binaryMACCSkeys'
    label_colname = target_name
    model_name_to_use = ['GbtreeLogistic']
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
    layer1_model_list = []
    evaluation_metric_name = 'ROCAUC'
    for data_dict in setting_list:
        for model_type in data_dict['model_type']:
            unique_name = 'layer1_' + data_dict['data_name'] + '_' + model_type + '_' + evaluation_metric_name
            model = first_layer_model.firstLayerModel(data_dict['data'],
                    evaluation_metric_name,model_type,unique_name)
            model.xgb_cv()
            model.generate_holdout_pred()
            layer1_model_list.append(model)


    #------------------------------------second layer models
    # use label from binary data to train layer2 models
    #layer1_model_list
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
            l2model.xgb_cv()
            layer2_model_list.append(l2model)


    #------------------------------------ evaluate model performance on test data
    # prepare test data, retrive from layer1 data
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

    # collect cv result and convert to a list
    all_model = layer1_model_list + layer2_model_list
    result = []
    for model in all_model:
        result = result + [list(item)[0] for item in np.array(model.cv_score_df())]
    # Retrieve corresponding name of cv result
    result_index = []
    for model in all_model:
        result_index.append(model.name + '_mean')
        result_index.append(model.name + '_std')
    # create a dataframe
    result = pd.DataFrame({target_name : result},index = result_index)
    result.to_csv(dir_to_store_result + 'each_target_cv_result/' + dataset_name + '_' + target_name + "_cv_result.csv")

    # collect test result
    result = pd.concat(test_result_list,axis = 0,ignore_index=False)
    result.to_csv(dir_to_store_result + 'each_target_test_result/' + dataset_name + '_' + target_name + "_test_result.csv")

    # moniter processing time
    with open(dir_to_store_result + "process_time.txt", "a") as text_file:
        text_file.write((target_name
         + " --- %s seconds ---\n" % (time.time() - start_time)))
