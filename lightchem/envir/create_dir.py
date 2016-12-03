"""
Setup required foler directory.
"""

import os

def CreateDir():

    dir_list = ['./analyzed_result',
                './cv_result',
                './folds_index',
                './xgb_data',
                './xgb_model',
                './xgb_param',
                './first_layer_holdout',
                './second_layer_holdout',
                './first_layer_prediction',
                './second_layer_prediction']

    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
