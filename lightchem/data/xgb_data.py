"""
Contains lightchem's data format.
"""

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
import itertools as it
import xgboost as xgb
import scipy
import os
import glob

class xgbData(object):
    """
    Class contains lightchem's data format.
    """

    def __init__(self,fold,X_data,y_data,createTestset = True):
        """Default data format.
        Parameters:
        -----------
        fold: pandas.DataFrame
        Contains cross validation folds information.
        X_data: numpy.ndarray
          Training features
        y_data: numpy.ndarray
          Label(Response variable)
        createTestset: boolean, default to True
          Whether to create test dataset. If True, use one fold as
          test set, remaining folds as training set.
        """
        self.__folds = fold
        self.__train_folds = None
        self.__num_train_fold = None
        self.__all_x = X_data
        self.__label = y_data
        self.__train_x = None
        self.__train_label = None
        self.__test_x = None
        self.__test_label = None
        self.__collect_dtrain = None
        self.__collect_dtest = None
        self.__has_test = createTestset


    def build(self):
        """
        Create each fold's training and validating data and tranform
        to xgboost requried data format. If createTestset == True, will create
        testset.
        """
        if self.__has_test: # Ceate both training set and test set
            # determine number of folds
            num_folds = self.__folds.shape[1]
            # use last folds,i.e.last column's valudate row as final test set.
            train_row_index = np.where(self.__folds.iloc[:,num_folds-1]!=1)[0]
            test_row_index = np.where(self.__folds.iloc[:,num_folds-1]==1)[0]
            self.__train_folds = self.__folds.iloc[train_row_index]
            self.__train_folds = self.__train_folds.iloc[:,0:num_folds-1]
            self.__num_train_fold = self.__train_folds.shape[1]
            # split all data into train and test
            self.__train_x = self.__all_x[train_row_index]
            self.__train_label = self.__label[train_row_index]
            self.__test_x = self.__all_x[test_row_index]
            self.__test_label = self.__label[test_row_index]
            self.__collect_dtrain = []
            self.__collect_dtest = []
            # convert to xgboost data format
            for i in range(self.__num_train_fold):
                dtrain = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__train_x[np.array(self.__train_folds.iloc[:,i]==0)])),
                                        label=self.__train_label[np.array(self.__train_folds.iloc[:,i]==0)])
                dvalidate = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__train_x[np.array(self.__train_folds.iloc[:,i]==1)])),
                                        label=self.__train_label[np.array(self.__train_folds.iloc[:,i]==1)])
                self.__collect_dtrain.append((dtrain,dvalidate))

            dtest = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__test_x)),
                                    label=self.__test_label)
            self.__collect_dtest.append(dtest)
        else: # only create training set. Treat whole data as training data.
            self.__train_folds = self.__folds
            self.__num_train_fold = self.__train_folds.shape[1]
            self.__train_x = self.__all_x
            self.__train_label = self.__label
            self.__collect_dtrain = []
            for i in range(self.__num_train_fold):
                dtrain = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__train_x[np.array(self.__train_folds.iloc[:,i]==0)])),label=self.__train_label[np.array(self.__train_folds.iloc[:,i]==0)])
                #xgb.DMatrix.save_binary(dtrain,"./xgb_data/dtrain_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                dvalidate = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__train_x[np.array(self.__train_folds.iloc[:,i]==1)])), label=self.__train_label[np.array(self.__train_folds.iloc[:,i]==1)])
                #xgb.DMatrix.save_binary(dvalidate,"./xgb_data/dvalidate_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                self.__collect_dtrain.append((dtrain,dvalidate))

    def numberOfTrainFold(self):
        """
        Return number of training fold
        """
        return self.__num_train_fold

    def get_dtrain(self,which_fold):
        """
        Return a list, containing tuples, each tuple contains training and
        validating data in xgboost data format.
        """
        if not isinstance(self.__collect_dtrain,list):
            raise ValueError('You must call `build` before `get_dtrain`')
        return self.__collect_dtrain[which_fold]

    def get_dtest(self):
        """
        Return a tuple, containing testing data in xgboost data format.
        """
        if not isinstance(self.__collect_dtest,list):
            raise ValueError('You must call `build` before `get_dtest`')
        return self.__collect_dtest[0]

    def get_train_fold(self):
        """
        Return a DataFrame containig training folds.
        """
        if not isinstance(self.__train_folds,pd.DataFrame):
            raise ValueError('You must call `build` before `get_train_fold`')
        return self.__train_folds

    def get_holdoutLabel(self):
        """
        Return an array containing training label
        """
        if not isinstance(self.__train_label,np.ndarray):
            raise ValueError('You must call `build` before `get_holdoutLabel`')
        return self.__train_label
