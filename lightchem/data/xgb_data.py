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

    def __init__(self,fold,X_data,y_data,createTestset = True):
        self.__folds = fold
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
        if self.__has_test:
            # determine number of folds
            num_folds = self.__folds.shape[1]
            # use last folds,i.e.last column's valudate row as final test set.
            train_row_index = np.where(self.__folds.iloc[:,num_folds-1]!=1)[0]
            test_row_index = np.where(self.__folds.iloc[:,num_folds-1]==1)[0]
            train_folds = self.__folds.iloc[train_row_index]
            train_folds = train_folds.iloc[:,0:num_folds-1]
            self.__num_train_fold = train_folds.shape[1]
            # split all data into train and test
            self.__train_x = self.__all_x[train_row_index]
            self.__train_label = self.__label[train_row_index]
            self.__test_x = self.__all_x[test_row_index]
            self.__test_label = self.__label[test_row_index]
            self.__collect_dtrain = []
            self.__collect_dtest = []
            for i in range(self.__num_train_fold):
                dtrain = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__train_x[np.array(train_folds.iloc[:,i]==0)])),label=self.train_label[np.array(train_folds.iloc[:,i]==0)])
                #xgb.DMatrix.save_binary(dtrain,"./xgb_data/dtrain_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                dvalidate = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__train_x[np.array(train_folds.iloc[:,i]==1)])),label=self.train_label[np.array(train_folds.iloc[:,i]==1)])
                #xgb.DMatrix.save_binary(dvalidate,"./xgb_data/dvalidate_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                self.__collect_dtrain.append((dtrain,dvalidate))

            dtest = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__test_x)),label=self.__test_label)
            self.__collect_dtest.append(dtest)
        else:
            self.__num_train_fold = self.__folds.shape[1]
            self.__train_x = self.__all_x
            self.__train_label = self.__label
            self.__collect_dtrain = []
            for i in range(self.__num_train_fold):
                dtrain = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__train_x[np.array(self.__folds.iloc[:,i]==0)])),label=self.train_label[np.array(self.__folds.iloc[:,i]==0)])
                #xgb.DMatrix.save_binary(dtrain,"./xgb_data/dtrain_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                dvalidate = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.__train_x[np.array(self.__folds.iloc[:,i]==1)])), label=self.train_label[np.array(self.__folds.iloc[:,i]==1)])
                #xgb.DMatrix.save_binary(dvalidate,"./xgb_data/dvalidate_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                self.__collect_dtrain.append((dtrain,dvalidate))

    def numberOfTrainFold(self):
        return self.__num_train_fold

    def get_dtrain(self,which_fold):
        if not isinstance(self.__collect_dtrain,list):
            raise ValueError('You must call `build` before `get_dtrain`')
        return self.__collect_dtrain[which_fold]

    def get_dtest(self):
        if not isinstance(self.__collect_dtest,list):
            raise ValueError('You must call `build` before `get_dtest`')
        return self.__collect_dtest[0]
