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
        self.folds = fold
        self.num_train_fold = None
        self.all_x = X_data
        self.label = y_data
        self.train_x = None
        self.train_label = None
        self.test_x = None
        self.test_label = None
        self.collect_dtrain = None
        self.collect_dtest = None
        self.has_test = createTestset


    def build(self):
        if self.has_test:
            # determine number of folds
            num_folds = self.folds.shape[1]
            # use last folds,i.e.last column's valudate row as final test set.
            train_row_index = np.where(self.folds.iloc[:,num_folds-1]!=1)[0]
            test_row_index = np.where(self.folds.iloc[:,num_folds-1]==1)[0]
            train_folds = self.folds.iloc[train_row_index]
            train_folds = train_folds.iloc[:,0:num_folds-1]
            self.num_train_fold = train_folds.shape[1]
            # split all data into train and test
            self.train_x = self.all_x[train_row_index]
            self.train_label = self.label[train_row_index]
            self.test_x = self.all_x[test_row_index]
            self.test_label = self.label[test_row_index]
            self.collect_dtrain = []
            self.collect_dtest = []
            for i in range(self.num_train_fold):
                dtrain = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.train_x[np.array(train_folds.iloc[:,i]==0)])),label=self.train_label[np.array(train_folds.iloc[:,i]==0)])
                #xgb.DMatrix.save_binary(dtrain,"./xgb_data/dtrain_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                dvalidate = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.train_x[np.array(train_folds.iloc[:,i]==1)])),label=self.train_label[np.array(train_folds.iloc[:,i]==1)])
                #xgb.DMatrix.save_binary(dvalidate,"./xgb_data/dvalidate_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                self.collect_dtrain.append((dtrain,dvalidate))

            dtest = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.test_x)),label=self.test_label)
            self.collect_dtest.append(dtest)
        else:
            self.num_train_fold = self.folds.shape[1]
            self.train_x = self.all_x
            self.train_label = self.label
            self.collect_dtrain = []
            for i in range(self.num_train_fold):
                dtrain = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.train_x[np.array(self.folds.iloc[:,i]==0)])),label=self.train_label[np.array(self.folds.iloc[:,i]==0)])
                #xgb.DMatrix.save_binary(dtrain,"./xgb_data/dtrain_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                dvalidate = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(self.train_x[np.array(self.folds.iloc[:,i]==1)])), label=self.train_label[np.array(self.folds.iloc[:,i]==1)])
                #xgb.DMatrix.save_binary(dvalidate,"./xgb_data/dvalidate_" + str(TARGET_NAME) + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
                self.collect_dtrain.append((dtrain,dvalidate))

    def numberOfTrainFold(self):
        return self.num_train_fold

    def get_dtrain(self,which_fold):
        if not isinstance(self.collect_dtrain,list):
            raise ValueError('You must call `build` before `get_dtrain`')
        return self.collect_dtrain[which_fold]

    def get_dtest(self):
        if not isinstance(self.collect_dtest,list):
            raise ValueError('You must call `build` before `get_dtest`')
        return self.collect_dtest[0]
