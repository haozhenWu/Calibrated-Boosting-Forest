import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


class fold(object):
    def __init__(self,X_data,y_data,k):
        self.num_row = X_data.shape[0]
        self.label = y_data
        self.num_fold = k
    # Generate kfold index according processed data's dimension and binary label.
    def generate_skfolds(self):
        '''Return k-fold index in DataFrame format. Each column is one fold, where 1 stands for
            test row, 0 stands for training row. '''
        X = range(self.num_row)
        y = pd.Series(self.label)
        skf = StratifiedKFold(n_splits=self.num_fold,shuffle = True)
        train_index_list = list()
        test_index_list = list()
        for train_index, test_index in skf.split(X, y):
            train_index_list.append(train_index)
            test_index_list.append(test_index)
        folds = pd.DataFrame(np.zeros((self.num_row,self.num_fold)),index = range(self.num_row),columns=["fold" + str(i) for i in xrange(1,self.num_fold+1)])
        for i in range(folds.shape[1]):
            folds.iloc[test_index_list[i],i] = 1 # validation rows marked as 1. Last column used as final test data
        return folds
