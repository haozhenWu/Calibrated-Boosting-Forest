"""
Create cross validation fold
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


class fold(object):
    """
    Lightchem's fold object
    """
    def __init__(self,X_data,y_data,k,seed = 2016):
        """Fold object
        Parameters:
        -----------
        X_data: numpy.ndarray
          Training features
        y_data: numpy.ndarray
          Label(Response variable)
        seed: int
          Control randomness
        """
        self.__num_row = X_data.shape[0]
        self.__label = y_data
        self.__num_fold = k
        self.__seed = seed
    # Generate kfold index according processed data's dimension and binary label.
    def generate_skfolds(self):
        '''
        Return k-fold index in DataFrame format. Each column is one
        fold, where value = 1 stands for test row,
        0 stands for training row.
        '''
        X = range(self.__num_row)
        y = pd.Series(self.__label)
        skf = StratifiedKFold(n_splits=self.__num_fold,
                                shuffle = True, random_state = self.__seed)
        train_index_list = list()
        test_index_list = list()
        for train_index, test_index in skf.split(X, y):
            train_index_list.append(train_index)
            test_index_list.append(test_index)
        folds = pd.DataFrame(np.zeros((self.__num_row,self.__num_fold)),
                             index = range(self.__num_row),
                             columns=["fold" + str(i) for i in xrange(1,self.__num_fold+1)])
        for i in range(folds.shape[1]):
            folds.iloc[test_index_list[i],i] = 1
        return folds
