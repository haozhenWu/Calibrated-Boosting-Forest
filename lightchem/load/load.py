"""
Read data and transform to ndarray
"""

import pandas as pd
import numpy as np


class readData(object):
        '''
        Class to load data
        '''
        def __init__(self,file_path,lable_name):
            self.__file_path = file_path
            self.__label_name = lable_name
            self.__X_data = None
            self.__y_data = None
        def read(self):
            data_pd = pd.read_csv(self.__file_path)
            self.__y_data = data_pd[self.__label_name]
            self.__y_data = self.__y_data.astype(np.float64)

            if 'fingerprint' in data_pd.columns:
                self.__X_data = []
                for raw_fps in data_pd['fingerprint']:
                    self.__X_data.append(np.array(list(raw_fps)))
                self.__X_data = np.array(self.__X_data)

                self.__X_data = self.__X_data.astype(np.float64)
                self.__y_data = np.array(self.__y_data)
                self.__y_data = self.__y_data.astype(np.float64)
            else: #havnt test below else code. just write a frame first
                features_cols = [ col for col in data_pd.columns if 'Feature_' in col]
                self.__X_data = data_pd[features_cols]
                self.__X_data = np.array(self.__X_Data)
                self.__X_data = self.__X_data.astype(np.float64)
                self.__y_data = np.array(self.__y_data)
                self.__y_data = self.__y_data.astype(np.float64)

        def features(self):
            if not isinstance(self.__X_data, np.ndarray):
               raise ValueError('You must call `read` before `features`')
            else:
                return self.__X_data

        def label(self):
            if not isinstance(self.__y_data, np.ndarray):
               raise ValueError('You must call `read` before `label`')
            else:
                return self.__y_data
