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
            self.file_path = file_path
            self.label_name = lable_name
            self.X_data = None
            self.y_data = None
        def read(self):
            data_pd = pd.read_csv(self.file_path)
            self.y_data = data_pd[self.label_name]
            self.y_data = self.y_data.astype(np.float64)

            if 'fingerprint' in data_pd.columns:
                self.X_data = []
                for raw_fps in data_pd['fingerprint']:
                    self.X_data.append(np.array(list(raw_fps)))
                self.X_data = np.array(self.X_data)

                self.X_data = self.X_data.astype(np.float64)
                self.y_data = np.array(self.y_data)
                self.y_data = self.y_data.astype(np.float64)
            else: #havnt test below else code. just write a frame first
                features_cols = [ col for col in data_pd.columns if 'Feature_' in col]
                self.X_data = data_pd[features_cols]
                self.X_data = np.array(self.X_Data)
                self.X_data = self.X_data.astype(np.float64)
                self.y_data = np.array(self.y_data)
                self.y_data = self.y_data.astype(np.float64)

        def features(self):
            if not isinstance(self.X_data, np.ndarray):
               raise ValueError('You must call `read` before `features`')
            else:
                return self.X_data

        def label(self):
            if not isinstance(self.y_data, np.ndarray):
               raise ValueError('You must call `read` before `label`')
            else:
                return self.y_data
