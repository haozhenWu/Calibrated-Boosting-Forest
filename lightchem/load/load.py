import pandas as pd
import numpy as np


class readData(object):
        '''Return training data as X_data, label data as y_data'''
        def __init__(self,file_path,lable_name):
            self.file_path = file_path
            self.label_name = lable_name
        def read(self):
            data_pd = pd.read_csv(self.file_path)
            y_data = data_pd[self.label_name]
            y_data = y_data.astype(np.float64)

            if 'fingerprint' in data_pd.columns:
                X_data = []
                for raw_fps in data_pd['fingerprint']:
                    X_data.append(np.array(list(raw_fps)))
                X_data = np.array(X_data)

                X_data = X_data.astype(np.float64)
                y_data = np.array(y_data)
                y_data = y_data.astype(np.float64)
            else: #havnt test below else code. just write a frame first
                features_cols = [ col for col in data_pd.columns if 'Feature_' in col]
                X_data = data_pd[features_cols]
                X_data = np.array(X_Data)
                X_data = X_data.astype(np.float64)
                y_data = np.array(y_data)
                y_data = y_data.astype(np.float64)

            return X_data, y_data
