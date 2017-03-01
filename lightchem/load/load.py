"""
Contains helper class to read data and transform to ndarray
"""

import pandas as pd
import numpy as np
from lightchem.utility import util

class readData(object):
        '''
        Class to read data,such as fingerprint stored as string in one column
        or column names starting with `Feature_`, and transform to ndarray.
        '''
        def __init__(self,data_loc,label_name):
            """
            Parameters:
            -----------
            data_loc: str/pandas.DataFrame
              if it is str, it is path to csv directory where data is stored.
              if it is pandas.DataFrame, it is an in memory DataFrame object.
            label_name: str
              Name of your label(Response) variable
            """
            assert (isinstance(data_loc,str) or isinstance(data_loc,pd.DataFrame))
            if isinstance(data_loc,pd.DataFrame):
                self.__data_pd = data_loc
            elif isinstance(data_loc,str):
                self.__data_pd = None
                self.__file_path = data_loc
            #self.__file_path = file_path
            assert isinstance(label_name,str)
            self.__label_name = label_name
            self.__X_data = None
            self.__y_data = None
        def read(self):
            """
            Read data
            """
            if not isinstance(self.__data_pd,pd.DataFrame):
                self.__data_pd = pd.read_csv(self.__file_path)
            #data_pd = pd.read_csv(self.__file_path)
            self.__y_data = self.__data_pd[self.__label_name]
            self.__y_data = self.__y_data.astype(np.float64)
            # extracting features from DataFrame
            if 'fingerprint' in self.__data_pd.columns:
                self.__X_data = util.fpString_to_array(self.__data_pd['fingerprint'])
                self.__y_data = np.array(self.__y_data)
                self.__y_data = self.__y_data.astype(np.float64)
            else: #havnt test below else code. just write a frame first
                features_cols = [ col for col in self.__data_pd.columns if 'Feature_' in col]
                self.__X_data = self.__data_pd[features_cols]
                self.__X_data = np.array(self.__X_Data)
                self.__X_data = self.__X_data.astype(np.float64)
                self.__y_data = np.array(self.__y_data)
                self.__y_data = self.__y_data.astype(np.float64)

        def features(self):
            """
            Method to return processed features data

            """
            if not isinstance(self.__X_data, np.ndarray):
               raise ValueError('You must call `read` before `features`')
            else:
                return self.__X_data

        def label(self):
            """
            Method to return processed label(response variable)

            """
            if not isinstance(self.__y_data, np.ndarray):
               raise ValueError('You must call `read` before `label`')
            else:
                return self.__y_data
