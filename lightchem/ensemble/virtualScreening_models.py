"""
A collection of pre-defined ensemble models for virtual screening.
"""

import numpy as np
import pandas as pd
from lightchem.load import load
from lightchem.fold import fold
from lightchem.data import xgb_data
from lightchem.eval import xgb_eval
from lightchem.eval import eval_testset
from lightchem.model import first_layer_model
from lightchem.model import second_layer_model
from lightchem.model import hyper_parameter
from lightchem.eval import defined_eval
from lightchem.utility import util
# For this specific model object, REQUIRED first label name always represent
# Binary label column, where value are 1 or 0.
class VsEnsembleModel(object):
    """
    Wrapper class to build default ensemble models structure
    """
    def __init__(self,training_info,eval_name,fold_info = 4,seed = 2016,verbose = False):
        """
        Parameters:
        ----------
        training_info: list
         List of tuple, where each tuple contains 2 items, first item is dataframe
         containing training data(required to have fingerprint column named 'fingerprint',
         and concatanate fingerprint in a string),
         second item is a list containing the one or more label name.
         The first label name of first tuple is the default label that will be
         used for both layer models.
         If multiple label names present, VsEnsembleModel will automatically build
         model based on each label and ensemble them together.
         EX: [(my_dataframe,['binary_label','continuous_label']),
                (my_dataframe2,['binary_label','continuous_label'])]
         For this specific model object, REQUIRED first label name always be
         Binary label name.
        eval_name: str
         Name of evaluation metric used to monitor and stop training process.
         Must in eval.defined_eval
        """
        self.__training_info = training_info
        self.__check_labelType()
        self.__eval_name = eval_name
        self.__setting_list = []
        self.seed  = seed
        self.__determine_fold(fold_info)
        self.__prepare_xgbdata_train()
        self.__layer1_model_list = []
        self.__layer2_model_list = []
        self.__best_model_result = None
        self.__best_model = None
        self.__verbose = verbose
        self.__test_data = None
        self.__all_model_result = None

    def __determine_fold(self, fold_info):
        if isinstance(fold_info, pd.DataFrame):
            self.__has_fold = True
            self.my_fold = fold_info
            self.__num_folds = fold_info.shape[1]
        elif isinstance(fold_info, int):
            self.__has_fold = False
            self.my_fold = None
            self.__num_folds = fold_info
        else:
            raise ValueError("'fold_info' should be either a DataFrame contains "
            "fold index or a single integer indicating number of fold to create")

    def __prepare_xgbdata_train(self):
        """
        Internal method to build required data(xgbData) objects
        """
        print 'Preparing data'
        #Based on how many unique number, automatically detect if label column
        # is binary or continuous.
        num_xgbData = 0
        for item in self.__training_info:
            temp_df = item[0]
            for column_name in item[1]:
                # if it is binary label, use models for binary label.
                if len(np.unique(temp_df[column_name])) == 2:
                    model_type_to_use = ['GbtreeLogistic','GblinearLogistic']
                    temp_labelType = 'binary'
                else:
                    model_type_to_use = ['GbtreeRegression','GblinearRegression']
                    temp_labelType = 'continuous'

                temp_data = load.readData(temp_df,column_name)
                temp_data.read()
                X_data = temp_data.features()
                y_data = temp_data.label()
                # Need to generate fold once, based on binary label
                if not self.__has_fold:
                    self.my_fold = fold.fold(X_data,y_data,self.__num_folds,self.seed)
                    self.my_fold = self.my_fold.generate_skfolds()
                    self.__has_fold = True
                data = xgb_data.xgbData(self.my_fold,X_data,y_data)
                data.build()
                temp_dataName = 'Number:' + str(num_xgbData) + " xgbData, " + 'labelType: ' + temp_labelType
                self.__setting_list.append({'data_name':temp_dataName,
                                            'model_type':model_type_to_use,
                                            'data':data})
                num_xgbData += 1

    def __prepare_xgbdata_test(self,testing_info):
        """
        Internal method to prepare test data.
        Since VsEnsembleModel will choose best model from first and second layer2
        model. If first layer model is the best, need to identify which data is
        used for that model. If it is second layer model, data used is just all
        the data we have. [df1,df2], where df is concatanated fp string.
        """
        # transform fp string into array
        list_test_x_array = []
        for item in testing_info:
            temp_df = item[0]
            temp_data = load.readData(temp_df)
            temp_data.read()
            X_data = temp_data.features()
            list_test_x_array.append(X_data)

        name = self.__best_model.name
        self.__test_data = []
        if 'layer2' in name:
            temp = []
            # retrive all the data
            for i,item in enumerate(self.__training_info):
                temp_data = list_test_x_array[i]
                for column_name in item[1]:
                    temp.append(temp_data)
            assert len(temp) == len(self.__setting_list)
            for i,data_dict in enumerate(self.__setting_list):
                temp_data = temp[i]
                for model_type in data_dict['model_type']:
                    self.__test_data.append(temp_data)
            assert len(self.__test_data) == len(self.__layer1_model_list)

        else: # find specific data for layer1 model
            #TODO Find a better way to use regular expression
            index = name.split('layer1_Number:')[1]
            index = index.split('xgbData')[0]
            index = np.int64(index.strip())
            #Use same loop procedure to find corresponding data.
            j = 0
            for i,item in enumerate(self.__training_info):
                for column_name in item[1]:
                    if j == index:
                        self.__test_data.append(list_test_x_array[i])
                    j += 1
            assert len(self.__test_data) == 1

    def __check_labelType(self):
        """
        Internal method to check whether label columns are numeric
        """
        for item in self.__training_info:
            temp_df = item[0]
            for name in item[1]:
                assert np.issubdtype(temp_df[name].dtype,np.number)

    def train(self):
        """
        Train the model. Train and check potential first and second layer models.
        """
        evaluation_metric_name = self.__eval_name
        print 'Building first layer models'
        #---------------------------------first layer models ----------
        for data_dict in self.__setting_list:
            for model_type in data_dict['model_type']:
                unique_name = 'layer1_' + data_dict['data_name'] + '_' + model_type + '_' + evaluation_metric_name
                model = first_layer_model.firstLayerModel(data_dict['data'],
                        evaluation_metric_name,model_type,unique_name)
                # Retrieve default parameter and change default seed.
                default_param,default_MAXIMIZE,default_STOPPING_ROUND = model.get_param()
                default_param['seed'] = self.seed
                if self.__verbose == True:
                    default_param['silent'] = 1
                elif self.__verbose == False:
                    default_param['verbose_eval'] = False
                model.update_param(default_param,default_MAXIMIZE,default_STOPPING_ROUND)
                model.xgb_cv()
                model.generate_holdout_pred()
                self.__layer1_model_list.append(model)

        #------------------------------------second layer models
        layer2_label_data = self.__setting_list[0]['data'] # layer1 data object containing the label for layer2 model
        layer2_modeltype = ['GbtreeLogistic','GblinearLogistic']
        layer2_evaluation_metric_name = [self.__eval_name]
        print 'Building second layer models'
        for evaluation_metric_name in layer2_evaluation_metric_name:
            for model_type in layer2_modeltype:
                unique_name = 'layer2' + '_' + model_type + '_' + evaluation_metric_name
                l2model = second_layer_model.secondLayerModel(layer2_label_data,self.__layer1_model_list,
                            evaluation_metric_name,model_type,unique_name)
                l2model.second_layer_data()
                # Retrieve default parameter and change default seed.
                default_param,default_MAXIMIZE,default_STOPPING_ROUND = l2model.get_param()
                default_param['seed'] = self.seed
                if self.__verbose == True:
                    default_param['silent'] = 0
                elif self.__verbose == False:
                    default_param['verbose_eval'] = False
                l2model.update_param(default_param,default_MAXIMIZE,default_STOPPING_ROUND)
                l2model.xgb_cv()
                self.__layer2_model_list.append(l2model)


        #------------------------------------ evaluate model performance on test data
        # prepare test data, retrive from layer1 data
        list_TestData = []
        for data_dict in self.__setting_list:
            for model_type in data_dict['model_type']:
                list_TestData.append(data_dict['data'].get_dtest())
        test_label = layer2_label_data.get_testLabel()
        test_result_list = []
        i = 0
        for evaluation_metric_name in layer2_evaluation_metric_name:
            for model_type in layer2_modeltype:
                test_result = eval_testset.eval_testset(self.__layer2_model_list[i],
                                                        list_TestData,test_label,
                                                        evaluation_metric_name)
                test_result_list.append(test_result)
                i += 1

        # merge cv and test result together. Calcuate the weighted average of
        # cv and test result for each model(layer1, layer2 model). Then use the best
        # model to predict.
        all_model = self.__layer1_model_list + self.__layer2_model_list
        result = []
        for model in all_model:
            result = result + [item for item in np.array(model.cv_score_df())[0]]
        # Retrieve corresponding name of cv result
        result_index = []
        for model in all_model:
            result_index.append(model.name)
        # create a dataframe
        cv_result = pd.DataFrame({'cv_result' : result},index = result_index)

        test_result = pd.concat(test_result_list,axis = 0,ignore_index=False)
        test_result = test_result.rename(columns = {self.__eval_name:'test_result'})
        #selet distinct row.
        test_result['temp_name'] = test_result.index
        test_result = test_result.drop_duplicates(['temp_name'])
        test_result = test_result.drop('temp_name',1)
        cv_test = pd.merge(cv_result,test_result,how='left',left_index=True,right_index=True)
        self.__num_folds = np.float64(self.__num_folds)
        cv_test['weighted_score'] = cv_test.cv_result * (self.__num_folds-1)/self.__num_folds + cv_test.test_result * (1/self.__num_folds)
        weighted_score = cv_test.cv_result * (self.__num_folds-1)/self.__num_folds + cv_test.test_result * (1/self.__num_folds)

        # Determine does current evaluation metric need to maximize or minimize
        eval_info = defined_eval.definedEvaluation()
        is_max = eval_info.is_maximize(self.__eval_name)
        if is_max:
            position = np.where(cv_test.weighted_score == cv_test.weighted_score.max())
            best_model_name = cv_test.weighted_score.iloc[position].index[0]
        else:
            position = np.where(cv_test.weighted_score == cv_test.weighted_score.min())
            best_model_name = cv_test.weighted_score.iloc[position].index[0]
        # find best model
        all_model_name = [model.name for model in all_model]
        model_position = all_model_name.index(best_model_name)
        self.__best_model = all_model[model_position]
        self.__best_model_result = pd.DataFrame(cv_test.loc[self.__best_model.name])
        self.__all_model_result = cv_test

    def training_result(self):
        if not isinstance(self.__best_model_result,pd.DataFrame):
            raise ValueError('You must call `train` before `training_result`')
        return self.__best_model_result

    def detail_result(self):
        """
        Get detail training and testing result for each models.
        """
        return self.__all_model_result

    def predict(self,list_test_x):
        """
        Use best model to predict on test data.
        Parameters:
        -----------
        list_test_x: list, storing xgboost.DMatrix/Pandas.DataFrame
            New test data
        """
        # prepare test data. If it is first layer model, need to retrive corresponding data.
        self.__prepare_xgbdata_test(list_test_x)
        pred = self.__best_model.predict(self.__test_data)
        return pred

    def get_validation_info(self):
        """
        Return validation info.
        """
        self.__best_model.generate_holdout_pred()
        return self.__best_model.get_validation_info()


#TODO: Future customized VsEnsembleModel_ should inherit from VsEnsembleModel.
# Read hyperparameters from a file, instead of changing inside the class.
class VsEnsembleModel_keck(object):
    """
    Wrapper class to build ensemble models structure for KECK dataset.
    """
    def __init__(self,training_info,eval_name,fold_info = 4,createTestset = True,
                    finalModel = None, num_gblinear = 1, num_gbtree = 1,
                    layer2_modeltype = ['GbtreeLogistic','GblinearLogistic'],
                    nthread = -1, seed = 2016,verbose = False):
        """
        Parameters:
        ----------
        training_info: list
          List of tuple, where each tuple contains 2 items, first item is dataframe
          containing training data(required to have fingerprint column named 'fingerprint',
          and concatanate fingerprint in a string),
          second item is a list containing the one or more label name.
          The first label name of first tuple is the default label that will be
          used for both layer models.
          If multiple label names present, VsEnsembleModel will automatically build
          model based on each label and ensemble them together.
          EX: [(my_dataframe,['binary_label','continuous_label']),
                    (my_dataframe2,['binary_label','continuous_label'])]
          For this specific model object, REQUIRED first label name always be
            Binary label name.
        eval_name: str
          Name of evaluation metric used to monitor and stop training process.
           Must in eval.defined_eval
        fold_info: int or DataFrame
          either a DataFrame contains fold index or a single integer indicating
          number of fold to create
        createTestset: logic
          Whether internally create a test set. Help selecting the best model
        finalModel: str
          Can be None, layer1, or layer2. Default is None
          If set to None, it will select the best model from layer1 and layer2 models.
          layer1: Only select best layer1 model as final model, even though it has
                  layer2 model ready.
          layer2: Only select best layer2 model as final model. Sometimes layer1
                  models perform better than layer2 models.
        num_gblinear: integer
          Number of hyper-parameter sets to generate for each first layer
            gblinear model.
        num_gbtree: integer
          Number of hyper-parameter sets to generate for each first layer
            gbtree model.
        """
        self.__training_info = training_info
        self.__check_labelType()
        self.__eval_name = eval_name
        self.__createTestset = createTestset
        self.__setting_list = []
        self.seed  = seed
        self.__determine_fold(fold_info)
        # If multiple labels present, first label type is the final one to use
        self.__final_labelType = None
        self.__prepare_xgbdata_train()
        self.__layer1_model_list = []
        self.__layer2_model_list = []
        self.__best_model_result = None
        self.__best_model = None
        self.__verbose = verbose
        self.__test_data = None
        self.__all_model_result = None
        self.set_final_model(finalModel)
        self.__model_has_finalLabel = None
        self.__num_gblinear = num_gblinear
        self.__num_gbtree = num_gbtree
        self.__layer2_modeltype = layer2_modeltype
        self.nthread = nthread

    def set_final_model(self, finalModel):
        if finalModel == None or finalModel == 'layer1' or finalModel == 'layer2':
            self.__finalModel = finalModel
        else:
            raise ValueError('finalModel should be `None`, `layer1` or `layer2`')
        # This if is used prevent calling self.__prepare_result when first
        # call __init__.
        if len(self.__layer2_model_list) >= 1:
            self.__prepare_result()

    def __determine_fold(self, fold_info):
        if isinstance(fold_info, pd.DataFrame):
            self.__has_fold = True
            self.my_fold = fold_info
            self.__num_folds = fold_info.shape[1]
        elif isinstance(fold_info, int):
            self.__has_fold = False
            self.my_fold = None
            self.__num_folds = fold_info
        else:
            raise ValueError("'fold_info' should be either a DataFrame contains "
            "fold index or a single integer indicating number of fold to create")

    def __prepare_xgbdata_train(self):
        """
        Internal method to build required data(xgbData) objects
        """
        print 'Preparing data'
        #Based on how many unique number, automatically detect if label column
        # is binary or continuous.
        num_xgbData = 0
        for item in self.__training_info:
            temp_df = item[0]
            for column_name in item[1]:
                # if it is binary label, use models for binary label.
                if len(np.unique(temp_df[column_name])) == 2:
                    model_type_to_use = ['GbtreeLogistic','GblinearLogistic']
                    temp_labelType = 'binary'
                else:
                    model_type_to_use = ['GbtreeRegression','GblinearRegression']
                    temp_labelType = 'continuous'
                if self.__final_labelType == None:
                    self.__final_labelType = temp_labelType
                temp_data = load.readData(temp_df,column_name)
                temp_data.read()
                X_data = temp_data.features()
                y_data = temp_data.label()
                # Need to generate fold once, based on binary label
                if not self.__has_fold:
                    self.my_fold = fold.fold(X_data,y_data,self.__num_folds,self.seed)
                    self.my_fold = self.my_fold.generate_skfolds()
                    self.__has_fold = True
                data = xgb_data.xgbData(self.my_fold,X_data,y_data,
                                        createTestset = self.__createTestset)
                data.build()
                temp_dataName = 'Number:' + str(num_xgbData) + " xgbData, " + 'labelType: ' + temp_labelType
                self.__setting_list.append({'data_name':temp_dataName,
                                            'model_type':model_type_to_use,
                                            'data':data})
                num_xgbData += 1

    def __prepare_xgbdata_test(self,testing_info):
        """
        Internal method to prepare test data.
        Since VsEnsembleModel will choose best model from first and second layer2
        model. If first layer model is the best, need to identify which data is
        used for that model. If it is second layer model, data used is just all
        the data we have. [df1,df2], where df is concatanated fp string.
        """
        # transform fp string into array
        list_test_x_array = []
        for item in testing_info:
            temp_df = item[0]
            temp_data = load.readData(temp_df)
            temp_data.read()
            X_data = temp_data.features()
            list_test_x_array.append(X_data)

        name = self.__best_model.name
        self.__test_data = []
        if 'layer2' in name:
            temp = []
            # retrive all the data. Use same loop logic as __prepare_xgbdata_train
            for i,item in enumerate(self.__training_info):
                temp_data = list_test_x_array[i]
                for column_name in item[1]:
                    temp.append(temp_data)
            assert len(temp) == len(self.__setting_list)
            # Use same loop logic as train
            for i,data_dict in enumerate(self.__setting_list):
                temp_data = temp[i]
                for model_type in data_dict['model_type']:
                    num_sets = 1
                    if 'tree' in model_type:
                        num_sets = self.__num_gbtree
                    elif 'linear' in model_type:
                        num_sets = self.__num_gblinear
                    for k in range(num_sets):
                        self.__test_data.append(temp_data)
            assert len(self.__test_data) == len(self.__layer1_model_list)

        else: # find specific data for layer1 model
            #TODO Find a better way to use regular expression
            index = name.split('layer1_Number:')[1]
            index = index.split('xgbData')[0]
            index = np.int64(index.strip())
            #Use same loop procedure to find corresponding data.
            j = 0
            for i,item in enumerate(self.__training_info):
                for column_name in item[1]:
                    if j == index:
                        self.__test_data.append(list_test_x_array[i])
                    j += 1
            assert len(self.__test_data) == 1

    def __check_labelType(self):
        """
        Internal method to check whether label columns are numeric
        """
        for item in self.__training_info:
            temp_df = item[0]
            for name in item[1]:
                assert np.issubdtype(temp_df[name].dtype,np.number)

    def train(self):
        """
        Train the model. Train and check potential first and second layer models.
        """
        evaluation_metric_name = self.__eval_name
        print 'Building first layer models'
        #---------------------------------first layer models ----------
        for data_dict in self.__setting_list:
            for model_type in data_dict['model_type']:
                num_sets = 1
                param_sets = {}
                if 'tree' in model_type:
                    num_sets = self.__num_gbtree
                    param_sets = hyper_parameter.paramGenerator(model_type,
                                                                num_sets, self.seed)
                elif 'linear' in model_type:
                    num_sets = self.__num_gblinear
                    param_sets = hyper_parameter.paramGenerator(model_type,
                                                                num_sets, self.seed)
                # Build model based on each hyper-parameter set
                for i in range(num_sets):
                    unique_name_p1 = 'layer1_' + data_dict['data_name'] + '_'
                    unique_name_p2 = model_type + '_' + evaluation_metric_name
                    unique_name_p3 = '_' + str(i)
                    unique_name = unique_name_p1 + unique_name_p2 + unique_name_p3
                    params = param_sets[i]
                    model = first_layer_model.firstLayerModel(data_dict['data'],
                            evaluation_metric_name,model_type,unique_name)
                    # Retrieve default parameter and change default seed.
                    default_param,default_MAXIMIZE,default_STOPPING_ROUND = model.get_param()
                    params['seed'] = self.seed
                    params['nthread'] = self.nthread
                    stopping_round = 200
                    model.update_param(params,default_MAXIMIZE,stopping_round)
                    model.xgb_cv()
                    model.generate_holdout_pred()
                    self.__layer1_model_list.append(model)

        #------------------------------------second layer models
        layer2_label_data = self.__setting_list[0]['data'] # layer1 data object containing the label for layer2 model
        layer2_modeltype = self.__layer2_modeltype
        layer2_evaluation_metric_name = [self.__eval_name]
        print 'Building second layer models'
        for evaluation_metric_name in layer2_evaluation_metric_name:
            for model_type in layer2_modeltype:
                num_sets = 1
                param_sets = {}
                if 'tree' in model_type:
                    num_sets = self.__num_gbtree
                    param_sets = hyper_parameter.paramGenerator(model_type,
                                                                num_sets, self.seed)
                elif 'linear' in model_type:
                    num_sets = self.__num_gblinear
                    param_sets = hyper_parameter.paramGenerator(model_type,
                                                                num_sets, self.seed)
                # Build model based on each hyper-parameter set
                for i in range(num_sets):
                    unique_name_p1 = 'layer2' + '_' + model_type + '_' + evaluation_metric_name
                    unique_name_p2 = "_" + str(i)
                    unique_name = unique_name_p1 + unique_name_p2
                    params = param_sets[i]
                    l2model = second_layer_model.secondLayerModel(layer2_label_data,self.__layer1_model_list,
                                evaluation_metric_name,model_type,unique_name)
                    l2model.second_layer_data()
                    # Retrieve default parameter and change default seed.
                    default_param,default_MAXIMIZE,default_STOPPING_ROUND = l2model.get_param()
                    params['seed'] = self.seed
                    params['nthread'] = self.nthread
                    if model_type == 'GbtreeLogistic':
                        params['eta'] = 0.06
                        default_STOPPING_ROUND = 500
                    elif model_type == 'GblinearLogistic':
                        params['eta'] = 0.1
                        default_STOPPING_ROUND = 300
                    elif model_type == 'GbtreeRegression':
                        params['eta'] = 0.06
                        default_STOPPING_ROUND = 500
                    elif model_type == 'GblinearRegression':
                        params['eta'] = 0.1
                        default_STOPPING_ROUND = 300

                    l2model.update_param(params,default_MAXIMIZE,default_STOPPING_ROUND)
                    l2model.xgb_cv()
                    self.__layer2_model_list.append(l2model)
        self.__prepare_result()


    def __prepare_result(self):
        # merge cv and test result together. Calcuate the weighted average of
        # cv and test result for each model(layer1, layer2 model). Then use the best
        # model to predict.
        all_model = self.__layer1_model_list + self.__layer2_model_list
        result = []
        for model in all_model:
            result = result + [item for item in np.array(model.cv_score_df())[0]]
        # Retrieve corresponding name of cv result
        result_index = []
        for model in all_model:
            result_index.append(model.name)
        # create a dataframe
        cv_result = pd.DataFrame({'cv_result' : result},index = result_index)

        #------------------------------------ evaluate model performance on test data
        # prepare test data, retrive from layer1 data
        if self.__createTestset:
            list_TestData = []
            layer2_modeltype = self.__layer2_modeltype
            for data_dict in self.__setting_list:
                for model_type in data_dict['model_type']:
                    list_TestData.append(data_dict['data'].get_dtest())
            test_label = layer2_label_data.get_testLabel()
            test_result_list = []
            i = 0
            for evaluation_metric_name in layer2_evaluation_metric_name:
                for model_type in layer2_modeltype:
                    test_result = eval_testset.eval_testset(self.__layer2_model_list[i],
                                                            list_TestData,test_label,
                                                            evaluation_metric_name)
                    test_result_list.append(test_result)
                    i += 1
            test_result = pd.concat(test_result_list,axis = 0,ignore_index=False)
            test_result = test_result.rename(columns = {self.__eval_name:'test_result'})
            #selet distinct row.
            test_result['temp_name'] = test_result.index
            test_result = test_result.drop_duplicates(['temp_name'])
            test_result = test_result.drop('temp_name',1)

            cv_test = pd.merge(cv_result,test_result,how='left',left_index=True,right_index=True)
            self.__num_folds = np.float64(self.__num_folds)
            cv_test['weighted_score'] = cv_test.cv_result * (self.__num_folds-1)/self.__num_folds + cv_test.test_result * (1/self.__num_folds)
            weighted_score = cv_test.cv_result * (self.__num_folds-1)/self.__num_folds + cv_test.test_result * (1/self.__num_folds)
        else:
            cv_test = cv_result
            cv_test['weighted_score'] = cv_result.cv_result

        # Based on user specific finalModel
        if self.__finalModel == None:
            final_cv_test = cv_test
        else:
            finalModel_names = [item for item in list(cv_test.index) if self.__finalModel in item]
            final_cv_test = cv_test.loc[finalModel_names]

        # Determine does current evaluation metric need to maximize or minimize
        eval_info = defined_eval.definedEvaluation()
        is_max = eval_info.is_maximize(self.__eval_name)
        if is_max:
            position = np.where(final_cv_test.weighted_score == final_cv_test.weighted_score.max())
            best_model_name = final_cv_test.weighted_score.iloc[position].index[0]
        else:
            position = np.where(final_cv_test.weighted_score == final_cv_test.weighted_score.min())
            best_model_name = final_cv_test.weighted_score.iloc[position].index[0]
        # find best model
        all_model_name = [model.name for model in all_model]
        model_position = all_model_name.index(best_model_name)
        self.__best_model = all_model[model_position]
        self.__best_model_result = pd.DataFrame(cv_test.loc[self.__best_model.name])
        self.__all_model_result = cv_test
        # Find model contains the final label
        if self.__final_labelType == 'binary':
            model_has_finalLabel = [item for item in list(cv_test.index) if 'Logistic' in item]
            model_position = all_model_name.index(model_has_finalLabel[0])
            self.__model_has_finalLabel = all_model[model_position]
        elif self.__final_labelType == 'continuous':
            model_has_finalLabel = [item for item in list(cv_test.index) if 'Regression' in item]
            model_position = all_model_name.index(model_has_finalLabel[0])
            self.__model_has_finalLabel = all_model[model_position]


    def training_result(self):
        if len(self.__layer1_model_list) == 0:
            raise ValueError('You must call `train` before `training_result`')
        return self.__best_model_result

    def detail_result(self):
        """
        Get detail training and testing result for each models.
        """
        return self.__all_model_result

    def predict(self,list_test_x):
        """
        Use best model to predict on test data.
        Parameters:
        -----------
        list_test_x: list, storing xgboost.DMatrix/Pandas.DataFrame
            New test data
        """
        # prepare test data. If it is first layer model, need to retrive corresponding data.
        self.__prepare_xgbdata_test(list_test_x)
        pred = self.__best_model.predict(self.__test_data)
        return pred

    def get_validation_info(self):
        """
        Return validation info.
        Because it has multiple models with different features and labels and the
        ultimate task can be either binary or continuous, when return this
        validation info, need to switch the label to the label of ultimate task.
        EX. If ultimate task is to predict binary score, and the current best
        model is trained based on continuous labe, change the label from binary
        to continuous while keeping the predicted score the same.
        """
        self.__best_model.generate_holdout_pred()
        self.__model_has_finalLabel.generate_holdout_pred()
        validation_info = self.__best_model.get_validation_info()
        temp = self.__model_has_finalLabel.get_validation_info()
        # Change labels of each fold to the label of ultimate task.
        for i,val in enumerate(validation_info):
            validation_info[i].label = temp[i].label
        return validation_info

    def variable_importance(self):
        """
        Return variable importance
        """
        return self.__best_model.variable_importance()
