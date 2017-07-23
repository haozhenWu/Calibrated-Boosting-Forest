"""
Wrapper class to build second layer model
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import scipy
import os
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import glob
import re
from lightchem.eval import xgb_eval
from lightchem.data import xgb_data
from lightchem.model import first_layer_model
from lightchem.eval import defined_eval
from lightchem.model import defined_model


class secondLayerModel(object):
    """
    second layer model object.
    """
    def __init__(self,xgbData,list_firstLayerModel,eval_name,model_type,model_name):
        """Use holdout(out of fold) predictions from several firstLayerModels as
        training features to train a secondLayerModel.(So called stacking model)
        Parameters:
        -----------
        xgbData: object
         contains the label you want to use in second layer model.
        list_firstLayerModel: list
         list contains firstLayerModel.
        eval_name: str
          Name of evaluation metric used to monitor training process. Must in
          pre-defined evaluation list.
        model_type: str
          Name of model type you want to use. Must in pre-defined model type list.
        model_name: str
          Unique name for this model.
        """
        self.name = model_name
        self.__preDefined_model = defined_model.definedModel()
        self.__preDefined_eval = defined_eval.definedEvaluation()
        self.__DEFINED_EVAL = self.__preDefined_eval.eval_list()
        self.__xgbData = xgbData
        assert all([isinstance(item,first_layer_model.firstLayerModel) for item in list_firstLayerModel])
        self.__list_firstLayerModel = list_firstLayerModel
        self.__preDefined_eval.validate_eval_name(eval_name)
        self.__eval_name = eval_name
        self.__preDefined_model.validate_model_type(model_type)
        self.__model_type_writeout = model_type
        self.__collect_model = None
        self.__track_best_ntree = pd.DataFrame(columns = ['model_name','best_ntree'])
        self.__best_score = list()
        self.__firstLayerModel_prediction = None
        self.__param = self.__preDefined_model.model_param(model_type)
        self.__eval_function = self.__preDefined_eval.eval_function(self.__eval_name)
        self.__MAXIMIZE = self.__preDefined_eval.is_maximize(self.__eval_name)
        self.__STOPPING_ROUND = self.__preDefined_eval.stopping_round(self.__eval_name)
        self.__holdout = None

    def second_layer_data(self):
        """
        Method to prepare training data for second layer model.
        """
        holdout_list = list()
        # Retrive first layer model's holdout prediction.
        for model in self.__list_firstLayerModel:
            holdout_list.append(model.get_holdout())
        holdout_df = pd.DataFrame(holdout_list).transpose()
        #Remove later: sort the column so that column index is always the same
        #holdout_df = holdout_df[np.sort(holdout_df.columns)]
        label = self.__xgbData.get_holdoutLabel()
        self.__xgbData = xgb_data.xgbData(self.__xgbData.get_train_fold(),
                                          np.array(holdout_df),
                                          np.array(label),
                                          False)
        self.__xgbData.build()

    def xgb_cv(self):
        '''
        Self-define wrapper to perform cross validation, which use training and
        validating data from xgbData to train k models where k = number of
        training folds.Later when do prediction, use the mean of k models'
        predictions.
        '''
        self.__collect_model = []
        num_folds = self.__xgbData.numberOfTrainFold()
        for i in range(num_folds):
            # load xgb data for one cross validation iteration.
            dtrain = self.__xgbData.get_dtrain(i)[0]
            dvalidate = self.__xgbData.get_dtrain(i)[1]
            # prepare watchlist for model training
            watchlist  = [(dtrain,'train'),(dvalidate,'eval')]

            # Since when doing prediction, ntree limit not available for
            # gblinear, use different training method for gbtree and gblinear
            if self.__param['booster'] == 'gbtree':
                if self.__param['objective'] == 'binary:logistic':
                    self.__param['scale_pos_weight'] = sum(dtrain.get_label()==0)/sum(dtrain.get_label()==1)
               # model training
                bst = xgb.train( self.__param, dtrain, 1000 , watchlist,
                                 feval = self.__eval_function,
                                 early_stopping_rounds = self.__STOPPING_ROUND,
                                 maximize = self.__MAXIMIZE
                                 #,callbacks=[xgb.callback.print_evaluation(show_stdv=True)]
                                 )
               # collect this model
                self.__collect_model.append(bst)
               # save best number of tree. Later when do prediction,
               # use best number of tree, not the last tree.
                ind_model_result = pd.DataFrame({'model_name' : 'Part' + str(i),
                                                 'best_ntree' : bst.best_ntree_limit},
                                                 index = ['Part' + str(i)])
                self.__track_best_ntree = self.__track_best_ntree.append(ind_model_result)

            elif self.__param['booster'] == 'gblinear':
                # model training
                bst = xgb.train(self.__param, dtrain,600 , watchlist,
                                feval = self.__eval_function,
                                early_stopping_rounds = self.__STOPPING_ROUND,
                                maximize = self.__MAXIMIZE
                                #,callbacks=[xgb.callback.print_evaluation(show_stdv=True)]
                                )
                # retrain model using best ntree
                temp_best_ntree = bst.best_ntree_limit
                bst = xgb.train(self.__param, dtrain,temp_best_ntree, watchlist,
                                feval = self.__eval_function,
                                early_stopping_rounds = self.__STOPPING_ROUND,
                                maximize = self.__MAXIMIZE
                                #,callbacks = [xgb.callback.print_evaluation(show_stdv=True)]
                                )
                self.__collect_model.append(bst)

            self.__best_score.append(bst.best_score)

    def generate_holdout_pred(self):
        """
        Method to generate holdout(out of fold) predictions.
        """
        if not isinstance(self.__collect_model,list):
            raise ValueError('You must call `xgb_cv` before `generate_holdout_pred`')
        # find number of folds User choosed
        num_folds = self.__xgbData.numberOfTrainFold()
        train_folds = self.__xgbData.get_train_fold()
        self.__holdout = np.zeros(train_folds.shape[0])
        for i in range(num_folds):
            bst = self.__collect_model[i]
            dvalidate = self.__xgbData.get_dtrain(i)[1]
            if self.__param['booster'] == 'gbtree':
                # Retrive saved best number of tree.
                best_ntree = self.__track_best_ntree.loc['Part' + str(i),'best_ntree']
                temp = bst.predict(dvalidate,ntree_limit = np.int64(np.float32(best_ntree)))
            else:
                temp = bst.predict(dvalidate)
            self.__holdout[np.where(train_folds.iloc[:,i]==1)] = temp

    def predict(self,list_test_x):
        """
        Method to predict new data. Return an a np.ndarray containing prediction
        Parameters:
        -----------
        list_test_x: list, storing xgboost.DMatrix/pandas.DataFrame
          List containing new test data for each firstLayerModel that passed into
          secondLayerModel. NOTE: Must make sure the order of new testset of
          each model in the list must be the SAME as the order of first layer
          models that pass into secondLayerModel
        """
        if not isinstance(self.__collect_model,list):
            raise ValueError('You must call `xgb_cv` before `predict`')
        # Convert test data into xgboost.DMatrix format
        for j,item in enumerate(list_test_x):
            if not isinstance(item,xgb.DMatrix):
                list_test_x[j] = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(item)))
            else:
                list_test_x[j] = item
        # Generate firstLayerModel predictions using new test dataset.
        test_x = []
        for j,model in enumerate(self.__list_firstLayerModel):
            test_x.append(model.predict([list_test_x[j]]))
        test_x = pd.DataFrame(test_x).transpose()
        firstLayerModel_names = [model.name for model in self.__list_firstLayerModel]
        self.__firstLayerModel_prediction = test_x
        self.__firstLayerModel_prediction.columns = firstLayerModel_names
        test_x = xgb.DMatrix(scipy.sparse.csr_matrix(np.array(test_x)))

        # find number of folds User choosed
        num_folds = self.__xgbData.numberOfTrainFold()
        predictions = []
        for i in range(num_folds):
            # Find model trained on ith cv iteration and its validation set.
            bst = self.__collect_model[i]
            if self.__param['booster'] == 'gbtree':
                # Retrive saved best number of tree.
                best_ntree = self.__track_best_ntree.loc['Part' + str(i),'best_ntree']
                temp = bst.predict(test_x,ntree_limit = np.int64(np.float32(best_ntree)))
            else:
                temp = bst.predict(test_x)
            predictions.append(temp)
        pred_df = pd.DataFrame(predictions)
        pred_mean = np.array(pred_df.mean())
        return pred_mean

    def get_holdout(self):
        """
        Return generated holdout(out of fold) prediction.
        """
        if not isinstance(self.__holdout,np.ndarray):
            raise ValueError('You must call `generate_holdout_pred` before `get_holdout`')
        return self.__holdout

    def get_holdoutLabel(self):
        """
        Return holdout(out of fold) label.
        """
        return self.__xgbData.get_holdoutLabel()

    def cv_score(self):
        """
        Print model's cross validation score.
        """
        print 'Evaluation metric: ' + self.__eval_name
        print 'Model name: ' + self.__model_type_writeout
        print "CV result mean: " + str(np.mean(self.__best_score))
        print "CV result std: " + str(np.std(self.__best_score))

    def cv_score_df(self):
        """
        return cv score as dataframe
        """
        return pd.DataFrame({self.__eval_name : [np.mean(self.__best_score),
                                                np.std(self.__best_score)]},
                            index = [self.name+"_mean",self.name+"_std"])

    def get_param(self):
        """
        Return 3 items, parameter used for model, whether to maximize the
        evaluation metric, number of stopping round.
        """
        return self.__param, self.__MAXIMIZE, self.__STOPPING_ROUND

    def update_param(self,new_param,maximize,stopping_round):
        """
        Allow user specific parameters.
        """
        self.__param = new_param
        self.__MAXIMIZE = maximize
        self.__STOPPING_ROUND = stopping_round

    def get_firstLayerModel_predictions(self):
        """
        Return firstLayerModel_predictions as a pd.DataFrame
        """
        if not isinstance(self.__firstLayerModel_prediction,pd.DataFrame):
            raise ValueError('You must call `predict` before `get_firstLayerModel_predictions`')
        return self.__firstLayerModel_prediction

    def custom_eval(self,function):
        """
        Allow user to pass custom evaluation function. Sometime we can train a
        model with continuous label and evaluate on classification based
        evaluation function. We just need to internally convert continuous label
        into binary label.
        Parameters:
        -----------
        function: function
          Custom evaluation function based on xgboost's format.
        """
        self.__eval_function = function

    def get_validation_info(self):
        """
        Return validation data info as a list, where the length is total number
        of training fold, each item is a pd.DataFrame with two
        columns(validation_pred, validation_label)
        """
        if not isinstance(self.__holdout,np.ndarray):
            raise ValueError('You must call `generate_holdout_pred` ',
                             'before `get_validation_info`')
        train_folds = self.__xgbData.get_train_fold()
        train_labels = self.__xgbData.get_holdoutLabel()
        final = train_folds.copy(deep=True)
        final.loc[:,'label'] = train_labels
        final.loc[:,'validation_pred'] = self.__holdout
        val_info = []
        for fold in list(train_folds.columns):
            temp = final.loc[final.loc[:,fold]==1,['label','validation_pred']]
            val_info.append(temp)
        return val_info

    def variable_importance(self):
        """
        Return average variable importance based on `gain`, `cover`, `weight`
        as a pd.DataFrame. Sort by 'weight'.
        """
        firstLayerModel_names = [model.name for model in self.__list_firstLayerModel]
        if not ininstance(self.__collect_model, list):
            raise ValueError('You must call `xgb_cv` ',
                             'before `variable_importance`')
        if self.__param['booster'] == 'gblinear':
            # Currently can only get feature importance from tree booster.
            imp_all = pd.DataFrame()
        else if self.__param['booster'] == 'gbtree':                
            name = firstLayerModel_names
            nfold = len(self.__collect_model)
            has_imp = False
            for i, model in enumerate(self.__collect_model):
                imp = pd.DataFrame({"name":name,
                                    "weight":np.repeat(0, len(name)),
                                    "gain":np.repeat(0, len(name)),
                                    "cover":np.repeat(0, len(name))})
                for importance_type in ["weight", "gain", "cover"]:
                    imp_score = model.get_score(importance_type=importance_type)
                    for key in imp_score.iterkeys():
                        pos = np.int(re.split("f", key)[1])
                        imp.loc[pos, importance_type] = imp_score[key]
                if has_imp:
                    imp_all = pd.concat([imp_all, imp], axis=0)
                else:
                    imp_all = imp
                    has_imp = True
            imp_all = imp_all.groupby("name").mean()
            imp_all = imp_all.sort_values("weight", ascending=True)
        return imp_all
