import pandas as pd
import numpy as np
import xgboost as xgb
import scipy
import os
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import glob
import re
import xgb_eval

class firstLayerModel(object):
    def __init__(self,xgbData,eval_name,model_type):
        self.__DEFINED_MODEL_TYPE = ['GbtreeLogistic','GbtreeRegression','GblinearLogistic','GblinearRegression']
        self.__DEFINED_EVAL = ['ROCAUC','PRAUC','EFR1','EFR015']
        self.__xgbData = xgbData
        assert eval_name in self.__DEFINED_EVAL
        self.__eval_name = eval_name
        assert model_type in self.__DEFINED_MODEL_TYPE
        self.__model_type_writeout = model_type
        self.__collect_model = None
        self.__track_best_ntree = pd.DataFrame(columns = ['model_name','best_ntree'])
        self.__best_score = list()
        self.__param = {}
        self.__eval_function = None
        self.__MAXIMIZE = None
        self.__STOPPING_ROUND = None
        self.__holdout = None
        self.__default_param()

    def __default_param(self):
        match = {'ROCAUC' : [xgb_eval.evalrocauc,True,100],
                'PRAUC' :   [xgb_eval.evalprauc,True,300],
                'EFR1' : [xgb_eval.evalefr1,True,50],
                'EFR015' : [xgb_eval.evalefr015,True,50]}
        self.__eval_function = match[self.__eval_name][0]
        self.__MAXIMIZE = match[self.__eval_name][1]
        self.__STOPPING_ROUND = match[self.__eval_name][2]

        if self.__model_type_writeout == 'GbtreeLogistic':
            # define model parameter
            self.__param = {'objective':'binary:logistic',
                'booster' : 'gbtree',
                'eta' : 0.1,
                'max_depth' : 6,
                'subsample' : 0.53, # change from 0.83
                'colsample_bytree' : 0.7, # change from 0.8
                'num_parallel_tree' : 1,
                'min_child_weight' : 5,
                'gamma' : 5,
                'max_delta_step':1,
                'silent':1,
                'seed' : 2016
                }
        elif self.__model_type_writeout == 'GblinearLogistic':
             # define model parameter
             self.__param = {'objective':'binary:logistic',
                     'booster' : 'gblinear',
                     'eta' : 0.2,
                     'lambda' : 0.1,
                     'alpha' : 0.001,
                     'silent':1,
                     'seed' : 2016
                    }
        elif self.__model_type_writeout == 'GbtreeRegression':
             # define model parameter
             self.__param = {'objective':'reg:linear',
                     'booster' : 'gbtree',
                     'eta' : 0.2,
                     'max_depth' : 6,
                     'subsample' : 0.53, # change from 0.83
                     'colsample_bytree' : 0.7, # change from 0.8
                     'num_parallel_tree' : 1,
                     'min_child_weight' : 5,
                     'gamma' : 5,
                     'max_delta_step':1,
                     'silent':1,
                     'seed' : 2016
                    }
        elif self.__model_type_writeout == 'GblinearRegression':
             # define model parameter
             self.__param = {'objective':'reg:linear',
                     'booster' : 'gblinear',
                     'eta' : 0.2,
                     'lambda' : 0.1,
                     'alpha' : 0.001,
                     'silent':1,
                     'seed' : 2016
                     }




    def xgb_cv(self):
        '''Self-define wrapper to perform xgb.cv, which can use pre-difined cv folds,not sklearn's kfolds. Train k seperate model each use k-1 of k folds data. Later when do prediction, use the mean of k models' predictions.'''

        self.__collect_model = []
        num_folds = self.__xgbData.numberOfTrainFold()
        for i in range(num_folds):
            # load xgb data for a specific target (TARGET_NAME) and 1 fold
            dtrain = self.__xgbData.get_dtrain(i)[0]
            dvalidate = self.__xgbData.get_dtrain(i)[1]
            # prepare watchlist for model training
            watchlist  = [(dtrain,'train'),(dvalidate,'eval')]

            # Since when doing prediction, ntree limit not available for gblinear, use different training method for gbtree and gblinear
            if self.__param['booster'] == 'gbtree':
                if self.__param['objective'] == 'binary:logistic':
                    self.__param['scale_pos_weight'] = sum(dtrain.get_label()==0)/sum(dtrain.get_label()==1)

               # model training
                bst = xgb.train( self.__param, dtrain, 1000 , watchlist,
                                 feval = self.__eval_function,
                                 early_stopping_rounds = self.__STOPPING_ROUND,
                                 maximize = self.__MAXIMIZE,
                                 callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
               # save model
                self.__collect_model.append(bst)
                # save best number of tree. Later when do prediction, use best ntree, not the last tree
                # read previous cross validation result and append the target's result to it
                #track_best_ntree = pd.read_csv("./xgb_param/All_models_best_ntree.csv")
                # if the model name appered before, update its best ntree, o.w. add new model
                ind_model_result = pd.DataFrame({'model_name' : 'Part' + str(i),
                                                 'best_ntree' : bst.best_ntree_limit},
                                                 index = ['Part' + str(i)])
                self.__track_best_ntree = self.__track_best_ntree.append(ind_model_result)

            elif self.__param['booster'] == 'gblinear':
                # model training
                bst = xgb.train(self.__param, dtrain,300 , watchlist,
                                feval = self.__eval_function,
                                early_stopping_rounds = self.__STOPPING_ROUND,
                                maximize = self.__MAXIMIZE,
                                callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
                # retrain model using best ntree
                temp_best_ntree = bst.best_ntree_limit
                bst = xgb.train(self.__param, dtrain,temp_best_ntree, watchlist,
                                feval = self.__eval_function,
                                early_stopping_rounds = self.__STOPPING_ROUND,
                                maximize = self.__MAXIMIZE,
                                callbacks = [xgb.callback.print_evaluation(show_stdv=True)])
                self.__collect_model.append(bst)

            self.__best_score.append(bst.best_score)

    def generate_holdout_pred(self):
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
                best_ntree = self.__track_best_ntree.loc['Part' + str(i),'best_ntree']
                temp = bst.predict(dvalidate,ntree_limit = np.int64(np.float32(best_ntree)))
            else:
                temp = bst.predict(dvalidate)
            self.__holdout[np.where(train_folds.iloc[:,i]==1)] = temp

    def get_holdout(self):
        if not isinstance(self.__holdout,np.ndarray):
            raise ValueError('You must call `generate_holdout_pred` before `get_holdout`')
        return self.__holdout

    def get_holdoutLabel(self):
        return self.__xgbData.get_holdoutLabel()

    def cv_score(self):
        print 'Evaluation metric: ' + self.__eval_name
        print "CV result mean: " + str(np.mean(self.__best_score))
        print "CV result std: " + str(np.std(self.__best_score))


    def get_param(self):
        return self.__param, self.__MAXIMIZE, self.__STOPPING_ROUND

    def update_param(self,new_param,maximize,stopping_round):
        self.__param = new_param
        self.__MAXIMIZE = maximize
        self.__STOPPING_ROUND = stopping_round
