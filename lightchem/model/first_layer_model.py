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

#DEFINED_LABEL_NAME = ['true_label_t50','Pria_SSB_%INH']
DEFINED_MODEL_TYPE = ['GbtreeLogistic','GbtreeRegression','GblinearLogistic','GblinearRegression']
DEFINED_EVAL = ['ROCAUC','PRAUC','EFR1','EFR015']
class firstLayerModel(object):
    def __init__(self,xgbData,eval_name,model_type):
        self.DEFINED_MODEL_TYPE = ['GbtreeLogistic','GbtreeRegression','GblinearLogistic','GblinearRegression']
        self.DEFINED_EVAL = ['ROCAUC','PRAUC','EFR1','EFR015']
        self.xgbData = xgbData
        assert eval_name in self.DEFINED_EVAL
        self.eval_name = eval_name
        assert model_type in self.DEFINED_MODEL_TYPE
        self.model_type_writeout = model_type
        self.collect_model = []
        self.track_best_ntree = pd.DataFrame(columns = ['model_name','best_ntree'])
        self.best_score = list()

    def xgb_cv(self):
        '''Self-define wrapper to perform xgb.cv, which can use pre-difined cv folds,not sklearn's kfolds. Train k seperate model each use k-1 of k folds data. Later when do prediction, use the mean of k models' predictions.'''
        # find number of folds User choosed
        num_folds = self.xgbData.numberOfTrainFold()
        for i in range(num_folds):
            # load xgb data for a specific target (TARGET_NAME) and 1 fold
            dtrain = self.xgbData.get_dtrain(i)[0]
            dvalidate = self.xgbData.get_dtrain(i)[1]
            #dtrain = xgb.DMatrix("./xgb_data/dtrain_" + TARGET_NAME + "_" + str(feature_name_writeout)+ "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")
            #dvalidate = xgb.DMatrix("./xgb_data/dvalidate_" + TARGET_NAME + "_" + str(feature_name_writeout) + "_" + str(label_name_writeout) + "_fold" + str(i) + "_v1.buffer")

            # prepare watchlist for model training
            watchlist  = [(dtrain,'train'),(dvalidate,'eval')]
            # match eval_name to associated eval function
            match = {'ROCAUC' : [xgb_eval.evalrocauc,True,100],
                    'PRAUC' :   [xgb_eval.evalprauc,True,300],
                    'EFR1' : [xgb_eval.evalefr1,True,50],
                    'EFR015' : [xgb_eval.evalefr015,True,50]}
            eval_function = match[self.eval_name][0]
            MAXIMIZE = match[self.eval_name][1]
            STOPPING_ROUND = match[self.eval_name][2]
            # Define model parameter
            if self.model_type_writeout == 'GbtreeLogistic':
                # define model parameter
                param = {'objective':'binary:logistic',
                    'booster' : 'gbtree',
                    'eta' : 0.1,
                    'max_depth' : 6,
                    'subsample' : 0.53, # change from 0.83
                    'colsample_bytree' : 0.7, # change from 0.8
                    'num_parallel_tree' : 1,
                    'min_child_weight' : 5,
                    'gamma' : 5,
                    'max_delta_step':1,
                    'scale_pos_weight':sum(dtrain.get_label()==0)/sum(dtrain.get_label()==1),
                    'silent':1,
                    'seed' : 2016
                    #'eval_metric': ['auc'],
                    }
            elif self.model_type_writeout == 'GblinearLogistic':
                 # define model parameter
                 param = {'objective':'binary:logistic',
                         'booster' : 'gblinear',
                         'eta' : 0.2,
                         'lambda' : 0.1,
                         'alpha' : 0.001,
                         'silent':1,
                         'seed' : 2016
                        }
            elif self.model_type_writeout == 'GbtreeRegression':
                 # define model parameter
                 param = {'objective':'reg:linear',
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
            elif self.model_type_writeout == 'GblinearRegression':
                 # define model parameter
                 param = {'objective':'reg:linear',
                         'booster' : 'gblinear',
                         'eta' : 0.2,
                         'lambda' : 0.1,
                         'alpha' : 0.001,
                         'silent':1,
                         'seed' : 2016
                         }

            # Since when doing prediction, ntree limit not available for gblinear, use different training method for gbtree and gblinear
            if param['booster'] == 'gbtree':
               # model training
               bst = xgb.train( param, dtrain, 1000 ,watchlist, feval = eval_function,
                                early_stopping_rounds = STOPPING_ROUND,maximize = MAXIMIZE,
                                callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
               #model_name = "bst_layer1" + model_type_writeout + "_" + TARGET_NAME + "_" + str (feature_name_writeout) + "_" + str(label_name_writeout)+ "_Optimized" + eval_name + "_fold" + str(i) + "_v1.model"
               # save model
               #bst.save_model("./xgb_model/" + model_name)
               self.collect_model.append(bst)
               # save best number of tree. Later when do prediction, use best ntree, not the last tree
               # read previous cross validation result and append the target's result to it
               #track_best_ntree = pd.read_csv("./xgb_param/All_models_best_ntree.csv")
               # if the model name appered before, update its best ntree, o.w. add new model
               ind_model_result = pd.DataFrame({'model_name' : 'Part' + str(i),
                                                'best_ntree' : bst.best_ntree_limit},
                                                index = ['Part' + str(i)])
               self.track_best_ntree = self.track_best_ntree.append(ind_model_result)

            elif param['booster'] == 'gblinear':
                # model training
                bst = xgb.train( param, dtrain,300 ,watchlist, feval = eval_function,
                                early_stopping_rounds = STOPPING_ROUND,maximize = MAXIMIZE,
                                callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
                # retrain model using best ntree
                temp_best_ntree = bst.best_ntree_limit
                bst = xgb.train( param, dtrain,temp_best_ntree, watchlist,
                                feval = eval_function,
                                early_stopping_rounds = STOPPING_ROUND,maximize = MAXIMIZE,
                                callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
                #model_name = "bst_layer1" + model_type_writeout + "_" + TARGET_NAME + "_" + str (feature_name_writeout) + "_" + str(label_name_writeout)+ "_Optimized" + eval_name + "_fold" + str(i) + "_v1.model"
                # save model
                #bst.save_model("./xgb_model/" + model_name)
                self.collect_model.append(bst)
                # save best number of tree. Later when do prediction, use best ntree, not the last tree
                # read previous cross validation result and append the target's result to it
                #track_best_ntree = pd.read_csv("./xgb_param/All_models_best_ntree.csv")
                # if the model name appered before, update its best ntree, o.w. add new model

            self.best_score.append(bst.best_score)

    def cv_score(self):
        print 'Evaluation metric: ' + self.eval_name
        print "CV result mean: " + str(np.mean(self.best_score))
        print "CV result std: " + str(np.std(self.best_score))
