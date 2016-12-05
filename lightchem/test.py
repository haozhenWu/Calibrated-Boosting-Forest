import sys
sys.path.remove('/usr/lib/python2.7/dist-packages')
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/load/")
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/envir/")
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/fold/")
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/data/")
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/model/")
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/eval/")

import create_dir
import load
import fold
import xgb_data
import xgb_eval
import first_layer_model
import second_layer_model

a = load.readData('/home/haozhen/Haozhen-data/pcba128_python/data/pcba128_mmtn_canon_ecfp1024.csv','pcba-aid995')
a.read()
X_data = a.features()
y_data = a.label()

myfold = fold.fold(X_data,y_data,4)
myfold = myfold.generate_skfolds()

data = xgb_data.xgbData(myfold,X_data,y_data)
data.build()

#data2 = xgb_data.xgbData(myfold,X_data,y_data,False)
#data2.build()

model = first_layer_model.firstLayerModel(data,'ROCAUC','GbtreeLogistic')
#model = firstLayerModel(data,'EFR1','GbtreeLogistic')
model.xgb_cv()
model.generate_holdout_pred()

model2 = first_layer_model.firstLayerModel(data,'EFR1','GbtreeLogistic')
#model = firstLayerModel(data,'EFR1','GbtreeLogistic')
model2.xgb_cv()
model2.generate_holdout_pred()

model3 = first_layer_model.firstLayerModel(data,'ROCAUC','GblinearLogistic')
#model = firstLayerModel(data,'EFR1','GbtreeLogistic')
model3.xgb_cv()
model3.generate_holdout_pred()

model4 = first_layer_model.firstLayerModel(data,'EFR1','GblinearLogistic')
#model = firstLayerModel(data,'EFR1','GbtreeLogistic')
model4.xgb_cv()
model4.generate_holdout_pred()


list_l1model = [model,model2,model3,model4]
l2model = second_layer_model.secondLayerModel(data,list_l1model,'ROCAUC','GbtreeLogistic')
#l2model = secondLayerModel(data,list_l1model,'ROCAUC','GbtreeLogistic')
l2model.xgb_cv()


"""
Evaluation metric: ROCAUC
CV result mean: 0.805586333333
CV result std: 0.00597256044107
"""

#metrics.roc_auc_score( model.get_holdoutLabel(), model.get_holdout())



## temp

dtrain = data.get_dtrain(0)[0]
dvalidate = data.get_dtrain(0)[1]
eval_function = xgb_eval.evalrocauc
STOPPING_ROUND = 5
MAXIMIZE = False
watchlist  = [(dtrain,'train'),(dvalidate,'eval')]

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

bst = xgb.train( param, dtrain, 1000 , watchlist, feval = eval_function,
                maximize = MAXIMIZE,  early_stopping_rounds = STOPPING_ROUND,
                callbacks=[xgb.callback.print_evaluation(show_stdv=True)])






class temp(object):

    def m(self):
        value = 3

    def get_m(self):
        print value
