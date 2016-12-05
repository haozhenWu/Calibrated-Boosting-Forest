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

# 14:12 start- 15:05. 53mins. Time might influenced by Spencer's job. 2 threads maybe.
# binary label
a = load.readData('/home/haozhen/Haozhen-data/pcba128_python/data/pcba128_mmtn_canon_ecfp1024.csv','pcba-aid411')
a.read()
X_data = a.features()
y_data = a.label()
myfold = fold.fold(X_data,y_data,4)
myfold = myfold.generate_skfolds()
data = xgb_data.xgbData(myfold,X_data,y_data)
data.build()

# continuous label
b = load.readData('/home/haozhen/Haozhen-data/pcba128_python/data/pcba128_canon_ecfp1024_logac50.csv','aid411_logAC50')
b.read()
X_data_b = b.features()
y_data_b = b.label()
datab = xgb_data.xgbData(myfold,X_data_b,y_data_b)
datab.build()

# model based on binary data
model = first_layer_model.firstLayerModel(data,'ROCAUC','GbtreeLogistic')
model.xgb_cv()
model.generate_holdout_pred()

model2 = first_layer_model.firstLayerModel(data,'EFR1','GbtreeLogistic')
model2.xgb_cv()
model2.generate_holdout_pred()

model3 = first_layer_model.firstLayerModel(data,'ROCAUC','GblinearLogistic')
model3.xgb_cv()
model3.generate_holdout_pred()

model4 = first_layer_model.firstLayerModel(data,'EFR1','GblinearLogistic')
model4.xgb_cv()
model4.generate_holdout_pred()

# model based on continuous data
modelb = first_layer_model.firstLayerModel(datab,'ROCAUC','GbtreeRegression')
modelb.xgb_cv()
modelb.generate_holdout_pred()

model2b = first_layer_model.firstLayerModel(datab,'EFR1','GbtreeRegression')
model2b.xgb_cv()
model2b.generate_holdout_pred()

model3b = first_layer_model.firstLayerModel(datab,'ROCAUC','GblinearRegression')
model3b.xgb_cv()
model3b.generate_holdout_pred()

model4b = first_layer_model.firstLayerModel(datab,'EFR1','GblinearRegression')
model4b.xgb_cv()
model4b.generate_holdout_pred()

# use label from binary data to train layer2 models
list_l1model = [model,model2,model3,model4,modelb,model2b,model3b,model4b]
# ROCAUC
l2model = second_layer_model.secondLayerModel(data,list_l1model,'ROCAUC','GbtreeLogistic')
l2model.second_layer_data()
l2model.xgb_cv()

l2model2 = second_layer_model.secondLayerModel(data,list_l1model,'ROCAUC','GblinearLogistic')
l2model2.second_layer_data()
l2model2.xgb_cv()
# EFR1
l2model3 = second_layer_model.secondLayerModel(data,list_l1model,'EFR1','GbtreeLogistic')
l2model3.second_layer_data()
l2model3.xgb_cv()

l2model4 = second_layer_model.secondLayerModel(data,list_l1model,'EFR1','GblinearLogistic')
l2model4.second_layer_data()
l2model4.xgb_cv()

model.cv_score()
model3.cv_score()
modelb.cv_score()
model3b.cv_score()
l2model.cv_score()
l2model2.cv_score()
"""
In [179]: model.cv_score()
Evaluation metric: ROCAUC
CV result mean: 0.902723
CV result std: 0.00144810381764

In [180]: model3.cv_score()
Evaluation metric: ROCAUC
CV result mean: 0.878986333333
CV result std: 0.00270194551306

In [181]: modelb.cv_score()
Evaluation metric: ROCAUC
CV result mean: 0.871077666667
CV result std: 0.0087030881237

In [182]: model3b.cv_score()
Evaluation metric: ROCAUC
CV result mean: 0.869077
CV result std: 0.00243051489744

In [183]: l2model.cv_score()
Evaluation metric: ROCAUC
CV result mean: 0.902115333333
CV result std: 0.00295798471185

In [184]: l2model2.cv_score()
Evaluation metric: ROCAUC
CV result mean: 0.903146333333
CV result std: 0.00165515846841
"""
model2.cv_score()
model4.cv_score()
model2b.cv_score()
model4b.cv_score()
l2model3.cv_score()
l2model4.cv_score()

"""
In [173]: model2.cv_score()
Evaluation metric: EFR1
CV result mean: 27.3657286667
CV result std: 0.552492711347

In [174]: model4.cv_score()
Evaluation metric: EFR1
CV result mean: 23.785166
CV result std: 1.99204101406

In [175]: model2b.cv_score()
Evaluation metric: EFR1
CV result mean: 29.6578756667
CV result std: 1.68795663542

In [176]: model4b.cv_score()
Evaluation metric: EFR1
CV result mean: 22.5007643333
CV result std: 0.59566238563

In [177]: l2model3.cv_score()
Evaluation metric: EFR1
CV result mean: 32.4808183333
CV result std: 2.18017249638

In [178]: l2model4.cv_score()
Evaluation metric: EFR1
CV result mean: 31.9693096667
CV result std: 4.19208533058
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




holdout_list = list()
for model in l2model._secondLayerModel__list_firstLayerModel:
    holdout_list.append(model.get_holdout())
holdout_df = pd.DataFrame(holdout_list).transpose()
# sort the column so that column index is always the same
#holdout_df = holdout_df[np.sort(holdout_df.columns)]
label = l2model._secondLayerModel__xgbData.get_holdoutLabel()
l2model._secondLayerModel__xgbData = xgb_data.xgbData(l2model._secondLayerModel__xgbData.get_train_fold(),
                                  np.array(holdout_df),
                                  np.array(label),
                                  False)
l2model._secondLayerModel__xgbData.build()
