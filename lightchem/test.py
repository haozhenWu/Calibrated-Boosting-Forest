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

a = load.readData('/home/haozhen/Haozhen-data/pcba128_python/data/pcba128_mmtn_canon_ecfp1024.csv','pcba-aid995')
a.read()
X_data = a.features()
y_data = a.label()

myfold = fold.fold(X_data,y_data,4)
myfold = myfold.generate_skfolds()

data = xgb_data.xgbData(myfold,X_data,y_data)
data.build()

data2 = xgb_data.xgbData(myfold,X_data,y_data,False)
data2.build()

model = first_layer_model.firstLayerModel(data,'EFR1','GbtreeLogistic')
model.xgb_cv()
