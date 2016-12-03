import sys
sys.path.remove('/usr/lib/python2.7/dist-packages')
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/load/")
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/envir/")
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/fold/")
sys.path.append("/home/haozhen/Haozhen-data/lightchem/lightchem/data/")

import create_dir
import load
import fold
import xgb_data

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
