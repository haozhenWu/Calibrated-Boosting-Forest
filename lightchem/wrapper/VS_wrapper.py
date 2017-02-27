'''
Wrapper script, provides high level interface.
'''
from lightchem.ensemble import virtualScreening_models
from lightchem.featurize import fingerprint
from lightchem.eval import defined_eval
import pandas as pd

# 1. convert SMILE string into fp, for both training and testing data.
# 2. call ensemble model
# 3. predict and store result.
    dir = "./datasets/muv/classification/deepchem_muv.csv.gz" # arg1
    smile_colname = 'smiles' # arg2
    label_name_list = ['MUV-466'] # arg3
    eval_name = 'ROCAUC' # arg4
    preDefined_eval = defined_eval.definedEvaluation()
    preDefined_eval.validate_eval_name(eval_name)

df = pd.read_csv("./datasets/muv/classification/deepchem_muv.csv.gz")
# identify NA row.
missing_row = pd.isnull(df.loc[:,label_name_list[0]])
df = df.loc[~missing_row]
df = df.reset_index(drop=True)
#backup = df
    fp = fingerprint.smile_to_fps(df,smile_colname)
    # morgan(ecfp) fp
    morgan_fp = fp.Morgan()

    # MACCSkeys fp
    fp = fingerprint.smile_to_fps(df,smile_colname)
    maccs_fp = fp.MACCSkeys()

comb1 = (morgan_fp,label_name_list)
comb2 = (maccs_fp,label_name_list)
training_info = [comb1,comb2]
model = virtualScreening_models.VsEnsembleModel(training_info,eval_name)
model.train()
cv_result = model.training_result()

future_data = [morgan_fp.fingerprint,maccs_fp.fingerprint]
pred = model.predict(future_data)

#
#            print len(self.__test_data)
#            print index
#            print j
#1
#1
#2
# test data feature name, mismatch, train feature name

if __name__ == "__main__":
    dir = "./datasets/muv/classification/deepchem_muv.csv.gz" # arg1
    smile_colname = 'smiles' # arg2
    label_name_list = ['MUV-466'] # arg3
    df = pd.read_csv(dir)
    name = label_name_list
    name.append(smile_colname)
    temp = df[name]
    fp = fingerprint.smile_to_fps(temp,smile_colname)
    # morgan(ecfp) fp
    morgan_fp = fp.Morgan()

    # MACCSkeys fp
    fp = fingerprint.smile_to_fps(muv,smile_colname)
    maccs_fp = fp.MACCSkeys()
