'''
Wrapper script, provides high level interface.
'''
import sys
sys.path.remove('/usr/lib/python2.7/dist-packages')
from lightchem.ensemble import virtualScreening_models
from lightchem.featurize import fingerprint
from lightchem.eval import defined_eval
import pandas as pd
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dir_train = os.path.join(current_dir,
                            "../../example/wrapper/muv-548/datasets/muv548_raw_train.csv.zip")
    dir_test =  os.path.join(current_dir,
                            "../../example/wrapper/muv-548/datasets/muv548_raw_test.csv.zip")
#    dir = "./datasets/muv/classification/deepchem_muv.csv.gz" # arg1
    smile_colname = 'smiles' # arg2
    label_name_list = ['MUV-548'] # arg3
    eval_name = 'ROCAUC' # arg4
    preDefined_eval = defined_eval.definedEvaluation()
    preDefined_eval.validate_eval_name(eval_name)
    df = pd.read_csv(dir_train)
    # identify NA row.
    missing_row = pd.isnull(df.loc[:,label_name_list[0]])
    df = df.loc[~missing_row]
    df = df.reset_index(drop=True)
    print 'Preparing training data fingerprints'
    # morgan(ecfp) fp
    fp = fingerprint.smile_to_fps(df,smile_colname)
    morgan_fp = fp.Morgan()
    # MACCSkeys fp
    fp = fingerprint.smile_to_fps(df,smile_colname)
    maccs_fp = fp.MACCSkeys()
    comb1 = (morgan_fp,label_name_list)
    comb2 = (maccs_fp,label_name_list)
    training_info = [comb1,comb2]
    print 'Building and selecting best model'
    model = virtualScreening_models.VsEnsembleModel(training_info,eval_name,num_of_fold=3)
    model.train()
    cv_result = model.training_result()
    print cv_result

    if True:
        df_test = pd.read_csv(dir_test)
        print 'Preparing testing data fingerprints'
        # morgan(ecfp) fp
        fp = fingerprint.smile_to_fps(df_test,smile_colname)
        morgan_fp = fp.Morgan()
        # MACCSkeys fp
        fp = fingerprint.smile_to_fps(df_test,smile_colname)
        maccs_fp = fp.MACCSkeys()
        test_data = [morgan_fp.fingerprint,maccs_fp.fingerprint]
        print 'Predict test data'
        pred = model.predict(test_data)
        pred = pd.DataFrame({'Prediction':pred})
        pred.to_csv("./prediction.csv")
