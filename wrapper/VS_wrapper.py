'''
Wrapper script, provides high level interface.
'''
import sys
from lightchem.ensemble import virtualScreening_models
from lightchem.featurize import fingerprint
from lightchem.eval import defined_eval
import pandas as pd
import os
import numpy as np

if __name__ == "__main__":
    """
    Require VS_wrapper_config.json.
    """
    # Read command line inpu and match input.
    with open(sys.argv[1], 'r') as f:
        info = f.read()
    info = pd.read_json(info)
    target_name = np.str(info.loc['target_name'][0])
    dir_train = np.str(info.loc['full_directory_to_training_data'][0])
    dir_test = np.str(info.loc['full_directory_to_dataToPredict_if_exit'][0])
    smile_colname = np.str(info.loc['smile_column_name'][0])
    label_name_list = info.loc['label_name_list'][0]
    label_name_list = [np.str(item) for item in label_name_list]
    eval_name = np.str(info.loc['evaluation_name'][0])
    dir_to_store = np.str(info.loc['full_directory_to_store_prediction'][0])
#    current_dir = os.path.dirname(os.path.realpath(__file__))
#    dir_train = os.path.join(current_dir,
#                            "../../example/wrapper/muv-548/datasets/muv548_raw_train.csv.zip")
#    dir_test =  os.path.join(current_dir,
#                            "../../example/wrapper/muv-548/datasets/muv548_raw_test.csv.zip")
#    dir = "./datasets/muv/classification/deepchem_muv.csv.gz" # arg1
#    smile_colname = 'smiles' # arg2
#    label_name_list = ['MUV-548'] # arg3
#    eval_name = 'ROCAUC' # arg4
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

    if dir_test != "":
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
        pred.to_csv(os.path.join(dir_to_store,target_name + "_prediction.csv"))
