# script to generate fingerprint for Tox21

from lightchem.featurize import fingerprint
import pandas as pd
import os
import time

if __name__ == "__main__":
    start_time = time.time()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    muv = pd.read_csv(current_dir + "/deepchem_muv.csv.gz")
    # get target names
    muv_targetName = pd.DataFrame(muv.columns[0:17])
    muv_targetName.to_csv(current_dir + "/../muv_TargetName.csv",index=False, header = False)

    # Assign 0 to missing label.
    muv = muv.fillna(0)

    smile_colname = 'smiles'
    fp = fingerprint.smile_to_fps(muv,smile_colname)
    # morgan(ecfp) fp
    morgan_fp = fp.Morgan()
    morgan_fp.to_csv(current_dir + "/muv_BinaryLabel_ecfp1024.csv",index = False)

    # MACCSkeys fp
    fp = fingerprint.smile_to_fps(muv,smile_colname)
    maccs_fp = fp.MACCSkeys()
    maccs_fp.to_csv(current_dir + "/muv_BinaryLabel_MACCSkey167.csv",index = False)

    print(" --- %s seconds ---\n" % (time.time() - start_time))
