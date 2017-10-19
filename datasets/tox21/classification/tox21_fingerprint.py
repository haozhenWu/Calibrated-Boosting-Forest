# script to generate fingerprint for Tox21

from lightchem.featurize import fingerprint
import pandas as pd
import os
import time

if __name__ == "__main__":
    start_time = time.time()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tox21 = pd.read_csv(current_dir + "/deepchem_tox21.csv.gz")
    # get target names
    tox21_targetName = pd.DataFrame(tox21.columns[0:11])
    tox21_targetName.to_csv(current_dir + "/../tox21_TargetName.csv",index=False, header = False)

    # Assign 0 to missing value.
    tox21 = tox21.fillna(0)

    smile_colname = 'smiles'
    fp = fingerprint.smile_to_fps(tox21,smile_colname)
    # morgan(ecfp) fp
    morgan_fp = fp.Morgan()
    morgan_fp.to_csv(current_dir + "/tox21_BinaryLabel_ecfp1024.csv",index = False)

    # MACCSkeys fp
    fp = fingerprint.smile_to_fps(tox21,smile_colname)
    maccs_fp = fp.MACCSkeys()
    maccs_fp.to_csv(current_dir + "/tox21_BinaryLabel_MACCSkey167.csv",index = False)

    print(" --- %s seconds ---\n" % (time.time() - start_time))
