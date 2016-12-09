from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import pandas as pd

base_data_dir = '/Users/alan/Desktop/study/SmallMolecular/pcba128_multitask_data/'
binaryLabel_dir = 'pcba128_mmtn_canon_ecfp1024.csv'
df = pd.read_csv(base_data_dir + binaryLabel_dir)
df = pd.read_csv('/home/haozhen/Haozhen-data/pcba128_python/data/pcba128_canon_ecfp1024_logac50.csv')

# input DataFrame contains col 'smile'
result_df = df.loc[:,['mol_id','smiles']]


class smile_to_fps(object):

    def __init__(self,input_df,smile_col_name):
        # TODO: assert smile_col_name in df
        self.__df = input_df
        self.__smile_col = smile_col_name
        self.__df['fingerprint'] = "none"

    def Morgan(self,radius = 2, nBits = 1024):
        k = 0
        for i,smile in enumerate(self.__df[self.__smile_col]):
            if Chem.MolFromSmiles(smile):
                tmp_mol = Chem.MolFromSmiles(smile)
                fps = AllChem.GetMorganFingerprintAsBitVect(tmp_mol,
                                                radius = radius,nBits = nBits ) # GetHashedTopologicalTorsionFingerprintAsBitVec
                self.__df.fingerprint[i] = fps.ToBitString()
                print i
            else:
                k = k+1
                print("fail")

    def RDKit(self):
        k = 0
        for i,smile in enumerate(self.__df[self.__smile_col]):
            if Chem.MolFromSmiles(smile):
                tmp_mol = Chem.MolFromSmiles(smile)
                fps = FingerprintMols.FingerprintMol(tmp_mol)
                self.__df.fingerprint[i] = fps.ToBitString()
                print i
            else:
                k = k+1
                print("fail")

    def MACCSkeys(self):
        k = 0
        for i,smile in enumerate(result_df.smiles):
            if Chem.MolFromSmiles(smile):
                tmp_mol = Chem.MolFromSmiles(smile)
                fps = MACCSkeys.GenMACCSKeys(tmp_mol)
                result_df.fingerprint[i] = fps.ToBitString()
                print i
            else:
                k = k+1
                print("fail")





from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols

result_df['fingerprint'] = "none"
k = 0
for i,smile in enumerate(result_df.smiles):
    if Chem.MolFromSmiles(smile):
        tmp_mol = Chem.MolFromSmiles(smile)
        fps = FingerprintMols.FingerprintMol(tmp_mol)
        result_df.fingerprint[i] = fps.ToBitString()
        print i
    else:
        k = k+1
        print("fail")


from rdkit.Chem import MACCSkeys

result_df['fingerprint'] = "none"
k = 0
for i,smile in enumerate(result_df.smiles):
    if Chem.MolFromSmiles(smile):
        tmp_mol = Chem.MolFromSmiles(smile)
        fps = MACCSkeys.GenMACCSKeys(tmp_mol)
        result_df.fingerprint[i] = fps.ToBitString()
        print i
    else:
        k = k+1
        print("fail")
