"""
Wrapper of rdkit package to transform molucule into fingerprint.
"""
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import pandas as pd

class smile_to_fps(object):

    def __init__(self,input_df,smile_col_name):
        # TODO: assert smile_col_name in df
        self.__df = input_df.copy()
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
            else:
                # fail to construct a RDKit molecule, mannual create fingerprint with all 0.
                fps = '0'
                for k in range(nBits-1):
                    fps = fps + '0'
                self.__df.fingerprint[i] = fps
                k += 1
        print 'Number of molecue failed: ' + str(k)
        return self.__df

    def MACCSkeys(self):
        k = 0
        for i,smile in enumerate(self.__df[self.__smile_col]):
            if Chem.MolFromSmiles(smile):
                tmp_mol = Chem.MolFromSmiles(smile)
                fps = MACCSkeys.GenMACCSKeys(tmp_mol)
                self.__df.fingerprint[i] = fps.ToBitString()
            else:
                # fail to construct a RDKit molecule, mannual create fingerprint with all 0.
                fps = '0'
                for k in range(166):
                    fps = fps + '0'
                self.__df.fingerprint[i] = fps
                k += 1
        print 'Number of molecue failed: ' + str(k)
        return self.__df

""" RDKit featurization method has problem. Create unequal length fingerprint.

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

"""
