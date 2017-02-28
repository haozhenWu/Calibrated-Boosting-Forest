'''
Test if fingerprint can be generated properly
'''
from lightchem.featurize import fingerprint
import pandas as pd

def test_fingerprint():
    '''
    Check if fingerprints are in expected length
    '''
    # Mannually create smiles string.
    smiles = ['Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1',
                'CN(C)C1(c2nnnn2-c2ccc(Cl)cc2)CCCCC1',
                'Cc1c(C(=O)Nc2c(C)n(C)n(-c3ccccc3)c2=O)oc2ccccc12',
                'Cc1cc(N2CCOCC2)n2nc(-c3cccc(F)c3)cc2n1',
                'ThisIsaFakeSmileString']
    # morgan(ecfp) fp
    df = pd.DataFrame({'smile':smiles})
    smile_colname = 'smile'
    fp = fingerprint.smile_to_fps(df,smile_colname)
    morgan_fp = fp.Morgan()
    assert len(smiles) == morgan_fp.shape[0]
    for fp in morgan_fp.fingerprint:
        assert len(fp) == 1024
    # MACCSkeys fp
    fp = fingerprint.smile_to_fps(df,smile_colname)
    maccs_fp = fp.MACCSkeys()
    assert len(smiles) == maccs_fp.shape[0]
    for fp in maccs_fp.fingerprint:
        assert len(fp) == 167
