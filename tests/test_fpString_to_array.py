'''
Test utility.util.fpString_to_array
'''
from lightchem.utility import util
import pandas as pd

def test_fpString_to_array():
    fp = pd.Series(['11010','01001'])
    fp_array = util.fpString_to_array(fp)
    assert len(fp) == len(fp_array)
    for i,item in enumerate(fp):
        assert len(item) == len(fp_array[i])
