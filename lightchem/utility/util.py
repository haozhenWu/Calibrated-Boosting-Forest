import numpy as np
import pandas as pd

def fpString_to_array(fp_col):
    """
    Convert fingerprint string into array.
    Parameters:
    -----------
    fp_col: Pandas.Series, each item is a fingerprint string. Ex: 000101,110100
    return fingerprint array.
    """
    fp_array = []
    for raw_fps in fp_col:
    # Split k bit fingerprint string into list containing k items.
    # Then transform list into array so that it can be used for
    # machine learning/
        fp_array.append(np.array(list(raw_fps)))
    fp_array = np.array(fp_array)
    fp_array = fp_array.astype(np.float64)
    return fp_array


# Generate fold index based on each file.
def reverse_generate_fold_index(whole_df, file_path, fold_num, join_on):
    """
    Use to regenerate fold index from created individual fold.
    Enable lightchem to use user defined fold index.
    """
    fold_index = whole_df
    for i,path in enumerate(file_path):
        temp_fold = pd.read_csv(file_path[i])
        temp_name = 'fold'+ str(fold_num[i])
        temp_fold.loc[:,temp_name] = 1
        temp_fold = temp_fold.loc[:,['Molecule',temp_name]]
        fold_index = pd.merge(fold_index, temp_fold,on='Molecule',how='left')
        fold_index.loc[:,temp_name] = fold_index.loc[:,temp_name].fillna(0)
    fold_names = [item for item in fold_index.columns if 'fold' in item ]
    fold_index = fold_index.loc[:,fold_names]
    return fold_index
