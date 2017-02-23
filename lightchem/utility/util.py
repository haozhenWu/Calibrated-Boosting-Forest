import numpy as np

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
