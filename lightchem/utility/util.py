import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn import metrics
import re

def fpString_to_array(fp_col, sep = ""):
    """
    Convert fingerprint string into array.
    Parameters:
    -----------
    fp_col: Pandas.Series, each item is a fingerprint string. Ex: 000101,110100
    sep: Value used to seperate original value.
    Return fingerprint array.
    """
    fp_array = []
    for raw_fps in fp_col:
        # Split k bit fingerprint string into list containing k items.
        # Then transform list into array so that it can be used for
        # machine learning
        if sep == "":
            fp_array.append(np.array(list(raw_fps)))
        elif sep == "|":
            fp_array.append(np.array(re.split("\|",raw_fps)))
        else:
            fp_array.append(np.array(re.split(sep,raw_fps)))
    fp_array = np.array(fp_array)
    fp_array = fp_array.astype(np.float64)
    return fp_array

def array_to_fpString(fp_array, sep = ""):
    '''
    Convert array back to original fingerprint string format
    Reverse function of fpString_to_array.
    '''
    fpString_list = []
    for array_1d in fp_array:
        fp = ""
        for index,i in enumerate(list(array_1d)):
            if index == len(list(array_1d))-1:
                fp = fp + str(i)
            else:
                fp = fp + str(i) + sep
        fpString_list.append(fp)
    return fpString_list

def reverse_generate_fold_index(whole_df, file_path, fold_num, join_on):
    """
    Use to regenerate fold index from created individual fold.
    Enable lightchem to use user defined fold index.
    Parameters:
    -----------
    whole_df: pd.DataFrame
      Original DataFrame contains all the folds before split.
    file_path: list
      List contains the path to each fold file.
    fold_num: list
      List contains the fold index of each path in file_path
    join_on: str
      which column to join on between fold and original DataFrame.
    """
    fold_index = whole_df
    for i,path in enumerate(file_path):
        temp_fold = pd.read_csv(file_path[i])
        temp_name = 'fold'+ str(fold_num[i])
        temp_fold.loc[:,temp_name] = 1
        temp_fold = temp_fold.loc[:,[join_on,temp_name]]
        fold_index = pd.merge(fold_index, temp_fold,on=join_on,how='left')
        fold_index.loc[:,temp_name] = fold_index.loc[:,temp_name].fillna(0)
    fold_names = [item for item in fold_index.columns if 'fold' in item ]
    fold_index = fold_index.loc[:,fold_names]
    return fold_index

def reliability_score(y_true, y_pred, n_bin=20):
    """
    Implementation of Reliability score that measure the quality of predicted
    probabilities.
    """
    df = pd.DataFrame({"label":y_true, "prediction":y_pred})
    df = df.loc[:, ["label", "prediction"]]
    df = df.sort_values("prediction")
    length = df.shape[0] / n_bin
    reminder = df.shape[0] % n_bin
    bins = list(np.repeat(range(1,n_bin+1),length))
    bins = bins + list(np.repeat(n_bin, reminder))
    df['categories'] = bins
    df = df.groupby('categories').mean() / df.loc[:,"label"].mean()
    diff = np.nanmean(abs(df.loc[:, "prediction"] - df.loc[:, "label"]))
    return diff

###### Utilities for calculating Normalized Enrichment factor AUC.
# Original codes of below Utilities come from Gitter's lab @UW-Madison
# Modified by Haozhen Wu. Vectorized some codes to improve speed.
def enrichment_factor(y_true, y_pred, perc_vec):
    """
    Calculates enrichment factor vector for a vector of percentiles for 1 label.
    This returns a 1D vector with the EF scores of the label.
    """
    sample_size_vec = y_true.shape[0] * perc_vec
    sample_size_vec = sample_size_vec.astype(np.int)
    y_pred_argsort = np.argsort(y_pred,kind="mergesort")[::-1]
    indices_vec = [y_pred_argsort[:size] for size in sample_size_vec]
    pred_vec = [y_pred[indices] for indices in indices_vec]
    n_actives = np.nansum(y_true)
    if n_actives == 0:
        ef_vec = np.repeat(np.nan, len(perc_vec))
        return ef_vec
    n_experimental_vec = [np.nansum( y_true[indices]  ) for indices in indices_vec]
    n_experimental_vec = np.float32(n_experimental_vec)
    ef_vec = ( n_experimental_vec /  n_actives ) / perc_vec

    return ef_vec

def max_enrichment_factor(y_true, y_pred, perc_vec):
    """
    Calculates max enrichment factor vector for a vector of percentiles for 1 label.
    This returns a 1D vector with the EF scores of the label.
    """
    sample_size_vec = y_true.shape[0] * perc_vec
    sample_size_vec = sample_size_vec.astype(np.int)
    n_actives = np.nansum(y_true)
    if n_actives == 0:
        max_ef_vec = np.repeat(np.nan, len(perc_vec))
        return max_ef_vec

    minimum_vec = [float(min(n_actives, sample_size)) for sample_size in sample_size_vec]
    max_ef_vec = ( minimum_vec /  n_actives ) / perc_vec

    return max_ef_vec

def norm_enrichment_factor(y_true, y_pred, perc_vec):
    """
    Calculates normalized enrichment factor vector for a vector of percentiles.
    This returns a 1D vector with norm_ef scores.
    """
    ef_mat = enrichment_factor(y_true, y_pred,
                               perc_vec)
    max_ef_mat = max_enrichment_factor(y_true, y_pred,
                                       perc_vec)

    nef_mat = ef_mat / max_ef_mat
    return nef_mat


def nef_auc(y_true, y_pred, perc_vec):
    """
    Returns nef auc value.
    """
    nef_mat = norm_enrichment_factor(y_true, y_pred, perc_vec)
    nef_auc_arr = auc(perc_vec, nef_mat)
    return nef_auc_arr / max(perc_vec)
###########

def __normalize_Logloss(input_arr):
    input_arr = np.array(input_arr)
    if max(input_arr) - min(input_arr) > 1:
        # normalize prediction into (0,1)
        input_arr = (input_arr - min(input_arr)) / (max(input_arr) - min(input_arr))
        input_arr[np.where(input_arr==0)[0]] = 0.000001
        input_arr[np.where(input_arr==1)[0]] = 0.999999
    return input_arr

def __normalize_minMax(input_arr):
    if max(input_arr) - min(input_arr) > 1:
        # normalize prediction into [0,1]
        input_arr = (input_arr - min(input_arr)) / (max(input_arr) - min(input_arr))
    return input_arr

def logloss(y_ture, y_pred):
    loss = np.sum(-(y_ture*np.log(y_pred) + (1-y_ture)*np.log(1-y_pred)))
    return loss

def ROC_auc(y_ture, y_pred):
    '''Use sklearn function to compute ROC AUC'''
    score = metrics.roc_auc_score(y_ture, y_pred)
    return score

def avg_precision(y_true, y_pred):
    '''Use sklearn function to compute average precision'''
    score = metrics.average_precision_score(y_true, y_pred)
    return score

def PRC_auc(y_ture, y_pred):
    precision, recall, _ = metrics.precision_recall_curve(y_ture, y_pred)
    return auc(recall, precision)
