import numpy as np
import pandas as pd
from sklearn.metrics import auc

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

def array_to_fpString(fp_array):
    '''
    Convert array back to original fingerprint string format
    Reverse function of fpString_to_array.
    '''
    fpString_list = []
    k = 0.0
    for array_1d in fp_array:
        fp = ""
        for i in list(array_1d):
            fp = fp + str(i)
        fpString_list.append(fp)
        k += 1
    return fpString_list

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

###### Utilities for calculating Normalized Enrichment factor AUC.
# Original codes of below Utilities come from Gitter's lab @UW-Madison
# Modified by Haozhen Wu
def enrichment_factor_single_perc(y_true, y_pred, percentile):
    """
    Calculates enrichment factor vector at the given percentile for 1 label.
    This returns a 1D vector with the EF scores of the labels.
    """
    nb_classes = 1
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
    else:
        y_true = y_true.reshape((y_true.shape[0], 1))
        y_pred = y_pred.reshape((y_pred.shape[0], 1))

    ef = np.zeros(nb_classes)
    sample_size = int(y_true.shape[0] * percentile)

    for i in range(len(ef)):
        true_labels = y_true[:, i]
        pred = np.sort(y_pred[:, i], axis=0)[::-1][:sample_size]
        indices = np.argsort(y_pred[:, i], axis=0)[::-1][:sample_size]

        n_actives = np.nansum(true_labels)
        n_experimental = np.nansum( true_labels[indices] )

        try:
            ef[i] = ( float(n_experimental) /  n_actives ) / percentile
        except ValueError:
            ef[i] = np.nan

    return ef


def max_enrichment_factor_single_perc(y_true, y_pred, percentile):
    """
    Calculates max enrichment factor vector at the given percentile for 1 label.
    This returns a 1D vector with the EF scores of the labels.
    """
    nb_classes = 1
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
    else:
        y_true = y_true.reshape((y_true.shape[0], 1))
        y_pred = y_pred.reshape((y_pred.shape[0], 1))

    max_ef = np.zeros(nb_classes)
    sample_size = int(y_true.shape[0] * percentile)

    for i in range(len(max_ef)):
        true_labels = y_true[:, i]
        n_actives = np.nansum(true_labels)

        try:
            max_ef[i] = ( float(min(n_actives, sample_size)) /  float(n_actives) ) / percentile
        except ValueError:
            max_ef[i] = np.nan

    return max_ef


def enrichment_factor(y_true, y_pred, perc_vec, label_names=None):
    """
    Calculates enrichment factor vector at the percentile vectors. This returns
    2D panda matrix where the rows are the percentile.
    """
    p_count = len(perc_vec)
    nb_classes = 1
    ef_mat = np.zeros((p_count, nb_classes))

    for curr_perc in range(p_count):
        ef_mat[curr_perc,:] = enrichment_factor_single_perc(y_true,
                                        y_pred, perc_vec[curr_perc])

    """
    Convert to pandas matrix row-col names
    """
    index_names = ['{:g}'.format(perc * 100) + ' %' for perc in perc_vec]
    ef_pd = pd.DataFrame(data=ef_mat,
                         index=index_names,
                         columns=label_names)
    ef_pd.index.name = 'EF'
    return ef_pd


def max_enrichment_factor(y_true, y_pred, perc_vec, label_names=None):
    """
    Calculates max enrichment factor vector at the percentile vectors. This returns
    2D panda matrix where the rows are the percentile.
    """
    p_count = len(perc_vec)
    nb_classes = 1
    max_ef_mat = np.zeros((p_count, nb_classes))

    for curr_perc in range(p_count):
        max_ef_mat[curr_perc,:] = max_enrichment_factor_single_perc(y_true,
                                        y_pred, perc_vec[curr_perc])

    """
    Convert to pandas matrix row-col names
    """

    index_names = ['{:g}'.format(perc * 100) + ' %' for perc in perc_vec]
    max_ef_pd = pd.DataFrame(data=max_ef_mat,
                         index=index_names,
                         columns=label_names)
    max_ef_pd.index.name = 'Max_EF'
    return max_ef_pd


def norm_enrichment_factor(y_true, y_pred, perc_vec, label_names=None):
    """
    Calculates normalized enrichment factor vector at the percentile vectors.
    This returns one 2D panda matrices norm_ef where the rows
    are the percentile.
    """
    ef_pd = enrichment_factor(y_true, y_pred,
                               perc_vec, label_names)
    max_ef_pd = max_enrichment_factor(y_true, y_pred,
                                       perc_vec, label_names)

    nef_mat = ef_pd.as_matrix() / max_ef_pd.as_matrix()
    index_names = ['{:g}'.format(perc * 100) + ' %' for perc in perc_vec]
    nef_pd = pd.DataFrame(data=nef_mat,
                         index=index_names,
                         columns=label_names)
    nef_pd.index.name = 'NEF'
    return nef_pd


def nef_auc(y_true, y_pred, perc_vec, label_names=None):
    """
    Returns a pandas df of nef auc values.
    """
    nef_mat  = norm_enrichment_factor(y_true, y_pred,
                                     perc_vec, label_names)
    nef_mat = nef_mat.as_matrix()
    nb_classes = 1
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]

    nef_auc_arr = np.zeros(nb_classes)
    for i in range(nb_classes):
        nef_auc_arr[i] = auc(perc_vec, nef_mat[:,i])

    nef_auc_pd = pd.DataFrame(data=nef_auc_arr.reshape(1,len(nef_auc_arr)) / max(perc_vec),
                             index=['NEF_AUC'],
                             columns=label_names)

    return nef_auc_pd
###########
