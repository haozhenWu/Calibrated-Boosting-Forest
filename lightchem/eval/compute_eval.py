"""
Self defined evaluation metrics.
"""

import numpy as np
from sklearn import metrics
from lightchem.utility import util

# Custom eval metric to calculate holdout result
def compute_roc_auc( labels_arr, scores_arr ):
    '''use an sklearn function to compute ROC AUC
        probably should add some other metrics to this'''
    if len(np.unique(labels_arr)) == 2:
        auc = metrics.roc_auc_score( labels_arr, scores_arr )
    else:
        auc = 'ND'
    return auc

def compute_PR_auc( labels_arr, scores_arr ):
    '''use an sklearn function to compute ROC AUC
        probably should add some other metrics to this'''
    if len(np.unique(labels_arr)) == 2:
        auc = metrics.average_precision_score( labels_arr, scores_arr )
    else:
        auc = 'ND'
    return auc

def enrichment_factor( labels_arr, scores_arr, percentile ):
    '''calculate the enrichment factor based on some upper fraction
        of library ordered by docking scores. upper fraction is determined
        by percentile (actually a fraction of value 0.0-1.0)'''
    #ef = 0
    sample_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    pred = np.sort(scores_arr)[::-1][:sample_size] # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr)[::-1][:sample_size] # get the index positions for these in library
    n_actives = np.nansum(labels_arr) # count number of positive labels in library
    n_experimental = np.nansum( labels_arr[indices] ) # count number of positive labels in subset
    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile # calc EF at percentile
    else:
        ef = 'ND'
    return ef

def compute_NEF_auc(labels_arr, scores_arr, max_percentile):
    '''
    Calculate the AUC of Normalized Enrichment Factor, where the upper bound of
    percentile is max_percentile
    '''
    if len(np.unique(labels_arr)) == 2:
        auc = util.nef_auc(labels_arr, scores_arr,
                           np.linspace(0.001, max_percentile, 10),['nefauc'])
        auc = float(auc.nefauc)
    else:
        auc = 'ND'
    return auc
