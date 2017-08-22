"""
Self defined evaluation metrics.
"""

import numpy as np
from sklearn import metrics
from lightchem.utility import util

# Custom eval metric to calculate holdout result
def compute_roc_auc(labels_arr, scores_arr ):
    '''
    Compute Area under the curve of Receiver Operating Characteristic
    '''
    if len(np.unique(labels_arr)) == 2:
        auc = util.ROC_auc(labels_arr, scores_arr)
    else:
        auc = 'ND'
    return auc

def compute_PR_auc(labels_arr, scores_arr ):
    '''
    Compute average precision. Treat it as AUC of PR
    '''
    if len(np.unique(labels_arr)) == 2:
        auc = util.avg_precision(labels_arr, scores_arr)
    else:
        auc = 'ND'
    return auc

def enrichment_factor(labels_arr, scores_arr, percentile ):
    '''Compute the enrichment factor'''
    ef = util.enrichment_factor(labels_arr, scores_arr, np.array([percentile]))[0]
    return ef

def compute_NEF_auc(labels_arr, scores_arr, max_percentile):
    '''
    Calculate the AUC of Normalized Enrichment Factor, where the upper bound of
    percentile is max_percentile.
    '''
    if len(np.unique(labels_arr)) == 2:
        auc = util.nef_auc(labels_arr, scores_arr,
                           np.linspace(0.001, max_percentile, 10))
    else:
        auc = 'ND'
    return auc

def compute_AEF(labels_arr, scores_arr, max_percentile):
    '''
    Return average enrichment factor at multiple threshold where max threshold
    is max_percentile.
    '''
    if len(np.unique(labels_arr)) == 2:
        percentile_list = np.linspace(0, max_percentile, 10)
        aef = util.enrichment_factor(labels_arr, scores_arr, percentile_list)
        aef = np.nanmean(aef)
    else:
        aef = 'ND'
    return aef

def compute_Logloss(labels_arr, scores_arr):
    '''
    Calculate the logistic loss for binary label.
    '''
    scores_arr = util.__normalize_Logloss(scores_arr)
    logloss = util.logloss(labels_arr, scores_arr)
    return logloss

def compute_ReliabilityScore(labels_arr, scores_arr):
    '''
    Calculate the Reliability Scores for binary label.
    '''
    scores_arr = util.__normalize_minMax(scores_arr)
    rs = util.reliability_score(labels_arr, scores_arr, n_bin=20)
    return rs
