"""
Contain self-defined evalutaion methods that used to monitor xgboost
training process.
"""

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
from lightchem.utility import util

def __map_cont_to_bin(dtrain, cut):
    labels = dtrain.get_label()
    unique = np.unique(labels)
    if len(unique) > 2: # which means it is continuous label
        if cut == None:
            cut = np.percentile(labels,99)
            if cut == unique[0]:
                cut = unique[1]
        labels[np.where(dtrain.get_label()>cut)] = 1
        labels[np.where(dtrain.get_label()<=cut)] = 0
    return labels

def __NA_to_zero(prediction):
    # Check infinite, NaN. Convert to 0.
    index = np.where(np.logical_or(np.isinf(prediction), np.isnan(prediction)))
    prediction[index] = 0
    return prediction

def evalrocauc(preds, dtrain, cut=None):
    '''
    Return ROC AUC score
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    return 'ROCAUC', util.ROC_auc(labels, preds)

def evalprauc(preds, dtrain, cut=None):
    '''
    Return Precision Recall AUC score
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    prauc = util.avg_precision(labels, preds)
    if len(np.unique(preds)) <= 128:
        prauc = 0
    return 'PRAUC', prauc

def evalefr1(preds, dtrain, cut=None):
    '''
    Return enrichment factor at 0.01
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    percentile = 0.01
    ef = util.enrichment_factor(labels, preds, np.array([percentile]))[0]
    return 'EFR1', ef

def evalefr015(preds, dtrain, cut=None):
    '''
    Return enrichment factor at 0.0015
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    percentile = 0.0015
    ef = util.enrichment_factor(labels, preds, np.array([percentile]))[0]
    return 'EFR015', ef


def evalNEFauc25(preds, dtrain, cut=None):
    '''
    Return Normalized Enrichment Factor AUC ranging from 0.001 to 0.25
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    nef = util.nef_auc(labels, preds, np.linspace(0.001, .25, 10))
    # EF calculation for first several rounds are wrong when trees are small and
    # unique predictions are low.
    # Mannually set to zero. 2^7 = 128 -> Tree with depth 7
    if len(np.unique(preds)) <= 128:
        nef = 0
    return 'NEFAUC25', nef

def evalNEFauc5(preds, dtrain, cut=None):
    '''
    Return Normalized Enrichment Factor AUC ranging from 0.001 to 0.05
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    nef = util.nef_auc(labels, preds, np.linspace(0.001, .05, 10))
    # EF calculation for first several rounds are wrong when trees are small and
    # unique predictions are low.
    # Mannually set to zero. 2^7 = 128 -> Tree with depth 7
    if len(np.unique(preds)) <= 128:
        nef = 0
    return 'NEFAUC5', nef

def evalAEF5(preds, dtrain, cut=None):
    '''
    Return average enrichment factor at multiple threshold where max threshold is 5%.
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    percentile_list = np.linspace(0,0.05,10)
    aef = util.enrichment_factor(labels, preds, percentile_list)
    aef = np.nanmean(aef)
    # EF calculation for first 2 round is wrong when trees are small.
    # Mannually set to zero. 2^7 = 128 -> Tree with depth 7
    if len(np.unique(preds)) <= 128:
        aef = 0
    return 'AEF', aef

def evalLogloss(preds, dtrain, cut=None):
    '''
    Return logistic loss
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    preds = util.__normalize_Logloss(preds)
    logloss = util.logloss(labels, preds)
    return 'Logloss', logloss

def evalReliabilityScore(preds, dtrain, cut=None):
    '''
    Return realibility scores
    '''
    preds = __NA_to_zero(preds)
    labels = __map_cont_to_bin(dtrain, cut)
    preds = util.__normalize_minMax(preds)
    rs = util.reliability_score(labels, preds, n_bin=20)
    return "ReliabilityScore", rs
