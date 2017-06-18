"""
Contain self-define evalutaion methods that used for monitor xgboost
training process.
"""

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
from lightchem.utility import util

def evalrocauc(preds, dtrain):
    '''
    Return ROC AUC score
    '''
    # Check infinite, NaN. Convert to 0.
    index = np.where(np.logical_or(np.isinf(preds), np.isnan(preds)))
    preds[index] = 0
    labels = dtrain.get_label()
    unique = np.unique(labels)
    if len(unique) > 2: # which means it is continuous label
        cut = np.percentile(labels,99)
        if cut == unique[0]:
            cut = unique[1]
        labels[np.where(dtrain.get_label()>cut)] = 1
        labels[np.where(dtrain.get_label()<=cut)] = 0
    # use sklearn.metrics to compute rocauc
    return 'ROCAUC', metrics.roc_auc_score( labels, preds )

def evalprauc(preds, dtrain):
    '''
    Return Precision Recall AUC score
    '''
    # Check infinite, NaN. Convert to 0.
    index = np.where(np.logical_or(np.isinf(preds), np.isnan(preds)))
    preds[index] = 0
    labels = dtrain.get_label()
    unique = np.unique(labels)
    if len(unique) > 2: # which means it is continuous label
        cut = np.percentile(labels,99)
        if cut == unique[0]:
            cut = unique[1]
        labels[np.where(dtrain.get_label()>cut)] = 1
        labels[np.where(dtrain.get_label()<=cut)] = 0
    # use sklearn.metrics to compute prauc
    return 'PRAUC', metrics.average_precision_score( labels, preds )

def evalefr1(preds, dtrain):
    '''
    Return enrichment factor at 0.01
    '''
    # Check infinite, NaN. Convert to 0.
    index = np.where(np.logical_or(np.isinf(preds), np.isnan(preds)))
    preds[index] = 0
    labels = dtrain.get_label()
    unique = np.unique(labels)
    if len(unique) > 2: # which means it is continuous label
        cut = np.percentile(labels,99)
        if cut == unique[0]:
            cut = unique[1]
        labels[np.where(dtrain.get_label()>cut)] = 1
        labels[np.where(dtrain.get_label()<=cut)] = 0

    labels_arr = labels
    scores_arr = preds
    percentile = 0.01
    sample_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    pred = np.sort(scores_arr)[::-1][:sample_size] # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr)[::-1][:sample_size] # get the index positions for these in library
    n_actives = np.nansum(labels_arr) # count number of positive labels in library
    n_experimental = np.nansum( labels_arr[indices] ) # count number of positive labels in subset
    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile # calc EF at percentile
    else:
        ef = -1
    return 'EFR1', ef

def evalefr015(preds, dtrain):
    '''
    Return enrichment factor at 0.0015
    '''
    # Check infinite, NaN. Convert to 0.
    index = np.where(np.logical_or(np.isinf(preds), np.isnan(preds)))
    preds[index] = 0
    labels = dtrain.get_label()
    unique = np.unique(labels)
    if len(np.unique(labels)) > 2: # which means it is continuous label
        cut = np.percentile(labels,99)
        if cut == unique[0]:
            cut = unique[1]
        labels[np.where(dtrain.get_label()>cut)] = 1
        labels[np.where(dtrain.get_label()<=cut)] = 0

    labels_arr = labels
    scores_arr = preds
    percentile = 0.0015
    sample_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    pred = np.sort(scores_arr)[::-1][:sample_size] # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr)[::-1][:sample_size] # get the index positions for these in library
    n_actives = np.nansum(labels_arr) # count number of positive labels in library
    n_experimental = np.nansum( labels_arr[indices] ) # count number of positive labels in subset
    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile # calc EF at percentile
    else:
        ef = -1
    return 'EFR015', ef


def evalNEFauc25(preds, dtrain):
    '''
    Return Normalized Enrichment Factor AUC ranging from 0.001 to 0.25
    '''
    # Check infinite, NaN. Convert to 0.
    index = np.where(np.logical_or(np.isinf(preds), np.isnan(preds)))
    preds[index] = 0
    labels = dtrain.get_label()
    unique = np.unique(labels)
    if len(np.unique(labels)) > 2: # which means it is continuous label
        cut = np.percentile(labels,99)
        if cut == unique[0]:
            cut = unique[1]
        labels[np.where(dtrain.get_label()>cut)] = 1
        labels[np.where(dtrain.get_label()<=cut)] = 0
    nef = util.nef_auc(labels, preds, np.linspace(0.001, .25, 10),['nefauc'])
    return 'NEFAUC25', float(nef.nefauc)

def evalNEFauc5(preds, dtrain):
    '''
    Return Normalized Enrichment Factor AUC ranging from 0.001 to 0.05
    '''
    # Check infinite, NaN. Convert to 0.
    index = np.where(np.logical_or(np.isinf(preds), np.isnan(preds)))
    preds[index] = 0
    labels = dtrain.get_label()
    unique = np.unique(labels)
    if len(np.unique(labels)) > 2: # which means it is continuous label
        cut = np.percentile(labels,99)
        if cut == unique[0]:
            cut = unique[1]
        labels[np.where(dtrain.get_label()>cut)] = 1
        labels[np.where(dtrain.get_label()<=cut)] = 0
    nef = util.nef_auc(labels, preds, np.linspace(0.001, .05, 10))
    return 'NEFAUC5', nef
