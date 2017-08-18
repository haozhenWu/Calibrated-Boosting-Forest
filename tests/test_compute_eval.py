from lightchem.eval.compute_eval import compute_roc_auc
from lightchem.eval.compute_eval import compute_PR_auc
from lightchem.eval.compute_eval import enrichment_factor
from lightchem.eval.compute_eval import compute_NEF_auc
from lightchem.eval.compute_eval import compute_AEF
from lightchem.eval.compute_eval import compute_Logloss
from lightchem.eval.compute_eval import compute_ReliabilityScore
import numpy as np

np.random.seed(2017)
bin_labels = np.random.binomial(1,0.5,2000)
bin_pred = np.random.normal(0.5,0.2,2000)

# Test whether compute_eval give same value

def test_compute_roc_auc():
    value = compute_roc_auc(bin_labels, bin_pred)
    value = np.around([value], 2)
    assert value == 0.5

def test_compute_PR_auc():
    value = compute_PR_auc(bin_labels, bin_pred)
    value = np.around([value], 2)
    assert value == 0.51

def test_enrichment_factor():
    value = enrichment_factor(bin_labels, bin_pred, 0.1)
    value = np.around([value], 2)
    assert np.float(value) == 1.07

def test_compute_NEF_auc():
    value = compute_NEF_auc(bin_labels, bin_pred, 0.5)
    value = np.around([value], 2)
    assert np.float(value) == 0.51

def test_compute_AEF():
    value = compute_AEF(bin_labels, bin_pred, 0.5)
    value = np.around([value], 2)
    assert np.float(value) == 1.02

def test_compute_Logloss():
    value = compute_Logloss(bin_labels, bin_pred)
    value = np.around([value], 2)
    assert np.float(value) == 1521.73

def test_compute_ReliabilityScore():
    value = compute_ReliabilityScore(bin_labels, bin_pred)
    value = np.around([value], 2)
    assert np.float(value) == 0.26
