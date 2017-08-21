from lightchem.eval.defined_eval import definedEvaluation
import numpy as np

# Testing pass contious label into evalutaion functions that required binary labels.
class pseudo_dtrain(object):
    def __init__(self, type):
        np.random.seed(2017)
        bin_labels = np.random.binomial(1,0.5,2000)
        self.bin_labels = bin_labels
        if type == "continuous":
            # Make the unique value of labels more than 0,1.
            index = np.where(self.bin_labels==1)[0]
            self.bin_labels[index[range(0, len(index)/3)]] = 5
            self.bin_labels[index[range(len(index)/3, len(index)*2/3)]] = 10
    def get_label(self):
        return self.bin_labels

np.random.seed(2017)
bin_pred = np.random.normal(0.5,0.2,2000)
dtrain_cont = pseudo_dtrain("continuous")
dtrain_bin = pseudo_dtrain("bin")

# If passing threshold to classification metric, eval function should internally
# convert label > threshold to 1.
def test_evalrocauc():
    evaluation = definedEvaluation()
    e = evaluation.eval_function("ROCAUC_0")
    value0 = e(bin_pred, dtrain_cont)
    evaluation = definedEvaluation()
    e = evaluation.eval_function("ROCAUC")
    value1 = e(bin_pred, dtrain_bin)
    assert value0 == value1

def test_evalprauc():
    evaluation = definedEvaluation()
    e = evaluation.eval_function("PRAUC_0")
    value0 = e(bin_pred, dtrain_cont)
    evaluation = definedEvaluation()
    e = evaluation.eval_function("PRAUC")
    value1 = e(bin_pred, dtrain_bin)
    assert value0 == value1

def test_evalefr1():
    evaluation = definedEvaluation()
    e = evaluation.eval_function("EFR1_0")
    value0 = e(bin_pred, dtrain_cont)
    evaluation = definedEvaluation()
    e = evaluation.eval_function("EFR1")
    value1 = e(bin_pred, dtrain_bin)
    assert value0 == value1

def test_evalefr015():
    evaluation = definedEvaluation()
    e = evaluation.eval_function("EFR015_0")
    value0 = e(bin_pred, dtrain_cont)
    evaluation = definedEvaluation()
    e = evaluation.eval_function("EFR015")
    value1 = e(bin_pred, dtrain_bin)
    assert value0 == value1

def test_evalNEFauc25():
    evaluation = definedEvaluation()
    e = evaluation.eval_function("NEFAUC25_0")
    value0 = e(bin_pred, dtrain_cont)
    evaluation = definedEvaluation()
    e = evaluation.eval_function("NEFAUC25")
    value1 = e(bin_pred, dtrain_bin)
    assert value0 == value1

def test_evalNEFauc5():
    evaluation = definedEvaluation()
    e = evaluation.eval_function("NEFAUC5_0")
    value0 = e(bin_pred, dtrain_cont)
    evaluation = definedEvaluation()
    e = evaluation.eval_function("NEFAUC5")
    value1 = e(bin_pred, dtrain_bin)
    assert value0 == value1

def test_evalAEF5():
    evaluation = definedEvaluation()
    e = evaluation.eval_function("AEF5_0")
    value0 = e(bin_pred, dtrain_cont)
    evaluation = definedEvaluation()
    e = evaluation.eval_function("AEF5")
    value1 = e(bin_pred, dtrain_bin)
    assert value0 == value1

# TODO: Logloss, RS. Convet pred = 1,0 into 0.99, 0.0001
