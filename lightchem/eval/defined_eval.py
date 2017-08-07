"""
Pre-defined evaluation metrics.
"""
from lightchem.eval import xgb_eval
import re
import functools

class definedEvaluation(object):
    """
    Pre-defined evaluation metircs and their default setting related to models.
    Currently supports:
        `ROCAUC`: Area under curve of ROC
        `PRAUC`: Area under curve of Precision-recall
        `EFR1`: Enrichment factor at 0.01
        `EFR015`: Enrichment factor at 0.0015
        `NEFAUC25`: Area under curve of Normalized Enrichment Factor,
            range between 0.001 and 0.25
        `NEFAUC5`: Area under curve of Normalized Enrichment Factor,
            range between 0.001 and 0.05
        `AEF5`: Average Enrichement factor at multiple threshold,
            max(threshold) = 0.05
        `Logloss`: Logistic loss for binary label
        `ReliabilityScore`: RS measure the quality of probabilities
            for binary labels.
    If passing user-specific threshold to intervally convert continuous label
        into binary label, use the format evalname_X where evalname is a
        pre-defined evaluation metric name and X is the threshold to cut.
    """
    def __init__(self):
        self.__DEFINED_EVAL = ['ROCAUC','PRAUC','EFR1','EFR015','NEFAUC25',
                                'NEFAUC5','AEF5', 'Logloss','ReliabilityScore']
        self.__MATCH =  {'ROCAUC' : [xgb_eval.evalrocauc,True,100],
                        'PRAUC' :   [xgb_eval.evalprauc,True,300],
                        'EFR1' : [xgb_eval.evalefr1,True,50],
                        'EFR015' : [xgb_eval.evalefr015,True,50],
                        'NEFAUC25': [xgb_eval.evalNEFauc25,True,100],
                        'NEFAUC5': [xgb_eval.evalNEFauc5,True,100],
                        'AEF5': [xgb_eval.evalAEF5,True,100],
                        'Logloss': [xgb_eval.evalLogloss,False,100],
                        'ReliabilityScore': [xgb_eval.evalReliabilityScore,
                                                False,100]}

    def __check_cut_value(self,name):
        eval_cut = re.split("_", name)
        if len(eval_cut) > 2:
            raise ValueError('In order to pass threshold to evaluation metric, '
                                'must follow evalname_X format where X is the '
                                'threshold value.')
        return eval_cut

    def eval_list(self):
        return self.__DEFINED_EVAL

    def eval_function(self,eval_name):
        evaluation_cut = self.__check_cut_value(eval_name)
        name = evaluation_cut[0]
        self.validate_eval_name(name)
        func = self.__MATCH[name][0]
        if len(evaluation_cut) == 2: # means user pass additional threshold.
            new_cut = float(evaluation_cut[1])
            func = functools.partial(func, cut=new_cut)
        return func

    def is_maximize(self,eval_name):
        evaluation_cut = self.__check_cut_value(eval_name)
        name = evaluation_cut[0]
        self.validate_eval_name(name)
        return self.__MATCH[name][1]

    def stopping_round(self,eval_name):
        evaluation_cut = self.__check_cut_value(eval_name)
        name = evaluation_cut[0]
        self.validate_eval_name(name)
        return self.__MATCH[name][2]

    def validate_eval_name(self,eval_name):
        evaluation_cut = self.__check_cut_value(eval_name)
        name = evaluation_cut[0]
        if not name in self.__DEFINED_EVAL:
            raise ValueError('Defined evaluation metric names are: ' + ', '.join(self.__DEFINED_EVAL))
