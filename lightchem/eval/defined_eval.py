"""
Pre-defined evaluation metrics.
"""
from lightchem.eval import xgb_eval
class definedEvaluation(object):
    """
    Pre-defined evaluation metircs and their default setting related to models.
    Currently supports:
    `ROCAUC`: Area under curve of ROC
    `PRAUC`: Area under curve of Precision-recall
    `EFR1`: Enrichment factor at 0.01
    `EFR015`: Enrichment factor at 0.0015
    `NEFAUC25`: Area under curve of Normalized Enrichment Factor, range between 0.001 and 0.25
    `NEFAUC5`: Area under curve of Normalized Enrichment Factor, range between 0.001 and 0.05
    """
    def __init__(self):
        self.__DEFINED_EVAL = ['ROCAUC','PRAUC','EFR1','EFR015','NEFAUC25']
        self.__MATCH =  {'ROCAUC' : [xgb_eval.evalrocauc,True,100],
                        'PRAUC' :   [xgb_eval.evalprauc,True,300],
                        'EFR1' : [xgb_eval.evalefr1,True,50],
                        'EFR015' : [xgb_eval.evalefr015,True,50],
                        'NEFAUC25': [xgb_eval.evalNEFauc25,True,100],
                        'NEFAUC5': [xgb_eval.evalNEFauc5,True,100]}

    def eval_list(self):
        return self.__DEFINED_EVAL

    def eval_function(self,eval_name):
        self.validate_eval_name(eval_name)
        return self.__MATCH[eval_name][0]

    def is_maximize(self,eval_name):
        self.validate_eval_name(eval_name)
        return self.__MATCH[eval_name][1]

    def stopping_round(self,eval_name):
        self.validate_eval_name(eval_name)
        return self.__MATCH[eval_name][2]

    def validate_eval_name(self,eval_name):
        if not eval_name in self.__DEFINED_EVAL:
            raise ValueError('Defined evaluation metric names are: ' + ', '.join(self.__DEFINED_EVAL))
