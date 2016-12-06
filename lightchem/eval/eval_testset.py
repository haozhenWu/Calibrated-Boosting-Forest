"""
Helper method to evaluate test dataset.
"""
import compute_eval
import pandas as pd

def eval_testset(model,list_data,label,eval_name):
    """
    Method to evaluate test dataset.
    Parameters:
    -----------
    model: object
      firstLayerModel or secondLayerModel
    list_data: list
      List containing test data.
    label: numpy.ndarray
      Test label
    eval_name: str
      Name of evaluation metric
    """
        match = {'ROCAUC' : [xgb_eval.evalrocauc,True,100],
                'PRAUC' :   [xgb_eval.evalprauc,True,300],
                'EFR1' : [xgb_eval.evalefr1,True,50],
                'EFR015' : [xgb_eval.evalefr015,True,50]}

    if isinstance(model,first_layer_model.firstLayerModel):
        pred = [model.predict(list_data)]
        name = [model.name]


    elif isinstance(model,second_layer_model.secondLayerModel):
        pred = 



        if eval_name == 'ROCAUC':
            result = xgb_eval.compute_roc_auc(label,pred)
        elif eval_name == 'PRAUC':
            result = xgb_eval.compute_PR_auc(label,pred)
        elif eval_name == 'EFR1':
            result = xgb_eval.enrichment_factor(label,pred,0.01)
        elif eval_name == 'EFR015':
            result = xgb_Eval.enrichment_factor(label,pred,0.0015)

        pd.DataFrame({eval_name : pred}, index = [model.name])
