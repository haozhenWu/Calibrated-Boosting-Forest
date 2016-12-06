"""
Helper method to evaluate test dataset.
"""
import compute_eval
import pandas as pd
import numpy as np
import first_layer_model

def eval_testset(model,list_data,label,eval_name):
    """
    Method to evaluate test dataset. Return a pd.DataFrame
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
    if isinstance(model,first_layer_model.firstLayerModel):
        pred = [model.predict(list_data)]
        name = [model.name]

    elif isinstance(model,second_layer_model.secondLayerModel):
        pred = [model.predict(list_data)]
        name = [model.name]
        firstLayerModel_predictions = model.get_firstLayerModel_predictions()
        for i in range(firstLayerModel_predictions.shape[1]):
            pred.append(np.array(firstLayerModel_predictions.iloc[:,i]))
            name.append(firstLayerModel_predictions.columns[i])

    result = []
    for i in range(len(pred)):
        if eval_name == 'ROCAUC':
            result.append(compute_eval.compute_roc_auc(label,pred[i]))
        elif eval_name == 'PRAUC':
            result.append(compute_eval.compute_PR_auc(label,pred[i]))
        elif eval_name == 'EFR1':
            result.append(compute_eval.enrichment_factor(label,pred[i],0.01))
        elif eval_name == 'EFR015':
            result.append(compute_eval.enrichment_factor(label,pred[i],0.0015))

    return pd.DataFrame({eval_name : result}, index = [name])
