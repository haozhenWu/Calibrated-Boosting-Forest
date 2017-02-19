"""
Unit-test for defined_model
"""
from lightchem.model import defined_model

def test_defined_model():
    model = defined_model.definedModel()
    param = {'objective':'binary:logistic',
            'booster' : 'gbtree',
            'eta' : 0.1,
            'max_depth' : 6,
            'subsample' : 0.53,
            'colsample_bytree' : 0.7,
            'num_parallel_tree' : 1,
            'min_child_weight' : 5,
            'gamma' : 5,
            'max_delta_step':1,
            'silent':1,
            'seed' : 2016
            }
    assert model.model_param('GbtreeLogistic') == param
    mark = 0
    try:
        model.validate_model_type('not_exist_model_name')
        assert mark == 1
    except ValueError:
        mark = 1
