import numpy as np
import random
from lightchem.model import defined_model
def paramGenerator(model, num_sets, seed=2016):
    '''
    Generate hyperparameters based on model type.
    Parameters:
    -----------
    model: str, model type, choose from GbtreeLogistic, GbtreeRegression,
                GblinearLogistic, GblinearRegression
    num_sets: integer, number of hyperparameter sets to generate.
    Return a list of length num_sets, where each item is a dictionary contains
        one hyperparameter set.
    '''
    if num_sets == 1:
        preDefined_model = defined_model.definedModel()
        param = preDefined_model.model_param(model)
        return [param]
    else:
        ## Set up parameter grids.
        gbtree_grid = {'eta': np.array([0.1]),
                        'gamma': np.arange(0,10,1),
                        'max_depth': np.arange(2, 10, 1),
                        'min_child_weight': np.arange(0, 10, 1),
                        'max_delta_step': np.arange(0, 10, 1),
                        'subsample': np.arange(0.1, 1.1, 0.1),
                        'colsample_bytree': np.arange(0.1, 1.1, 0.1),
                        'colsample_bylevel': np.arange(0.1, 1.1, 0.1),
                        'lambda': np.arange(0, 10, 1),
                        'alpha': np.arange(0, 10, 1),
                        'silent': np.array([1]),
                        'seed': np.array([2016])
                        }
        gblinear_grid = {'eta': np.array([0.1]),
                          'lambda': np.arange(0, 10, 1),
                          'alpha': np.arange(0, 10, 1),
                          'lambda_bias': np.arange(0, 10,1)}
        # Change order of values
        keys = gbtree_grid.keys()
        local_seed = 1
        for key in keys:
            random.seed(local_seed)
            random.shuffle(gbtree_grid[key])
            local_seed += 1

        keys = gblinear_grid.keys()
        local_seed = 1
        for key in keys:
            random.seed(local_seed)
            random.shuffle(gblinear_grid[key])
            local_seed += 1
        # Select num_sets from each parameter space
        param_dict = {}
        if model == 'GbtreeLogistic' or model == 'GbtreeRegression':
            keys = gbtree_grid.keys()
            for key in keys:
                random.seed(seed)
                param_dict[key] = [random.choice(gbtree_grid[key]) for _ in range(num_sets)]
        elif model == 'GblinearLogistic' or model == 'GblinearRegression':
            keys = gblinear_grid.keys()
            for key in keys:
                random.seed(seed)
                param_dict[key] = [random.choice(gblinear_grid[key]) for _ in range(num_sets)]
        else:
            raise ValueError('Model name not recognized')
        # Seperate selected parameters into dictionary
        keys = param_dict.keys()
        param_dict_list = []
        for i in range(num_sets):
            temp_dict = {}
            for key in keys:
                temp_dict[key] = param_dict[key][i]
            param_dict_list.append(temp_dict)

        return param_dict_list
