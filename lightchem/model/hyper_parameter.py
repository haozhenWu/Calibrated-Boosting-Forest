import numpy as np
import random
from lightchem.model import defined_model

def shuffle_param(dict_params):
    '''
    Shuffle the order of value within each parameter space. This is to help
        sample the parameters more random.
    '''
    keys = dict_params.keys()
    local_seed = 1
    for key in keys:
        random.seed(local_seed)
        random.shuffle(dict_params[key])
        local_seed += 1
    return dict_params

def select_param(dict_params, num_sets, seed):
    '''
    Randomly select num_set value from each parameter space.
    '''
    param_dict = {}
    keys = dict_params.keys()
    for key in keys:
        random.seed(seed)
        param_dict[key] = [random.choice(dict_params[key]) for _ in range(num_sets)]
    return param_dict

def paramGenerator(model, num_sets, which_layer, seed=2016):
    '''
    Generate hyperparameters based on model type.
    Parameters:
    -----------
    model: str, model type, choose from GbtreeLogistic, GbtreeRegression,
                GblinearLogistic, GblinearRegression
    num_sets: integer, number of hyperparameter sets to generate.
    which_layer: str, layer1 and layer2 have different grid.
                    Choose from `layer1` or 'layer2'
    Return a list of length num_sets, where each item is a dictionary contains
        one hyperparameter set.
    '''
    if num_sets == 1:
        preDefined_model = defined_model.definedModel()
        param = preDefined_model.model_param(model)
        return [param]
    else:
        ## Set up parameter grids for layer1
        GbtreeLogistic_grid_layer1 = {'booster': np.array(['gbtree']),
                                      'objective': np.array(['binary:logistic']),
                                      'eta': np.array([0.1]),
                                      'gamma': np.arange(0,10,1),
                                      'max_depth': np.arange(2, 10, 1),
                                      'min_child_weight': np.arange(0, 10, 1),
                                      'max_delta_step': np.arange(0, 10, 1),
                                      'subsample': np.arange(0.1, 1.01, 0.1),
                                      'colsample_bytree': np.arange(0.1, 1.01, 0.1),
                                      'colsample_bylevel': np.arange(0.1, 1.01, 0.1),
                                      'lambda': np.arange(0, 10, 1),
                                      'alpha': np.arange(0, 10, 1),
                                      'silent': np.array([1]),
                                      'seed': np.array([2016])
                                      }
        GbtreeRegression_grid_layer1 = {'booster': np.array(['gbtree']),
                                        'objective': np.array(['reg:linear']),
                                        'eta': np.array([0.1]),
                                        'gamma': np.arange(0,10,1),
                                        'max_depth': np.arange(2, 10, 1),
                                        'min_child_weight': np.arange(0, 10, 1),
                                        'max_delta_step': np.arange(0, 10, 1),
                                        'subsample': np.arange(0.1, 1.01, 0.1),
                                        'colsample_bytree': np.arange(0.1, 1.01, 0.1),
                                        'colsample_bylevel': np.arange(0.1, 1.01, 0.1),
                                        'lambda': np.arange(0, 10, 1),
                                        'alpha': np.arange(0, 10, 1),
                                        'silent': np.array([1]),
                                        'seed': np.array([2016])
                                        }
        GblinearLogistic_grid_layer1 = {'booster': np.array(['gblinear']),
                                        'objective': np.array(['binary:logistic']),
                                        'eta': np.array([0.1]),
                                        'lambda': np.arange(0, 10, 1),
                                        'alpha': np.arange(0, 10, 1),
                                        'lambda_bias': np.arange(0, 10, 1)
                                        }
        GblinearRegression_grid_layer1 = {'booster': np.array(['gblinear']),
                                          'objective': np.array(['reg:linear']),
                                          'eta': np.array([0.1]),
                                          'lambda': np.arange(0, 10, 1),
                                          'alpha': np.arange(0, 10, 1),
                                          'lambda_bias': np.arange(0, 10, 1)
                                          }
        ## Set up parameter grids for layer2. More conservative.
        GbtreeLogistic_grid_layer2 = {'booster': np.array(['gbtree']),
                                      'objective': np.array(['binary:logistic']),
                                      'eta': np.array([0.1]),
                                      'gamma': np.array([10]),
                                      'max_depth': np.array([2,3]),
                                      'min_child_weight': np.array([1]),
                                      'max_delta_step': np.array([10]),
                                      'subsample': np.array([1]),
                                      'colsample_bytree': np.arange(0.1, 1.01, 0.1),
                                      'colsample_bylevel': np.arange(0.1, 1.01, 0.1),
                                      'lambda': np.arange(5, 10, 2),
                                      'alpha': np.arange(5, 10, 2),
                                      'silent': np.array([1]),
                                      'seed': np.array([2016])
                                      }
        GbtreeRegression_grid_layer2 = {'booster': np.array(['gbtree']),
                                          'objective': np.array(['reg:linear']),
                                          'eta': np.array([0.1]),
                                          'gamma': np.array([10]),
                                          'max_depth': np.array([2,3]),
                                          'min_child_weight': np.array([1]),
                                          'max_delta_step': np.array([10]),
                                          'subsample': np.array([1]),
                                          'colsample_bytree': np.arange(0.1, 1.01, 0.1),
                                          'colsample_bylevel': np.arange(0.1, 1.01, 0.1),
                                          'lambda': np.arange(5, 10, 2),
                                          'alpha': np.arange(5, 10, 2),
                                          'silent': np.array([1]),
                                          'seed': np.array([2016])
                                          }
        GblinearLogistic_grid_layer2 = {'booster': np.array(['gblinear']),
                                        'objective': np.array(['binary:logistic']),
                                        'eta': np.array([0.1]),
                                        'lambda': np.arange(0, 20, 2),
                                        'alpha': np.arange(0, 20, 2),
                                        'lambda_bias': np.array([0])
                                        }
        GblinearRegression_grid_layer2 = {'booster': np.array(['gblinear']),
                                          'objective': np.array(['reg:linear']),
                                          'eta': np.array([0.1]),
                                          'lambda': np.arange(0, 20, 2),
                                          'alpha': np.arange(0, 20, 2),
                                          'lambda_bias': np.array([0])
                                          }


        # Change order of values
        GbtreeLogistic_grid_layer1 = shuffle_param(GbtreeLogistic_grid_layer1)
        GbtreeRegression_grid_layer1 = shuffle_param(GbtreeRegression_grid_layer1)
        GblinearLogistic_grid_layer1 = shuffle_param(GblinearLogistic_grid_layer1)
        GblinearRegression_grid_layer1 = shuffle_param(GblinearRegression_grid_layer1)
        GbtreeLogistic_grid_layer2 = shuffle_param(GbtreeLogistic_grid_layer2)
        GbtreeRegression_grid_layer2 = shuffle_param(GbtreeRegression_grid_layer2)
        GblinearLogistic_grid_layer2 = shuffle_param(GblinearLogistic_grid_layer2)
        GblinearRegression_grid_layer2 = shuffle_param(GblinearRegression_grid_layer2)

        # Select num_sets from each parameter space
        param_dict = {}
        if which_layer == "layer1":
            if model == 'GbtreeLogistic':
                param_dict = select_param(GbtreeLogistic_grid_layer1, num_sets, seed)
            elif model == 'GbtreeRegression':
                param_dict = select_param(GbtreeRegression_grid_layer1, num_sets, seed)
            elif model == 'GblinearLogistic':
                param_dict = select_param(GblinearLogistic_grid_layer1, num_sets, seed)
            elif model == 'GblinearRegression':
                param_dict = select_param(GblinearRegression_grid_layer1, num_sets, seed)
            else:
                raise ValueError('Model name not recognized')
        elif which_layer == "layer2":
            if model == 'GbtreeLogistic':
                param_dict = select_param(GbtreeLogistic_grid_layer2, num_sets, seed)
            elif model == 'GbtreeRegression':
                param_dict = select_param(GbtreeRegression_grid_layer2, num_sets, seed)
            elif model == 'GblinearLogistic':
                param_dict = select_param(GblinearLogistic_grid_layer2, num_sets, seed)
            elif model == 'GblinearRegression':
                param_dict = select_param(GblinearRegression_grid_layer2, num_sets, seed)
            else:
                raise ValueError('Model name not recognized')
        else:
            raise ValueError("Which layer not recognized, choose from `layer1`, `layer2`")
        # Seperate selected parameters into dictionary
        keys = param_dict.keys()
        param_dict_list = []
        for i in range(num_sets):
            temp_dict = {}
            for key in keys:
                temp_dict[key] = param_dict[key][i]
            param_dict_list.append(temp_dict)

        return param_dict_list
