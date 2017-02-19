"""
Pre-defined model.
"""

class definedModel(object):
    """
    Pre-defined models and their default setting.
    """
    def __init__(self):
        self.__DEFINED_MODEL_TYPE = ['GbtreeLogistic','GbtreeRegression',
                                        'GblinearLogistic','GblinearRegression']

    def model_type(self):
        return self.__DEFINED_MODEL_TYPE

    def model_param(self,model_name):
        self.validate_model_type(model_name)
        param = {}
        if model_name == 'GbtreeLogistic':
            # define model parameter
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
        elif model_name == 'GblinearLogistic':
             # define model parameter
             param = {'objective':'binary:logistic',
                     'booster' : 'gblinear',
                     'eta' : 0.2,
                     'lambda' : 0.1,
                     'alpha' : 0.001,
                     'silent':1,
                     'seed' : 2016
                    }
        elif model_name == 'GbtreeRegression':
             # define model parameter
             param = {'objective':'reg:linear',
                     'booster' : 'gbtree',
                     'eta' : 0.2,
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
        elif model_type_writeout == 'GblinearRegression':
             # define model parameter
             param = {'objective':'reg:linear',
                     'booster' : 'gblinear',
                     'eta' : 0.2,
                     'lambda' : 0.1,
                     'alpha' : 0.001,
                     'silent':1,
                     'seed' : 2016
                     }
        return param


    def validate_model_type(self,model_name):
        if not model_name in self.__DEFINED_MODEL_TYPE:
            raise ValueError('Defined models are: ' + ', '.join(self.__DEFINED_MODEL_TYPE))
