from os.path import join
import torch
import wandb
import pickle
import random
import numpy as np


# torch.manual_seed(1222)
# random.seed(159)
# np.random.seed(2333)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)



class Params():
    def __init__(self):
        pass

def set_params(data_name, task):
    params = Params()

    params.data_name = data_name

    if data_name == 'cn15k':
        params.VOCAB_SIZE = 15000
        params.REL_VOCAB_SIZE = 36
    elif data_name == 'nl27k':
        params.VOCAB_SIZE = 27221
        params.REL_VOCAB_SIZE = 417

    params.data_dir = join('./data', data_name)
    params.model_dir = join('./trained_models/', data_name)
    params.hr_map = load_hr_map(params.data_dir)

    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task == 'mse':
        params.early_stop = 'valid_mse'  # 'valid_mse' or 'valid_mae' or 'ndcg'
    else:
        params.early_stop = 'ndcg'

    params.whichmodel = 'bigumbelbox'
    if data_name == 'cn15k':
        if params.early_stop == 'valid_mse' or params.early_stop == 'valid_mae':
            params.DIM = 64
            params.NEG_PER_POS = 30
            params.LR = 1e-4
            params.EPOCH = 1000
            params.BATCH_SIZE = 4096
            params.regularization = {'delta': 1, 'min': 1e-3, 'rel_trans': 1e-3, 'rel_scale': 1e-3,
                                     'inverse': 0, 'transitive': 0.1, 'composite': 0}  # no composition rule for CN15k
            params.GUMBEL_BETA = 0.01  # gumbel box
            params.NEG_RATIO = 1
        elif params.early_stop == 'ndcg':
            params.DIM = 300
            params.NEG_PER_POS = 30
            params.LR = 1e-4
            params.EPOCH = 1000
            params.BATCH_SIZE = 2048
            params.regularization = {'delta': 0.5, 'min': 0, 'rel_trans': 0, 'rel_scale': 0,
                                     'inverse': 0, 'transitive': 0.1, 'composite': 0} # no composition rule for CN15k
            params.GUMBEL_BETA = 0.001  # gumbel box
            params.NEG_RATIO = 1
    elif data_name == 'nl27k':
        if params.early_stop == 'valid_mse' or params.early_stop == 'valid_mae':
            params.DIM = 64
            params.NEG_PER_POS = 30
            params.LR = 1e-4
            params.EPOCH = 1000
            params.BATCH_SIZE = 2048
            params.regularization = {'delta': 1, 'min': 1e-3, 'rel_trans': 1e-3, 'rel_scale': 1e-3,
                                     'inverse': 0, 'transitive': 0.1, 'composite': 0.1}
            params.GUMBEL_BETA = 0.01  # gumbel box
            params.NEG_RATIO = 1
        elif params.early_stop == 'ndcg':
            params.DIM = 150
            params.NEG_PER_POS = 30
            params.LR = 1e-4
            params.EPOCH = 1000
            params.BATCH_SIZE = 256
            params.regularization = {'delta': 0, 'min': 0, 'rel_trans': 0, 'rel_scale': 0,
                                     'inverse': 0, 'transitive': 0.1, 'composite': 0.1}
            params.GUMBEL_BETA = 0.0001  # gumbel box
            params.NEG_RATIO = 1


    # define RULE_CONFIGS
    if data_name == 'cn15k':
        params.RULE_CONFIGS = {
            'transitive': { # (a,r,b)^(b,r,c)=>(a,r,c)
                'use': True,
                'relations': [0, 3, 22],
            },
        }
    elif data_name == 'nl27k':
        params.RULE_CONFIGS = {
            'transitive': {
                'use': True,
                'relations': [272, 178, 294],
            },
            'composite':{
                'use': True,
                'relations': [(57, 35, 78)],
            }
        }

    return params


def wandb_initialize(params):
    run = wandb.init(
        project="boxprob",
        config={
            'model': params.whichmodel,
            'dim': params.DIM,
            'lr': params.LR,
            'batch_size': params.BATCH_SIZE,
            'regularization': params.regularization,
            'data': params.data_name,
            'rule_configs': params.RULE_CONFIGS,
            'epoch': params.EPOCH,
            'gumbel beta': params.GUMBEL_BETA,
            'neg_per_pos': params.NEG_PER_POS,
            'early_stop': params.early_stop
        }
    )
    return run

def load_hr_map(data_dir):
    file = join(data_dir, 'ndcg_test.pickle')
    with open(file, 'rb') as f:
        hr_map = pickle.load(f)
    return hr_map


