"""
Generates the config JSON object for the experiment runfile.
"""
import os
import json
import graph_models as gm

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-m', '--mode',
                    dest='mode',
                    help='mode',
                    default='simple')

# global, unchanging params

config_folder = 'configs'
F_OBJ = 10
F_OUT = 2

empty_config = {
    'setting': None,
    'train_datasets': None,
    'train_dataset_indices': None,
    'test_datasets': None,
    'seeds': None,
    'hparams': None,
    'hparam_fn': None,
    'load_dir': None,
    'save_dir': None,
    'models': None
}

def default_hparam_fn(*args, **kwargs):
    H = kwargs['H']
    h = kwargs['h']
    N = kwargs['N']
    n_layers = kwargs['n_layers']
    f_dict = {
        'f_x': F_OBJ,
        'f_e': F_OBJ,
        'f_u': F_OBJ,
        'h': H,
        'f_out': F_OUT}
    return ([h] * n_layers, N, f_dict)

######### simple setting ##########

def simple_hparam_fn(*args, **kwargs):
    m = args[0]
    H = kwargs['H']
    h = kwargs['h']
    N = kwargs['N']
    n_layers = kwargs['n_layers']
    f_dict = {
        'f_x': F_OBJ,
        'f_e': F_OBJ,
        'f_u': F_OBJ,
        'h': H,
        'f_out': F_OUT}
    if isinstance(m, gm.DeepSet):
        return ([h] * 4, N, f_dict)
    elif isinstance(m, gm.DeepSetplus):
        return ([h] * 2, N, f_dict)
    elif isinstance(m, gm.GNN_NAgg):
        return ([h] * 1, N, f_dict)

def get_default_simple_config(n_max=5, n_obj=5):
    simple_data_path = 'data/same_config_alt'
    d_path = os.listdir(simple_data_path)
    train = sorted(
        [p for p in d_path if re.search(r'^{}_.+_10{4}$'.format(n_obj), p)])
    test = sorted(
        [p for p in d_path if re.search(r'^{}_.+_test$'.format(n_obj), p)])
    # to limit the size of the datasets used
    if not n_max == -1 and n_max <= len(train):
        train = train[:n_max]
    if not n_max == -1 and n_max <= len(test):
        test = test[:n_max]
    default_simple_config = {
        'setting': 'simple',
        'train_datasets': train,
        'train_dataset_indices': list(range(len(train))),
        'test_datasets': test
        'seeds': [1, 2, 3, 4, 5],
        'hparams': {h: 16, N: 1; lr: 1e-3, H: 16, n_layers: 1, n_epochs=20},
        'hparam_fn': simple_hparam_fn,
        'load_dir': 'data/same_config_alt',
        'save_dir': 'experimental_results/same_config_alt',
        'models': [gm.DeepSet, gm.DeepSetPlus, gm.GNN_NAgg]
    }
    return default_simple_config

########### double setting ##########

def double_hparam_fn(*args, **kwargs):
    m = args[0]
    H = kwargs['H']
    h = kwargs['h']
    N = kwargs['N']
    n_layers = kwargs['n_layers']
    f_dict = {
        'f_x': F_OBJ,
        'f_e': F_OBJ,
        'f_u': F_OBJ,
        'h': H,
        'f_out': F_OUT}
    if m.component == 'DS':
        return ([h] * 4, N, f_dict)
    elif m.component == 'RDS':
        return ([h] * 2, N, f_dict)
    elif m.component == 'MPGNN':
        return ([h] * 1, N, f_dict)

def get_default_double_config(n_obj=5):
    double_data_path = 'data/compare_config_alt_cur'
    d_path = os.listdir(double_data_path)
    # change the following to deal with multiple curriculums
    train_cur = sorted([p for p in d_path if re.search(r'^rotcur.+0$', p)])
    test = 'rotcur4_5_0_10000_test'
    # to limit the size of the datasets used
    if not n_max == -1 and n_max <= len(train):
        train = train[:n_max]
    if not n_max == -1 and n_max <= len(test):
        test = test[:n_max]
    model_list = [
        AlternatingDoubleDS,
        AlternatingDoubleRDS,
        AlternatingDouble,
        RecurrentGraphEmbeddingDS
        RecurrentGraphEmbeddingRDS,
        RecurrentGraphEmbedding,
    ]
    default_double_config = {
        'setting': 'double',
        'train_datasets': train,
        'train_dataset_indices': [0],
        'test_datasets': test
        'seeds': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'hparams': {h: 16, N: 1; lr: 1e-3, H: 16, n_layers: 1, n_epochs=5},
        'hparam_fn': double_hparam_fn,
        'load_dir': 'data/compare_config_alt_cur',
        'save_dir': 'experimental_results/compare_config_alt_cur',
        'models': model_list
    }
    return default_double_config
