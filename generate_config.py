"""
Generates the config JSON object for the experiment runfile.
"""
import os
import re
import json
import pathlib

import graph_models as gm

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-m', '--mode',
                    dest='mode',
                    help='mode',
                    default='simple')

if __name__ == '__main__':
    args = parser.parse_args()

# global, unchanging params

config_folder = 'configs'
F_OBJ = 10
F_OUT = 2

type_to_string = lambda t: re.search('^.+\.(\w+)\'>$', str(t))[1]

empty_config = {
    'setting': None,
    'expe_idx': None,
    'train_datasets': None,
    'train_dataset_indices': None,
    'test_datasets': None,
    'test_dataset_indices': None,
    'seeds': None,
    'hparams': None,
    'hparam_list': None,
    'load_dir': None,
    'save_dir': None,
    'models': None,
    'cuda': None
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
    if m == gm.DeepSet:
        return ([h] * 4, N, f_dict)
    elif m == gm.DeepSetPlus:
        return ([h] * 2, N, f_dict)
    elif m == gm.GNN_NAgg:
        return ([h] * 1, N, f_dict)

def get_default_simple_config(n_max=5, n_obj=5):
    simple_data_path = 'data/same_config_alt'
    d_path = os.listdir(simple_data_path)
    train = sorted(
        [p for p in d_path if re.search(r'^{}_.+_.+0$'.format(n_obj), p)])
    test = sorted(
        [p for p in d_path if re.search(r'^{}_.+_test$'.format(n_obj), p)])
    # to limit the size of the datasets used
    if not n_max == -1 and n_max <= len(train):
        train = train[:n_max]
    if not n_max == -1 and n_max <= len(test):
        test = test[:n_max]
    model_list = [gm.DeepSet, gm.DeepSetPlus, gm.GNN_NAgg]
    hparams = {
        'h': 16,
        'N': 1,
        'lr': 1e-3,
        'H': 16,
        'n_layers': 1,
        'n_epochs': 20}
    default_simple_config = {
        'setting': 'simple',
        'expe_idx': 0,
        'train_datasets': train,
        'train_dataset_indices': list(range(len(train))),
        'test_datasets': test,
        'test_dataset_indices': list(range(len(test))),
        'seeds': [1, 2, 3, 4, 5],
        'hparams': hparams,
        'hparam_list': [simple_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': 'data/same_config_alt',
        'save_dir': 'experimental_results/new',
        'models': [type_to_string(m) for m in model_list],
        'cuda': False,
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
    if m in [gm.AlternatingDoubleDS, gm.RecurrentGraphEmbeddingDS]:
        return ([h] * 4, N, f_dict)
    elif m in [gm.AlternatingDoubleRDS, gm.RecurrentGraphEmbeddingRDS]:
        return ([h] * 2, N, f_dict)
    elif m in [gm.AlternatingDouble, gm.RecurrentGraphEmbedding]:
        return ([h] * 1, N, f_dict)

def get_default_double_config(n_obj=5):
    double_data_path = 'data/compare_config_alt_cur'
    d_path = os.listdir(double_data_path)
    # change the following to deal with multiple curriculums
    train_cur = sorted([p for p in d_path if \
        re.search(r'^rotcur._{}.+0$'.format(n_obj), p)])
    test = ['rotcur4_{}_0_10000_test'.format(n_obj)]
    model_list = [
        gm.AlternatingDoubleDS,
        gm.AlternatingDoubleRDS,
        gm.AlternatingDouble,
        gm.RecurrentGraphEmbeddingDS,
        gm.RecurrentGraphEmbeddingRDS,
        gm.RecurrentGraphEmbedding,
    ]
    hparams = {
        'h': 16,
        'N': 2,
        'lr': 1e-3,
        'H': 16,
        'n_layers': 1,
        'n_epochs': 5}
    default_double_config = {
        'setting': 'double',
        'expe_idx': 1,
        'train_datasets': train_cur,
        'train_dataset_indices': [0] * len(train_cur),
        'test_datasets': test,
        'test_dataset_indices': [0],
        'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'hparams': hparams,
        'hparam_list': [double_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': 'data/compare_config_alt_cur',
        'save_dir': 'experimental_results/new',
        'models': [type_to_string(m) for m in model_list],
        'cuda': False,
    }
    return default_double_config

########### supplementary expes ##########

def get_easy_hard_config():
    config = get_default_simple_config()
    config['load_dir'] = 'data/same_config_alt'
    config['seeds'] = list(range(10))
    config['train_datasets'] = [
        'easy0_train',
        'easy1_train',
        '5_0_10000',
        'hard0_train',
        'hard1_train']
    config['train_dataset_indices'] = list(range(5))
    config['test_datasets'] = [
        'easy0_test',
        'easy1_test',
        '5_0_5000_test',
        'hard0_test',
        'hard1_test']
    config['test_dataset_indices'] = list(range(5))
    return config

def get_var_n_test_double_config():
    config = get_default_double_config(n_obj=5)
    config['load_model'] = True
    config['train_datasets'] = []
    config['train_dataset_indices'] = []
    config['test_datasets'] = [''] # TODO : fill in this
    config['test_dataset_indices'] = []

########################################

def save_config(config, config_id=-1):
    if config_id == -1:
        # search max config id
        paths = os.listdir(config_folder)
        search = lambda p: re.search(r'^config([0-9]+)$', p)
        config_id = max(
            [-1] + [int(search(p)[1]) for p in paths if search(p)]) + 1
    config['expe_idx'] = config_id
    path = os.path.join(config_folder, 'config%s' % config_id)
    with open(path, 'w') as f:
        f.write(json.dumps(config))

def export_config(mode, n_obj=5, config_id=-1):
    # if config_id is -1, just increments the max config found in the config 
    # folder
    pathlib.Path(config_folder).mkdir(parents=True, exist_ok=True)
    if mode == 'simple':
        config = get_default_simple_config(n_obj=n_obj)
    elif mode == 'double':
        config = get_default_double_config(n_obj=n_obj)
    elif mode == 'double_var_n_obj':
        n_obj_list = [10, 20]
        config = [get_default_double_config(n_obj=n) for n in n_obj_list]
    else:
        config = empty_config
    if isinstance(config, dict):
        save_config(config, config_id)
    elif isinstance(config, list):
        for c in config:
            save_config(c, -1)

def load_config(path):
    with open(path, 'r') as f:
        config = json.loads(f.readlines()[0])
    return config

if __name__ == '__main__':
    export_config(args.mode)