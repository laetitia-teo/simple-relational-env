"""
Generates the config JSON object for the experiment runfile.
"""
import os
import re
import json
import pathlib

import graph_models as gm
import baseline_models as bm

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-m', '--mode',
                    dest='mode',
                    help='mode',
                    default='simple')
parser.add_argument('-No', '--n-objects',
                    dest='n_obj',
                    help='number of objects',
                    default='3')
parser.add_argument('-Nm', '--n-objects-max',
                    dest='n_max',
                    help='number of objects',
                    default='8')
parser.add_argument('-l', '--load_model',
                    dest='l',
                    help='index of the config to load, if appropriate',
                    default='2')
parser.add_argument('-c', '--cuda',
                    dest='cuda',
                    help='if True, use gpu. Defaults to False',
                    default=None)
parser.add_argument('--c-id',
                    dest='config_id',
                    help='use this to override config id',
                    default=-1)
parser.add_argument('--hparam',
                    dest='hparam',
                    help='whether to perform hparam search',
                    default='')
parser.add_argument('-p', '--parallel',
                    dest='parallel',
                    help='parallelize experiment',
                    default='')
parser.add_argument('--cut',
                    dest='cut',
                    help='whether to use only a subset of length cut of the '\
                        +'train set, must be an integer',
                    default=None)

if __name__ == '__main__':
    args = parser.parse_args()

# global, unchanging params

config_folder = 'configs'
F_OBJ = 10
F_OUT = 2

type_to_string = lambda t: re.search(r'^.+\.(\w+)\'>$', str(t))[1]

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
        'f_out': F_OUT
        }
    return ([h] * n_layers, N, f_dict)

######## utility functions ########

def parallelize_config(config):
    """
    Takes in a config as returned by one of the functions below, and returns a
    config for each separate model.
    """
    configlist = []

    for h, m in zip(config['hparam_list'], config['models']):
        d = dict(config)
        d['models'] = [m]
        d['hparam_list'] = [h]
        configlist.append(d)
    
    return configlist

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
        'f_out': F_OUT
        }
    if m == gm.DeepSet:
        return ([h] * 4, N, f_dict)
    elif m in [gm.DeepSetPlus]:
        return ([h] * 2, N, f_dict)
    elif m in [gm.GNN_NAgg, gm.GNN_NAgg_NGI]:
        return ([h] * 1, N, f_dict)

def get_default_simple_config(
        n_min=3,
        n_max=8,
        cuda=False,
        simple_data_path='data/simple',
        restricted_models=False,
        cut=None):

    d_path = os.listdir(simple_data_path)

    train_dict = {}
    test_dict = {}
    flatten = lambda l: [e for sublist in l for e in sublist]

    for n_obj in range(n_min, n_max + 1):
        train_dict[n_obj - n_min] = sorted(
            [p for p in d_path if re.search(rf'^{n_obj}_.+_.+0$', p)])
        
        test_dict[n_obj - n_min] = sorted(
            [p for p in d_path if re.search(rf'^{n_obj}_.+_test$', p)])

    train = flatten(list(train_dict.values()))
    train_idx = flatten([[k] * len(v) for k, v in train_dict.items()])

    test = flatten(list(test_dict.values()))
    test_idx = flatten([[k] * len(v) for k, v in test_dict.items()])

    # to limit the size of the datasets used
    if not n_max == -1 and n_max <= len(train):
        train = train[:n_max]
    if not n_max == -1 and n_max <= len(test):
        test = test[:n_max]
    if restricted_models:
        model_list = [gm.GNN_NAgg]
    else:
        model_list = [
            gm.DeepSet,
            gm.DeepSetPlus,
            # gm.GNN_NAgg_NGI,
            gm.GNN_NAgg]

    hparams = {
        'n_objects': n_max,
        'h': 16,
        'N': 1,
        'lr': 1e-3,
        'H': 16,
        'n_layers': 1,
        'n_epochs': 20
        }
    default_simple_config = {
        'setting': 'simple',
        'expe_idx': 0,
        'train_datasets': train,
        'train_dataset_indices': train_idx,
        'test_datasets': test,
        'test_dataset_indices': test_idx,
        'seeds': list(range(10)),
        'hparams': hparams,
        'hparam_list': [simple_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': simple_data_path,
        'save_dir': 'experimental_results',
        'models': [type_to_string(m) for m in model_list],
        'cuda': cuda,
    }

    # for using only a subset of the train set
    if cut is not None:
        default_simple_config['cut'] = cut

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
        'f_out': F_OUT
        }
    if m in [
            gm.AlternatingDoubleDS,
            gm.RecurrentGraphEmbeddingDS,
            gm.ParallelDS]:
        return ([h] * 3, N, f_dict)
    elif m in [
            gm.AlternatingDoubleRDS,
            gm.RecurrentGraphEmbeddingRDS,
            gm.ParallelRDS]:
        return ([h] * 2, N, f_dict)
    elif m in [
            gm.AlternatingDouble,
            gm.RecurrentGraphEmbedding,
            gm.Parallel,
            gm.Parallel_NGI]:
        return ([h] * n_layers, N, f_dict)

def get_default_double_config(
        n_min=3,
        n_max=8,
        cuda=True,
        double_data_path='data/double',
        restricted_models=False,
        alternative_models=False,
        seeds=10,
        H=16,
        **kwargs):

    d_path = os.listdir(double_data_path)

    if n_max is None:
        n_max = n_min

    train_cur = sorted([p for p in d_path if \
        re.search(rf'^rotcur._{n_min}_{n_max}.+0$', p)])
    test = [p for p in d_path if \
        re.search(rf'^rotcur._{n_min}_{n_max}.+0_test$', p)]
    
    if restricted_models:
        model_list = [
            gm.Parallel,
        ]
    elif alternative_models:
        model_list = [
            gm.RecurrentGraphEmbedding,
            gm.RecurrentGraphEmbeddingRDS,
            gm.RecurrentGraphEmbeddingDS
        ]
    else:
        model_list = [
            gm.Parallel,
            # gm.Parallel_NGI,
            gm.ParallelRDS,
            gm.ParallelDS
        ]
    hparams = {
        'n_objects': n_max,
        'h': H,
        'N': 2,
        'lr': 1e-3,
        'H': H,
        'n_layers': 1,
        'n_epochs': 5
        }
    default_double_config = {
        'setting': 'double',
        'expe_idx': 1,
        'train_datasets': train_cur,
        'train_dataset_indices': [0] * len(train_cur),
        'test_datasets': test,
        'test_dataset_indices': [0],
        'seeds': list(range(seeds)),
        'hparams': hparams,
        'hparam_list': [double_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': double_data_path,
        'save_dir': 'experimental_results',
        'models': [type_to_string(m) for m in model_list],
        'cuda': cuda,
    }

    # for using only a subset of the train set
    if cut is not None:
        default_double_config['cut'] = cut

    return default_double_config

def get_double_parallel_config(n_obj=5, cuda=False):
    config = get_default_double_config(n_obj=n_obj)
    configlist = []
    for h, m in zip(config['hparam_list'], config['models']):
        d = dict(config)
        d['models'] = [m]
        d['hparam_list'] = [h]
        configlist.append(d)
    return configlist

########### MLP baselines ################

def simple_baseline_hparam_fn(*args, **kwargs):
    m = args[0]
    H = kwargs['H']
    h = kwargs['h']
    N = kwargs['N']
    n_obj = kwargs['n_objects']
    n_layers = kwargs['n_layers']
    f_dict = {
        'f_x': F_OBJ,
        'f_e': F_OBJ,
        'f_u': F_OBJ,
        'h': H,
        'f_out': F_OUT
        }
    if m == bm.NaiveMLP:
        return (n_obj, f_dict['f_x'], [h] * n_layers)
    elif m == bm.NaiveLSTM:
        return (f_dict['f_x'], h, [h] * n_layers)

def get_simple_baseline_config(n_min=3, n_max=8, cuda=False, **kwargs):

    config = get_default_simple_config(
        n_min=n_min,
        n_max=n_max,
        cuda=cuda,
        **kwargs)

    model_list = [
        bm.NaiveMLP,
    ]

    mid = int((n_min + n_max)/2)

    hparams = {
        'n_objects': n_max,
        'h': mid * F_OBJ,
        'N': 1,
        'lr': 1e-3,
        'H': mid * F_OBJ,
        'n_layers': 2,
        'n_epochs': 20
        }

    config['hparams'] = hparams

    config['models'] = [type_to_string(m) for m in model_list]

    config['hparam_list'] = [
        simple_baseline_hparam_fn(m, **config['hparams']) for m in model_list]
    
    return config

def double_baseline_hparam_fn(*args, **kwargs):
    m = args[0]
    H = kwargs['H']
    h = kwargs['h']
    N = kwargs['N']
    n_obj = kwargs['n_objects']
    n_layers = kwargs['n_layers']
    f_dict = {
        'f_x': F_OBJ,
        'f_e': F_OBJ,
        'f_u': F_OBJ,
        'h': H,
        'f_out': F_OUT
        }
    if m == bm.DoubleNaiveMLP:
        return (n_obj, f_dict['f_x'], [2 * h] * n_layers)
    elif m == bm.SceneMLP:
        return (n_obj, f_dict['f_x'], [2 * h] * n_layers, 2 * H, [h] * n_layers)
    elif m in [bm.DoubleNaiveLSTM, bm.SceneLSTM]:
        return (f_dict['f_x'], 2 * h, [2 * h] * n_layers)

def get_double_baseline_config(
        n_min=3,
        n_max=8,
        cuda=False,
        seeds=10,
        **kwargs):

    config = get_default_double_config(
        n_min=n_min,
        n_max=n_max,
        cuda=False,
        seeds=seeds,
        **kwargs)

    model_list = [
        bm.DoubleNaiveMLP,
    ]
    
    mid = int((n_min + n_max)/2)

    hparams = {
        'n_objects': n_max,
        'h': mid * F_OBJ,
        'N': 1,
        'lr': 1e-3,
        'H': mid * F_OBJ,
        'n_layers': 2,
        'n_epochs': 5
    }

    config['models'] = [type_to_string(m) for m in model_list]
    config['hparams'] = hparams
    config['hparam_list'] = [
        double_baseline_hparam_fn(m, **hparams) for m in model_list]

    return config

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
        'hard1_train'
        ]
    config['train_dataset_indices'] = list(range(5))
    config['test_datasets'] = [
        'easy0_test',
        'easy1_test',
        '5_0_5000_test',
        'hard0_test',
        'hard1_test'
        ]
    config['test_dataset_indices'] = list(range(5))
    return config

def get_var_n_test_double_config(n_test, n_obj=5, cuda=False):
    config = get_default_double_config(n_obj=n_obj)
    double_data_path = 'data/comparison'
    d_path = os.listdir(double_data_path)
    test = sorted(
        [p for p in d_path if re.search(r'^testdouble_([0-9]+).*0$', p)],
        key=lambda p: int(re.search(r'^testdouble_([0-9]+).*0$', p)[1]))
    config['preload_model'] = True
    config['load_idx'] = 2
    config['train_datasets'] = []
    config['train_dataset_indices'] = []
    config['test_datasets'] = test
    config['test_dataset_indices'] = [0] * len(test)
    return config

########### rebuttal expes ##########

def get_parallel_double_config(n_obj=5, cuda=False):
    double_data_path = 'data/comparison'
    d_path = os.listdir(double_data_path)
    train_cur = sorted([p for p in d_path if \
        re.search(r'^rotcur._{}.+0$'.format(n_obj), p)])
    test = [p for p in d_path if \
        re.search(r'^rotcur._{}.+0_test$'.format(n_obj), p)][0]
    model_list = [
        gm.Parallel,
        gm.AlternatingDouble,
        gm.RecurrentGraphEmbedding
    ]
    hparams = {
        'n_objects': n_obj,
        'h': 16,
        'N': 2,
        'lr': 1e-3,
        'H': 16,
        'n_layers': 1,
        'n_epochs': 5
        }
    default_double_config = {
        'setting': 'double',
        'expe_idx': 1,
        'train_datasets': train_cur,
        'train_dataset_indices': [0] * len(train_cur),
        'test_datasets': test,
        'test_dataset_indices': [0],
        'seeds': [0, 1, 2, 3, 4],
        'hparams': hparams,
        'hparam_list': [double_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': 'data/comparison',
        'save_dir': 'experimental_results',
        'models': [type_to_string(m) for m in model_list],
        'cuda': cuda,
    }
    return default_double_config

def double_lstm_hparam_fn(*args, **kwargs):
    m = args[0]
    H = kwargs['H']
    h = kwargs['h']
    N = kwargs['N']
    n_obj = kwargs['n_objects']
    n_layers = kwargs['n_layers']
    f_dict = {
        'f_x': F_OBJ,
        'f_e': F_OBJ,
        'f_u': F_OBJ,
        'h': H,
        'f_out': F_OUT
        }
    return (f_dict['f_x'], h, [2 * h] * n_layers)

def get_lstm_double_config(n_obj=5, cuda=False):
    double_data_path = 'data/comparison'
    d_path = os.listdir(double_data_path)
    train_cur = sorted([p for p in d_path if \
        re.search(r'^rotcur._{}.+0$'.format(n_obj), p)])
    test = [p for p in d_path if \
        re.search(r'^rotcur._{}.+0_test$'.format(n_obj), p)][0]
    model_list = [
        bm.SceneLSTM
    ]
    hparams = {
        'n_objects': n_obj,
        'h': 16,
        'N': 2,
        'lr': 1e-3,
        'H': 16,
        'n_layers': 2,
        'n_epochs': 5
        }
    default_double_config = {
        'setting': 'double',
        'expe_idx': 1,
        'train_datasets': train_cur,
        'train_dataset_indices': [0] * len(train_cur),
        'test_datasets': test,
        'test_dataset_indices': [0],
        'seeds': [0, 1, 2, 3, 4],
        'hparams': hparams,
        'hparam_list': \
            [double_lstm_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': 'data/comparison',
        'save_dir': 'experimental_results',
        'models': [type_to_string(m) for m in model_list],
        'cuda': cuda,
    }
    return default_double_config

def get_varnobj_double_config(n_obj_min=3, n_obj_max=8, cuda=False):
    double_data_path = 'data/comparison'
    d_path = os.listdir(double_data_path)
    train_cur = sorted([p for p in d_path if \
        re.search(
            r'^rotcur[0-9]+_{0}_{1}_100000$'.format(
                n_obj_min, n_obj_max), p)])
    test = [f'rotcur4_{n_obj_min}_{n_obj_max}_100000_test']
    model_list = [
        gm.AlternatingDouble,
        gm.RecurrentGraphEmbedding
    ]
    hparams = {
        'n_objects': (n_obj_min, n_obj_max),
        'h': 16,
        'N': 2,
        'lr': 1e-3,
        'H': 16,
        'n_layers': 1,
        'n_epochs': 5
        }
    default_double_config = {
        'setting': 'double',
        'expe_idx': 1,
        'train_datasets': train_cur,
        'train_dataset_indices': [0] * len(train_cur),
        'test_datasets': test,
        'test_dataset_indices': [0],
        'seeds': [0, 1, 2, 3, 4],
        'hparams': hparams,
        'hparam_list': [double_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': 'data/comparison',
        'save_dir': 'experimental_results',
        'models': [type_to_string(m) for m in model_list],
        'cuda': cuda,
    }
    return default_double_config

def get_big_mlp_simple_config(n_obj=5, cuda=False):
    simple_data_path = 'data/recognition'
    d_path = os.listdir(simple_data_path)
    train = sorted(
        [p for p in d_path if re.search(r'^{}_.+_.+0$'.format(n_obj), p)])
    test = sorted(
        [p for p in d_path if re.search(r'^{}_.+_test$'.format(n_obj), p)])
    model_list = [
        bm.NaiveMLP
    ]
    hparams = {
        'n_objects': n_obj,
        'h': n_obj * F_OBJ,
        'N': 2,
        'lr': 1e-3,
        'H': n_obj * F_OBJ,
        'n_layers': 2,
        'n_epochs': 20
        }
    config = {
        'setting': 'simple',
        'expe_idx': 0,
        'train_datasets': train,
        'train_dataset_indices': list(range(len(train))),
        'test_datasets': test,
        'test_dataset_indices': list(range(len(test))),
        'seeds': [0, 1, 2, 3, 4],
        'hparams': hparams,
        'hparam_list': \
            [simple_baseline_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': 'data/recognition',
        'save_dir': 'experimental_results',
        'models': [type_to_string(m) for m in model_list],
        'cuda': cuda,
    }
    return config

def get_mpgnn_simple_config(n_obj=5, cuda=False):
    simple_data_path = 'data/recognition'
    d_path = os.listdir(simple_data_path)
    train = sorted(
        [p for p in d_path if re.search(r'^{}_.+_.+0$'.format(n_obj), p)])
    test = sorted(
        [p for p in d_path if re.search(r'^{}_.+_test$'.format(n_obj), p)])
    model_list = [
        gm.GNN_NAgg
    ]
    hparams = {
        'n_objects': n_obj,
        'h': 16,
        'N': 2,
        'lr': 1e-3,
        'H': 16,
        'n_layers': 2,
        'n_epochs': 20
        }
    config = {
        'setting': 'simple',
        'expe_idx': 0,
        'train_datasets': train,
        'train_dataset_indices': list(range(len(train))),
        'test_datasets': test,
        'test_dataset_indices': list(range(len(test))),
        'seeds': [0, 1, 2, 3, 4],
        'hparams': hparams,
        'hparam_list': \
            [simple_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': 'data/recognition',
        'save_dir': 'experimental_results',
        'models': [type_to_string(m) for m in model_list],
        'cuda': cuda,
    }
    return config

def get_big_mlp_double_config(n_obj=5, cuda=False):
    double_data_path = 'data/comparison'
    d_path = os.listdir(double_data_path)
    train_cur = sorted([p for p in d_path if \
        re.search(
            r'^rotcur[0-9]+_{0}_{1}_100000$'.format(
                n_obj, n_obj), p)])
    test = [f'rotcur4_{n_obj}_{n_obj}_100000_test']
    model_list = [
        bm.DoubleNaiveMLP
    ]
    hparams = {
        'n_objects': n_obj,
        'h': 2 * n_obj * F_OBJ,
        'N': 2,
        'lr': 1e-3,
        'H': n_obj * F_OBJ,
        'n_layers': 2,
        'n_epochs': 20
        }
    default_double_config = {
        'setting': 'double',
        'expe_idx': 1,
        'train_datasets': train_cur,
        'train_dataset_indices': [0] * len(train_cur),
        'test_datasets': test,
        'test_dataset_indices': [0],
        'seeds': [0, 1, 2, 3, 4],
        'hparams': hparams,
        'hparam_list': \
            [double_baseline_hparam_fn(m, **hparams) for m in model_list],
        'load_dir': 'data/comparison',
        'save_dir': 'experimental_results',
        'models': [type_to_string(m) for m in model_list],
        'cuda': cuda,
    }
    return default_double_config

######### perturb/abstract expes #######

########################################

def save_config(config, config_id=-1):
    if config_id == -1:
        # search max config id
        paths = os.listdir(config_folder)
        search = lambda p: re.search(r'^config([0-9]+)$', p)
        config_id = max(
            [-1] + [int(search(p)[1]) for p in paths if search(p)]) + 1
    config['expe_idx'] = config_id
    print(config_id)
    path = os.path.join(config_folder, 'config%s' % config_id)
    with open(path, 'w') as f:
        f.write(json.dumps(config))

def export_config(
        mode,
        config_id=-1,
        n_test=None,
        **kwargs):
    """
    if config_id is -1, just increments the max config found in the config 
    folder
    if parallel is True, then we have several config files they are counted as
    the same expe, and saved in the same directory.
    """
    pathlib.Path(config_folder).mkdir(parents=True, exist_ok=True)
    if mode == 'simple':
        config = get_default_simple_config(**kwargs)
    elif mode == 'double':
        config = get_default_double_config(**kwargs)
    elif mode == 'double_var_n_obj':
        n_obj_list = [10, 20]
        config = [get_default_double_config(n_obj=n, **kwargs) for n in n_obj_list]
    elif mode == 'double_parallel':
        config = get_double_parallel_config(**kwargs)
    elif mode == 'test_double':
        if n_test is None:
            raise Exception('Please provide a configuration file index to ' +\
                'load from using the -l flag.')
        config = get_var_n_test_double_config(n_test=2, **kwargs)
    elif mode == 'easy_hard':
        config = get_easy_hard_config()
    elif mode == 'baseline_simple':
        config = get_simple_baseline_config(**kwargs)
    elif mode == 'baseline_double':
        config = get_double_baseline_config(**kwargs)
    elif mode == 'parallel':
        config = get_parallel_double_config(**kwargs)
    elif mode == 'lstm':
        config = get_lstm_double_config(**kwargs)
    elif mode == 'var':
        config = get_varnobj_double_config(**kwargs)
    elif mode == 'bigmlp_s':
        config = get_big_mlp_simple_config(**kwargs)
    elif mode == 'mpgnn_s':
        config = get_mpgnn_simple_config(**kwargs)
    elif mode == 'bigmlp_d':
        config = get_big_mlp_double_config(**kwargs)
    # perturb/abstract
    elif mode == 'simple_perturb':
        config = get_default_simple_config(
            simple_data_path='data/simple_perturb',
            **kwargs)
    elif mode == 'simple_abstract':
        config = get_default_simple_config(
            simple_data_path='data/simple_abstract',
            **kwargs)
    elif mode == 'simple_distractors':
        config = get_default_simple_config(
            simple_data_path='data/simple_distractors',
            **kwargs)
    elif mode == 'simple_abstract_distractors':
        config = get_default_simple_config(
            simple_data_path='data/simple_abstract_distractors',
            **kwargs)
    elif mode == 'double_perturb':
        config = get_default_double_config(
            double_data_path='data/double_perturb',
            **kwargs)
    elif mode == 'double_abstract':
        config = get_default_double_config(
            double_data_path='data/double_abstract',
            **kwargs)
    elif mode == 'double_distractors':
        config = get_default_double_config(
            double_data_path='data/double_distractors',
            **kwargs)
    elif mode == 'double_abstract_distractors':
        config = get_default_double_config(
            double_data_path='data/double_abstract_distractors',
            **kwargs)
    elif mode == 'baseline_simple_abstract':
        config = get_simple_baseline_config(
            simple_data_path='data/simple_abstract',
            **kwargs)
    elif mode == 'baseline_simple_distractors':
        config = get_simple_baseline_config(
            simple_data_path='data/simple_distractors',
            **kwargs)
    elif mode == 'baseline_simple_abstract_distractors':
        config = get_simple_baseline_config(
            simple_data_path='data/simple_abstract_distractors',
            **kwargs)
    elif mode == 'baseline_double_abstract':
        config = get_double_baseline_config(
            double_data_path='data/double_abstract',
            **kwargs)
    elif mode == 'baseline_double_distractors':
        config = get_double_baseline_config(
            double_data_path='data/double_distractors',
            **kwargs)
    elif mode == 'baseline_double_abstract_distractors':
        config = get_double_baseline_config(
            double_data_path='data/double_abstract_distractors',
            **kwargs)
    else:
        config = empty_config
    if isinstance(config, dict):
        
        # parallel experiment
        if args.parallel:
            configlist = parallelize_config(config)
            for i, c in enumerate(configlist):
                if cuda:
                    config['cuda'] = True
                if config_id == -1:
                    save_config(c, -1)
                else:
                    save_config(c, config_id + i)
        # serial experiment
        else:
            if cuda:
                config['cuda'] = True
            save_config(config, config_id)

    elif isinstance(config, list):
        for c in config:
            if cuda:
                c['cuda'] = True
            save_config(c, -1)

def load_config(path):
    with open(path, 'r') as f:
        config = json.loads(f.readlines()[0])
        for key in config.keys():
            try:
                config[key] = int(config[key])
            except:
                pass
    return config

if __name__ == '__main__':
    try:
        n = int(args.n_obj)
    except ValueError:
        print('Please provide a valid number for the -n flag.')
        raise
    n_test = args.l
    cuda = args.cuda is not None
    config_id = int(args.config_id)
    
    if args.cut is None:
        cut = None
    else:
        cut = int(args.cut)

    hparam = args.hparam

    export_config(
        args.mode,
        n_test=n_test,
        cuda=cuda,
        config_id=config_id,
        n_min=int(args.n_obj),
        n_max=int(args.n_max),
        cut=cut)
    
    from make_runfile import *