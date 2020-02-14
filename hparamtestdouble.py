"""
This module performs hyperparameter testing, and validation of our models.
"""
import time
import re
import os
import pathlib
import numpy as np
import torch

import graph_models as gm

from argparse import ArgumentParser

from gen import SameConfigGen
from dataset import collate_fn
from graph_utils import data_to_graph_simple

from run_utils import load_dl, one_run, nparams
from graph_utils import data_to_graph_double
# script arguments

parser = ArgumentParser()
parser.add_argument('-m', '--mode',
                    dest='mode',
                    help='mode : \'all\' for all available models, index of the'
                        + ' model for a single model',
                    default='some')
parser.add_argument('-d', '--directory',
                    dest='directory',
                    help='path of the save and log directory',
                    default='experimental_results/compare_config_alt_cur')
parser.add_argument('-r', '--run-index',
                    dest='run_idx',
                    help='index of the run',
                    default='N1')
parser.add_argument('-c', '--curriculum',
                    dest='cur',
                    help='whether to use a curriculum of rotations',
                    default='yep')

args = parser.parse_args()

# global params

B_SIZE = 128
# L_RATE = 1e-3
N_EPOCHS = 1
F_OBJ = 10
# H = 16
# N = 1
F_OUT = 2


# dict of hparams and their possible values

hparams = {
    'n_layers': [1, 2],
    'h': [F_OBJ, 16, 32], # size of hidden layer
    'lr': [10e-4, 5*10e-4, 10e-3],
    'N': [1, 2, 3]
}

# default hparams

n_layers = 1
h = 16
lr = 1e-3
N = 1
seeds = [5, 6, 7, 8, 9, 10]
n_epochs = 5
H = 16

f_dict = {
    'f_x': F_OBJ,
    'f_e': F_OBJ,
    'f_u': F_OBJ,
    'h': H,
    'f_out': F_OUT}

# data paths

prefix = 'data/compare_config_alt'

d_path = os.listdir(prefix)
train_5 = sorted([p for p in d_path if re.search(r'^5_.+_10{5}$', p)])
val_5 = sorted([p for p in d_path if re.search(r'^5_.+10{4}_val$', p)])
train_3 = sorted([p for p in d_path if re.search(r'^3_.+_10{5}$', p)])
val_3 = sorted([p for p in d_path if re.search(r'^3_.+10{4}_val$', p)])
train_norot = sorted([p for p in d_path if re.search(r'^norot_.+_10{5}$', p)])
val_norot = sorted([p for p in d_path if re.search(r'^norot_.+10{4}_val$', p)])
train_newrot = sorted([p for p in d_path if re.search(r'^newrot2_.+_10{5}$', p)])
val_newrot = sorted([p for p in d_path if re.search(r'^newrot2_.+10{4}_val$', p)])

cur_prefix = 'data/compare_config_alt_cur'

cur_d_path = os.listdir(cur_prefix)
train_cur = sorted([p for p in cur_d_path if re.search(r'^rotcur.+0$', p)])
val_cur = sorted([p for p in cur_d_path if re.search(r'^rotcur.+_val$', p)])
val = 'rotcur4_5_0_10000_val'

# train_10 = sorted([p for p in d_path if re.search(r'^10_.+_10{4}$', p)])
# val_10 = sorted([p for p in d_path if re.search(r'^10_.+_val$', p)])
# train_20 = sorted([p for p in d_path if re.search(r'^20_.+_10{4}$', p)])
# val_20 = sorted([p for p in d_path if re.search(r'^20_.+_val$', p)])
params = ([h] * 1, N, f_dict)

params1 = ([h] * 1, N, f_dict)
params2 = ([h] * 2, N, f_dict)
params3 = ([h] * 3, N, f_dict)

param_dict = {
    0: params1,
    2: params3,
    3: params1,
    5: params3,
    8: params3,
    9: params3
}

# for quick testing purposes

dl = load_dl(os.path.join(prefix, train_5[0]), double=True)
data = next(iter(dl))
g1, g2 = data_to_graph_double(data)

def run(m_idx, run_idx):
    dset = 0
    print('model number %s' % m_idx)
    print('model name %s' % gm.model_list_double[m_idx].__name__)
    for dpath_train, dpath_val in zip(train_5, val_5):
        print('dset %s;' % dset)
        t0 = time.time()
        dl_train = load_dl(
            os.path.join(prefix, dpath_train), double=True)
        dl_val = load_dl(
            os.path.join(prefix, dpath_val), double=True)
        path = os.path.join(
            args.directory, 'run%s' % run_idx, 'model' + str(m_idx))
        pathlib.Path(
            os.path.join(path, 'data')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(path, 'models')).mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = gm.model_list_double[m_idx](*params)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            one_run(
                dset,
                seed,
                n_epochs,
                model,
                opt,
                dl_train,
                dl_val,
                path,
                cuda=False)
        t = time.time()
        print('total running time for one ds %s seconds' % str(t - t0))
        dset += 1

def cur_run(m_idx, run_idx, params=params):
    """
    For using a curriculum of dataloaders.
    """
    dlist = [load_dl(os.path.join(cur_prefix, p), double=True) \
        for p in train_cur]
    dl_val = load_dl(os.path.join(cur_prefix, val), double=True)
    dset = 0
    path = os.path.join(
        args.directory, 'run%s' % run_idx, 'model' + str(m_idx))
    pathlib.Path(
        os.path.join(path, 'data')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(
        os.path.join(path, 'models')).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = gm.model_list_double[m_idx](*params)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        # res = model(g1, g2)
        one_run(
            dset,
            seed,
            n_epochs,
            model,
            opt,
            dlist,
            dl_val,
            path,
            cuda=False)
        t = time.time()
        print('total running time for one seed %s seconds' % str(t - t0))
    # dset += 1

# 5 objects

if __name__ == '__main__':
    if args.run_idx is None:
        raise Exception('No run index was provided, please use the -r flag')
    if args.mode == 'all':
        for m_idx in range(len(gm.model_list_double)):
            if not args.cur:
                run(m_idx, args.run_idx)
            else:
                cur_run(m_idx, args.run_idx)
    elif args.mode == 'some':
        indices = [0, 2, 3, 5, 8, 9]
        for m_idx in indices:
            if not args.cur:
                run(m_idx, args.run_idx)
            else:
                params = param_dict[m_idx]
                cur_run(m_idx, args.run_idx, params=params)
    else:
        try:
            m_idx = int(args.mode)
            if not args.cur:
                run(m_idx, args.run_idx)
            else:
                cur_run(m_idx, args.run_idx)
        except ValueError:
            print('Invalid mode for the script, must be \'all\' or integer')
            raise