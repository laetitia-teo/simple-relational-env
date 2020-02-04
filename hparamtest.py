"""
This module performs hyperparameter testing, and validation of our models.
"""
import time
import re
import os
import pathlib
import numpy as np
import torch

import graph_models_v2 as gm

from argparse import ArgumentParser

from gen import SameConfigGen
from dataset import collate_fn
from graph_utils import data_to_graph_simple

from run_utils import load_dl, one_run, data_to_graph_simple

# script arguments

parser = ArgumentParser()
parser.add_argument('-m', '--mode',
                    dest='mode',
                    help='mode : \'all\' for all available models, index of the'
                        + ' model for a single model',
                    default='1')
parser.add_argument('-d', '--directory',
                    dest='directory',
                    help='path of the save and log directory',
                    default='experimental_results/same_config_alt')
parser.add_argument('-r', '--run-index',
                    dest='run_idx',
                    help='index of the run',
                    default='base')
parser.add_argument('-l', '--list-mode',
                    dest='list_mode',
                    default='all')

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
seeds = [0, 1, 2, 3, 4]
n_epochs = 20
H = 16

f_dict = {
    'f_x': F_OBJ,
    'f_e': F_OBJ,
    'f_u': F_OBJ,
    'h': H,
    'f_out': F_OUT}

# data paths

prefix = 'data/same_config_alt'

d_path = os.listdir(prefix)
train_5 = sorted([p for p in d_path if re.search(r'^5_.+_10{4}$', p)])[:5]
val_5 = sorted([p for p in d_path if re.search(r'^5_.+_val$', p)])[:5]
train_10 = sorted([p for p in d_path if re.search(r'^10_.+_10{4}$', p)])[:5]
val_10 = sorted([p for p in d_path if re.search(r'^10_.+_val$', p)])[:5]
train_20 = sorted([p for p in d_path if re.search(r'^20_.+_10{4}$', p)])[:5]
val_20 = sorted([p for p in d_path if re.search(r'^20_.+_val$', p)])[:5]

params = ([h] * n_layers, N, f_dict)

# for quick testing purposes

# data_fn = data_to_graph_simple
# dl = load_dl('data/same_config_alt/5_0_10000')
# data = next(iter(dl))
# graph = data_fn(data)
# m = gm.TGNN(*params)

def run(m_idx, run_idx, params=params, list_mode='all'):
    dset = 0
    print('model number %s' % m_idx)
    print('model name %s' % gm.model_list[m_idx].__name__)
    for dpath_train, dpath_val in zip(train_5, val_5):
        print('dset %s;' % dset)
        t0 = time.time()
        dl_train = load_dl(
            os.path.join(prefix, dpath_train))
        dl_val = load_dl(
            os.path.join(prefix, dpath_val))
        path = os.path.join(
            args.directory, 'run%s' % run_idx, 'model' + str(m_idx))
        pathlib.Path(
            os.path.join(path, 'data')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(
            os.path.join(path, 'models')).mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = gm.model_list[m_idx](*params)
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
                cuda=False,
                list_mode=list_mode)
        t = time.time()
        print('total running time for one ds %s seconds' % str(t - t0))
        dset += 1

# 5 objects

if __name__ == '__main__':
    if args.run_idx is None:
        raise Exception('No run index was provided, please use the -r flag')
    if args.mode == 'all':
        for m_idx in range(len(gm.model_list)):
            run(m_idx, int(args.run_idx))
    else:
        try:
            m_idx = int(args.mode)
            run(m_idx, args.run_idx)
        except ValueError:
            print('Invalid mode for the script, must be \'all\' or integer')
            raise

    # N = 1
    # params = ([h] * n_layers, N, f_dict)
    # for m_idx in range(len(gm.model_list)):
    #     run(m_idx, 0, params)
    # N = 2
    # params = ([h] * n_layers, N, f_dict)
    # for m_idx in range(len(gm.model_list)):
    #     run(m_idx, 1, params)
    # for m_idx in range(len(gm.model_list)):
    #     run(m_idx, 2, params, list_mode='last')
