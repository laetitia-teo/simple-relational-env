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

from run import several_steps
from gen import SameConfigGen
from dataset import collate_fn
from graph_utils import data_to_graph_simple

from run_utils import load_dl, one_run

# script arguments

parser = ArgumentParser()
parser.add_argument('-m', '--model',
                    dest='model_idx',
                    help='index of the model',
                    default='0')
parser.add_argument('-d', '--directory',
                    dest='directory',
                    help='path of the save and log directory',
                    default='experimental_results/same_config')

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

n_layers = 2
h = 16
lr = 10e-3
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

d_path = os.listdir('data/same_config')
train_5 = sorted([p for p in d_path if re.search(r'^5_.+_10{4}$', p)])[:10]
val_5 = sorted([p for p in d_path if re.search(r'^5_.+_val$', p)])[:10]
train_10 = sorted([p for p in d_path if re.search(r'^10_.+_10{4}$', p)])[:10]
val_10 = sorted([p for p in d_path if re.search(r'^10_.+_val$', p)])[:10]
train_20 = sorted([p for p in d_path if re.search(r'^20_.+_10{4}$', p)])[:10]
val_20 = sorted([p for p in d_path if re.search(r'^20_.+_val$', p)])[:10]

# 5 objects

params = ([h] * n_layers, N, f_dict)

dset = 0

for m_idx in range(len(gm.model_list)):
    print('model number %s' % m_idx)
    for dpath_train, dpath_val in zip(train_5, val_5):
        print('daset %s;' % dset)
        t0 = time.time()
        dl_train = load_dl(os.path.join('data/same_config', dpath_train))
        dl_val = load_dl(os.path.join('data/same_config', dpath_val))
        path = os.path.join(args.directory, 'run1', 'model' + m_idx)
        pathlib.Path(os.path.join(path, 'data')).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(path, 'models')).mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = gm.model_list[m_idx](*params)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            one_run(dset, seed, n_epochs, model, opt, dl_train, dl_val, path)
        t = time.time()
        print('total running time for one ds %s seconds' % str(t - t0))
        dset += 1