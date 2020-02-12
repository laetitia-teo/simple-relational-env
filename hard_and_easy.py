"""
A small runfile for judging performance of the models in the case of hard
versus easy datasets.
"""
import time
import re
import os
import os.path as op
import pathlib
import numpy as np
import torch

import graph_models as gm

from argparse import ArgumentParser

from torch.utils.data import DataLoader

from gen import SameConfigGen
from dataset import collate_fn
from graph_utils import data_to_graph_simple

from run_utils import data_to_graph_simple, one_run, load_dl
from run_utils import save_model

# path of datasets

datasets_train = [
    'easy0_train',
    'easy1_train',
    'hard0_train',
    'hard1_train',
    '5_0_10000' # medium guy
]

datasets_test = [
    'easy0_test',
    'easy1_test',
    'hard0_test',
    'hard1_test',
    '5_0_5000_test' # medium guy
]

# hparams for the models

F_OBJ = 10
F_OUT = 2
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

params = ([h] * 1, N, f_dict)

params1 = ([h] * 1, N, f_dict)
params2 = ([h] * 2, N, f_dict)
params3 = ([h] * 4, N, f_dict)

param_dict = {
    0: params2,
    1: params1,
    2: params3,
}

m_idx = 1 # for now

dpath = 'data/same_config_alt'
save_path = 'experimental_results/hard_easy'
allnames = os.listdir(dpath)

train = [load_dl(op.join(dpath, p)) for p in datasets_train]
test = [load_dl(op.join(dpath, p)) for p in datasets_test]

def run(m_idx, run_idx, params=params, list_mode='all'):
    dset = 0
    print('model number %s' % m_idx)
    print('model name %s' % gm.model_list[m_idx].__name__)
    for dl_train, dl_test in zip(train, test):
        print('dset %s;' % dset)
        t0 = time.time()
        path = os.path.join(
            save_path, 'run%s' % run_idx, 'model' + str(m_idx))
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
                dl_test,
                path,
                cuda=False,
                list_mode=list_mode)
        t = time.time()
        print('total running time for one ds %s seconds' % str(t - t0))
        dset += 1

run(1, 0)