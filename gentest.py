"""
Module for the generalization tests.
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

from run_utils import data_to_graph_simple, one_step, load_dl
from run_utils import save_model

n_epochs = 20
gridsize = 100

def load_dl_noshuffle(path):
    gen = SameConfigGen()
    gen.load(path)
    ds = gen.to_dataset()
    dl = DataLoader(
        ds,
        shuffle=False,
        batch_size=gridsize,
        collate_fn=collate_fn)
    # print('done')
    return dl

# test setting list

setting_list = [
    ('s_inter', 'sgrid'),
    ('s_extra1', 'sgrid'),
    ('s_extra2', 'sgrid'),
    ('r_inter', 'rgrid'),
    ('t_inter1', 'tgrid'),
    ('t_inter2', 'tgrid'),
    ('t_inter3', 'tgrid'),
    ('t_extra1', 'tgrid'),
    ('t_extra2', 'tgrid'),
    ('t_extra3', 'tgrid'),
    ('t_extra4', 'tgrid')
]

grid_dict = {
    'tgrid': [],
    'sgrid': [],
    'rgrid': []
}

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

params1 = ([h] * 1, N, f_dict)
params2 = ([h] * 2, N, f_dict)
params3 = ([h] * 4, N, f_dict)

param_dict = {
    0: params2,
    1: params1,
    2: params3,
}

m_idx = 1 # for now

# load all datasets

dpath = 'data/gen_same_config'
save_path = 'experimental_results/generalization/'
allnames = os.listdir(dpath)

data_dict = {}

# initialize dictionnary of test datasets

for e in grid_dict.keys():
    paths = sorted([p for p in allnames if re.search(r'^.+%s_100_1$' % e, p)])
    grid_dict[e] = [load_dl_noshuffle(os.path.join(dpath, p)) for p in paths]

for setting, typ in setting_list:
    print(setting)
    train = sorted(
        [p for p in allnames if re.search(r'^.+%s$' % setting, p)])
    train_dls = [load_dl(os.path.join(dpath, p)) for p in train]
    if typ == 'tgrid':
        data_dict[setting] = torch.zeros((0, 100, 100))
    else:
        data_dict[setting] = torch.zeros((0, 100))
    # create directories for save
    mpath = op.join(save_path, setting, 'models')
    apath = op.join(save_path, setting, 'data')
    pathlib.Path(mpath).mkdir(parents=True, exist_ok=True)
    pathlib.Path(apath).mkdir(parents=True, exist_ok=True)

    for dl_idx, dl in enumerate(train_dls):
        model = gm.model_list[m_idx](*param_dict[m_idx])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        # train model
        for j in range(n_epochs):
            print('epoch %s' % j)
            l, a = one_step(model, opt, dl)
        test_dl = grid_dict[typ][dl_idx]
        if typ == 'tgrid':
            # handle translation grid here
            pholder = torch.zeros((0, 100))
            for row in test_dl:
                with torch.no_grad():
                    pred = model(*data_to_graph_simple(row))
                if isinstance(pred, list):
                    pred = pred[-1]
                pred = (pred[:, 1] >= pred[:, 0]).float()
                pred = pred.unsqueeze(0)
                pholder = torch.cat([pholder, pred])
            data_dict[setting] = torch.cat([data_dict[setting]])
        else:
            # only one batch here
            data = next(iter(test_dl))
            with torch.no_grad():
                pred = model(*data_to_graph_simple(data))
            if isinstance(pred, list):
                pred = pred[-1]
            pred = (pred[:, 1] >= pred[:, 0]).float().unsqueeze(0)
            print(pred.shape)
            data_dict[setting] = torch.cat([data_dict[setting], pred])
        save_model(model, op.join(mpath, str(dl_idx) + '.pt'))
    np.save(op.join(apath, 'data.npy'), data_dict[setting].numpy())