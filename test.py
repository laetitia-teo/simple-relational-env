"""
Small test script for quick debugging.
"""
import os.path as op
import torch

import graph_models_v2 as gm

from torch.utils.data import DataLoader

from gen import SameConfigGen
from dataset import collate_fn
from graph_utils import data_to_graph_simple

N_SH = 3
N_OBJ = 3
B_SIZE = 128
L_RATE = 1e-3
N_EPOCHS = 1
F_OBJ = 10
H = 16
N = 1
F_OUT = 2

f_dict = {
    'f_x': F_OBJ,
    'f_e': F_OBJ,
    'f_u': F_OBJ,
    'h': H,
    'f_out': F_OUT}

mlp_layers = [16, 16]

def data_to_clss_parts(data):
    return data[2]

gen = SameConfigGen()
gen.load('data/same_config/5_0_10000') # example dataset
dl = DataLoader(
    gen.to_dataset(),
    shuffle=True,
    batch_size=B_SIZE,
    collate_fn=collate_fn)

d_sample = next(iter(dl))
data_fn = data_to_graph_simple
clss_fn = data_to_clss_parts

# test all models, one prediction, see if no errors
for mc in gm.model_list:
    model = mc(mlp_layers, N, f_dict)
    opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
    print(model(*data_to_graph_simple(d_sample)))
    print('######################################################')