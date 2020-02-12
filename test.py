"""
Small test script for quick debugging.
"""
import os.path as op
import cv2
import torch

import graph_models as gm

from glob import glob

from torch.utils.data import DataLoader

from gen import SameConfigGen
from dataset import collate_fn
from graph_utils import data_to_graph_simple, data_to_graph_double

B_SIZE = 128
L_RATE = 1e-3
N_EPOCHS = 1
F_OBJ = 10
h = 16
n_layers = 2
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

gen = SameConfigGen()
gen.load('data/same_config_alt/5_5_10000')
gen.render('images/config_illustration/5')