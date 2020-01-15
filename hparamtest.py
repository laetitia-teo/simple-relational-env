"""
This module performs hyperparameter testing, and validation of our models.
"""
import os.path as op
import torch

import graph_model_v2 as gm

from run import several_steps

from glob import glob

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

hparams_def = {
    'n_layers': 2,
    'h':  16,
    'lr': 10e-3,
    'N': [1]
}

