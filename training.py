"""
Training module. This file defines the architecture of the training procedure,
given a model that is already defined.
"""

import torch

from torch_geometric.data import Data

from baseline_models import NaiveMLP

# hparams

N_SH = 3
N_OBJ = 3
B_SIZE = 32
L_RATE = 1e-3
N_EPOCHS = 500
F_OBJ = 10
H = 16

optimizer = torch.optim.Adam(L_RATE)
scene_model = bm.SceneMLP(N_SH, F_OBJ, [H, H], H, [H, H])
data_fn = lambda data: torch.reshape(data, (B_SIZE, 2, F_OBJ*N_OBJ))

def one_step(model, dataloader, data_fn):
    for data, clss in dataloader:
        # reshape data to fit in MLP
        data = data_fn(data)
        
        