"""
Training module. This file defines the architecture of the training procedure,
given a model that is already defined.
"""

import torch

from torch_geometric.data import Data

from baseline_models import NaiveMLP

# hparams

learning_rate = 1e-3
nb_epochs = 500

optimizer = torch.optim.Adam(learning_rate)

def one_step(model, dataset):
    for graph1, graph2 in dataset:
        # graphs in Data type
        