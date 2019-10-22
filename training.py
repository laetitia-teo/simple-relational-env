"""
Training module. This file defines the architecture of the training procedure,
given a model that is already defined.
"""
import os.path as op
import pickle
import numpy as np
import torch

import baseline_models as bm
import graph_models as gm

from torch.utils.data import DataLoader
from torch_geometric.data import Data

# hparams

N_SH = 3
N_OBJ = 3
B_SIZE = 32
L_RATE = 1e-3
N_EPOCHS = 500
F_OBJ = 10
H = 16

# objects

def data_fn_scene(data):
    data = torch.reshape(data, (-1, 2, F_OBJ*N_OBJ))
    return data[:, 0, :], data[:, 1, :]

def data_fn_graphs(data):
    """
    Transforms object data in 2 graphs for graph models. 
    """
    data = torch.reshape(data, (-1, 2, F_OBJ*N_OBJ))
    return gm.tensor_to_graphs(data)

scene_model = bm.SceneMLP(N_SH, F_OBJ, [H, H], H, [H, H])

opt = torch.optim.Adam(scene_model.parameters(), lr=L_RATE)
criterion = torch.nn.CrossEntropyLoss()

def load_dl(name):
    dpath = op.join('data', 'simple_task', 'dataset_binaries', name)
    print('loading dataset...')
    with open(dpath, 'rb') as f:
        ds = pickle.load(f)
    print('done')
    dataloader = DataLoader(ds, batch_size=B_SIZE, shuffle=True)
    return dataloader

def compute_accuracy(pred_clss, clss):
    """
    Computes accuracy on one prediction.
    Assumes pred_clss is detached from the computation graph.
    """
    pred_clss = (pred_clss[:, 1] >= pred_clss[:, 0]).long()
    accurate = np.logical_not(np.logical_xor(pred_clss, clss))
    return torch.sum(accurate).item()/len(accurate)

def one_step(model, dl, data_fn, optimizer, train=True):
    accs = []
    losses = []
    n_passes = 0
    cum_loss = 0
    cum_acc = 0
    for data, clss in dl:
        optimizer.zero_grad()
        # reshape data to fit in MLP
        clss = clss.long()[:, 1] # do this processing in dataset
        pred_clss = model(*data_fn(data))

        loss = criterion(pred_clss, clss)
        loss.backward()

        if train:
            optimizer.step()

        l = loss.detach().item()
        a = compute_accuracy(pred_clss.detach(), clss)
        cum_loss += l
        cum_acc += a
        losses.append(l)
        accs.append(a)
        n_passes += 1
    return cum_loss/n_passes, cum_acc/n_passes

def run(n_epochs, model, dl, data_fn, optimizer):
    for epoch in range(n_epochs):
        mean_loss, mean_acc = one_step(model, dl, data_fn, optimizer)
        print('Epoch : {}, Mean loss : {}, Mean Accuracy {}'.format(
            epoch, mean_loss, mean_acc))

    