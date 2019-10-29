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

# seed

SEED = 42
torch.manual_seed(SEED)

# hparams

N_SH = 3
N_OBJ = 3
B_SIZE = 32
L_RATE = 1e-3
N_EPOCHS = 10
F_OBJ = 10
H = 16

# functions

def data_fn_naive(data):
    return (torch.reshape(data, [-1, 60]),)

def data_fn_scene(data):
    data = torch.reshape(data, (-1, 2, F_OBJ*N_OBJ))
    return data[:, 0, :], data[:, 1, :]

def data_fn_graphs(n):
    """
    Transforms object data in 2 graphs for graph models.

    n is number of objects. 
    """
    def data_fn_gr(data):
        return gm.tensor_to_graphs(data)
    return data_fn_gr

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

        if type(pred_clss) is list:
            # we sum the loss of all the outputs of the model
            loss = sum([criterion(pred, clss) for pred in pred_clss])

        else:
            loss = criterion(pred_clss, clss)

        loss.backward()

        if train:
            optimizer.step()

        l = loss.detach().item()
        if type(pred_clss) is list:
            # we evaluate accuracy on the last prediction
            a = compute_accuracy(pred_clss[-1].detach(), clss)
        else:
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

# objects

f_dict = {
    'f_x': F_OBJ,
    'f_e': F_OBJ,
    'f_u': F_OBJ,
    'f_out': 2}

I = gm.identity_mapping

data_fn = data_fn_graphs(N_OBJ)

nn_model = bm.SceneMLP(N_SH, F_OBJ, [H, H], H, [H, H])
nn_model = gm.ObjectMean([H, H], f_dict)
nn_model = gm.ObjectMeanDirectAttention([16, 16], f_dict)
nn_model = gm.GraphEmbedding([16], 16, 5, f_dict)
# nn_model = gm.GraphDifference([16, 16], 16, 5, f_dict, I)
# nn_model = gm.Alternating([16, 16], 16, 5, f_dict)

opt = torch.optim.Adam(nn_model.parameters(), lr=L_RATE)
criterion = torch.nn.CrossEntropyLoss()

# training

dl = load_dl('trainobject1')

# run(N_EPOCHS, nn_model, dl, data_fn, opt)

# testing

# test_dl = load_dl('testobject1')

# one_step(nn_model, test_dl, data_fn, opt, train=False)

# testing on rotations

# rot_dl = load_dl('testrotations')

# def data_fn(data):
#     x1 = data[:, :3, :]
#     x2 = data[:, 3:, :]
#     u1 = torch.mean(x1, 1)
#     u2 = torch.mean(x2, 1)
#     return (torch.cat([u1, u2], 1),)

# def identity(data):
#     return data

# def stupid_mean_predictor(data):
#     shape = (data.shape[0], 2)
#     res = torch.zeros(shape)
#     idx = torch.mean((data[:, :3] == data[:, 10:13]).float(), 1).long()
#     res[:, 1] = idx
#     return res

# acc = 0
# count = 0
# for data, clss in dl:
#     clss = clss.long()[:, 1]
#     res = stupid_mean_predictor(*data_fn(data))
#     print(compute_accuracy(res, clss))
#     acc += compute_accuracy(res, clss)
#     count += 1

# print(acc/count)