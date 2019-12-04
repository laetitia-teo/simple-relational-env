"""
Training module. This file defines the architecture of the training procedure,
given a model that is already defined.
"""
import os.path as op
import pickle
import numpy as np
import cv2
import torch

import baseline_models as bm
import graph_models as gm

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch_geometric.data import Data

from gen import PartsGen
from dataset import collate_fn
from graph_utils import tensor_to_graphs

from env import Env

# seed

# SEED = 42
# torch.manual_seed(SEED)

# hparams

N_SH = 3
N_OBJ = 3
B_SIZE = 32
L_RATE = 1e-3
N_EPOCHS = 10
F_OBJ = 10
H = 16

# other params

image_viz_path = op.join('data')

### General utils ###

def nparams(model):
    """
    Returns the number of trainable parameters in a pytorch model.
    """
    return sum(p.numel() for p in model.parameters())

### Data transformation functions ###

def data_to_clss_parts(data):
    """
    Get the ground truth from the Parts task dataloader.
    """
    return data[2]

def data_to_clss_simple(data):
    """
    Get the ground truth from the Simple task dataloader.
    """
    return data[1].long()[:, 1]

def data_fn_naive(data):
    return (torch.reshape(data, [-1, 60]),)

def data_fn_scene(data):
    data = torch.reshape(data, (-1, 2, F_OBJ*N_OBJ))
    return data[:, 0, :], data[:, 1, :]

def data_fn_naivelstm(data):
    return (data.permute(1, 0, 2),)

def data_fn_scenelstm(data):
    # TODO : optimize this
    data = torch.reshape(data, (-1, 2, N_OBJ, F_OBJ))
    d1, d2 = data[:, 0, ...], data[:, 1, ...]
    return d1.permute(1, 0, 2), d2.permute(1, 0, 2)

def data_fn_graphs(n):
    """
    Transforms object data in 2 graphs for graph models.

    n is number of objects. 
    """
    def data_fn_gr(data):
        return tensor_to_graphs(data[0])
    return data_fn_gr

def data_to_state_lists(data):
    """
    Used to transform data given by the parts DataLoader into a list of state
    lists usable by env.Env to generate the configuration image corresponding
    to this state of the environment.
    There is one state list per graph in the batch, and there are two scenes
    per batch (the target, smaller scene, and the reference scene).
    """
    targets, refs, labels, t_batch, r_batch = data
    f_x = targets.shape[-1]
    t_state_lists = []
    r_state_lists = []
    n = int(t_batch[-1] + 1)
    for i in range(n):
        t_idx = (t_batch == i).nonzero(as_tuple=True)[0]
        r_idx = (r_batch == i).nonzero(as_tuple=True)[0]
        target = list(targets[t_idx].numpy())
        ref = list(refs[r_idx].numpy())
        t_state_lists.append(target)
        r_state_lists.append(ref)
    return t_state_lists, r_state_lists, list(labels.numpy())

data_fn_graphs_three = data_fn_graphs(3)

### Data loading utilities ###

def load_dl_legacy(name):
    """
    Loads a DataLoader, for the old data format in SimpleTask.
    """
    path = op.join('data', 'simple_task', 'dataset_binaries', name)
    print('loading dataset...')
    with open(path, 'rb') as f:
        ds = pickle.load(f)
    print('done')
    dataloader = DataLoader(ds, batch_size=B_SIZE, shuffle=True)
    return dataloader

def load_dl_parts(name, bsize=128):
    """
    Loads a DataLoader in the Parts Task data format.
    """
    path = op.join('data', 'parts_task', 'old', name)
    print('loading data ...')
    p = PartsGen()
    p.load(path)
    dataloader = DataLoader(p.to_dataset(),
                            batch_size=bsize,
                            shuffle=True,
                            collate_fn=collate_fn)
    print('done')
    return dataloader

def load_dl(path, bsize=128):
    print('loading data...')
    p = PartsGen()
    p.load(path)
    dataloader = DataLoader(p.to_dataset(),
                            batch_size=bsize,
                            shuffle=True,
                            collate_fn=collate_fn)
    print('done')
    return dataloader

### Model evaluation utilities ###

def compute_accuracy(pred_clss, clss):
    """
    Computes accuracy on one batch prediction.
    Assumes pred_clss is detached from the computation graph.
    """
    pred_clss = (pred_clss[:, 1] >= pred_clss[:, 0]).long()
    accurate = np.logical_not(np.logical_xor(pred_clss, clss))
    return torch.sum(accurate).item()/len(accurate)

def compute_precision(pred_clss, clss):
    """
    Computes precision on one batch prediction.
    Assumes pred_clss is detached from the computation graph.
    """
    pred_clss = (pred_clss[:, 1] >= pred_clss[:, 0]).long()
    tp = torch.sum((pred_clss == 1) and (clss == 1))
    fp = torch.sum((pred_clss == 1) and (clss == 0))
    return tp / (tp + fp)

def compute_recall(pred_clss, clss):
    pred_clss = (pred_clss[:, 1] >= pred_clss[:, 0]).long()
    tp = torch.sum((pred_clss == 1) and (clss == 1))
    fn = torch.sum((pred_clss == 0) and (clss == 1))
    return tp / (tp + fn)

def compute_f1(precision, recall):
    """
    Computes the F1 score when given the precision and recall.
    """
    return 2 / ((1 / precision) + (1 / recall))

### Training functions ###

def one_step(model,
             dl,
             data_fn,
             clss_fn,
             optimizer,
             criterion,
             train=True,
             cuda=False):
    accs = []
    losses = []
    n_passes = 0
    cum_loss = 0
    cum_acc = 0
    for data in tqdm(dl):
        optimizer.zero_grad()
        # ground truth, model prediction
        clss = clss_fn(data)
        if cuda:
            clss = clss.cuda()
        pred_clss = model(*data_fn(data))

        if type(pred_clss) is list:
            # we sum the loss of all the outputs of the model
            loss = sum([criterion(pred, clss) for pred in pred_clss])

        else:
            loss = criterion(pred_clss, clss)

        loss.backward()

        if train:
            optimizer.step()

        l = loss.detach().cpu().item()
        if type(pred_clss) is list:
            # we evaluate accuracy on the last prediction
            a = compute_accuracy(pred_clss[-1].detach().cpu(), clss.cpu())
        else:
            a = compute_accuracy(pred_clss.detach().cpu(), clss.cpu())
        cum_loss += l
        cum_acc += a
        losses.append(l)
        accs.append(a)
        n_passes += 1
    return losses, accs

def run(n_epochs, model, dl, data_fn, optimizer, criterion):
    for epoch in range(n_epochs):
        mean_loss, mean_acc = one_step(model,
                                       dl,
                                       data_fn,
                                       optimizer,
                                       criterion)
        print('Epoch : {}, Mean loss : {}, Mean Accuracy {}'.format(
            epoch, mean_loss, mean_acc))

def several_inits(seeds, dl, data_fn):
    """
    Does several runs of one step on different models, initialized with the
    random seeds provided.
    """
    metrics = {}
    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        g_model = gm.GraphEmbedding([16], 16, 5, f_dict)
        opt = torch.optim.Adam(g_model.parameters(), lr=L_RATE)
        metrics[seed] = one_step(g_model, dl, data_fn, opt)
    return metrics

def save_model(m, name):
    """
    Saves the model m with name name in the model save folder.
    """
    prefix = op.join('saves', 'models')
    torch.save(m.state_dict(), op.join(prefix, name))

def load_model(m, name):
    """
    Loads the model parameters with name name into model m.
    """
    prefix = op.join('saves', 'models')
    m.load_state_dict(torch.load(op.join(prefix, name)))
    return m

### Visualization/Image generation utilities ###

def batch_to_images(data, path):
    """
    Transforms a batch of data into a set of images.
    """
    t_state_lists, r_state_lists, labels = data_to_state_lists(data)
    env = Env(16, 20)
    for tsl, rsl, l in zip(t_state_lists, r_state_lists, labels):
        # state to env
        # generate image
        # save it, name is hash code
        env.reset()
        env.from_state_list(tsl, norm=True)
        t_img = env.render(show=False)
        env.reset()
        env.from_state_list(rsl, norm=True)
        r_img = env.render(show=False)
        # separator is gray for false examples and white for true examples
        sep = np.ones((2, t_img.shape[1], 3)) * 127.5  + (l * 127.5)
        img = np.concatenate((t_img, sep, r_img)) # concatenate on what dim ?
        img_name = str(hash(img.tostring())) + '.jpg'
        cv2.imwrite(op.join(path, img_name), img)