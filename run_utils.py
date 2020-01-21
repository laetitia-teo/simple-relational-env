import os
import re
import os.path as op
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

import gen
import baseline_models as bm
import graph_models_v2 as gm

from tqdm import tqdm
from torch.utils.data import DataLoader
try:
    from torch_geometric.data import Data
except ModuleNotFoundError:
    from utils import Data

# data utilities

from gen import SameConfigGen
from dataset import collate_fn, make_collate_fn
from baseline_utils import data_to_obj_seq_parts
from graph_utils import tensor_to_graphs, data_to_graph_parts
from graph_utils import data_to_graph_simple

# viz

from env import Env
from test_utils import ModelPlayground

# params

B_SIZE = 128

def nparams(model):
    return sum(p.numel() for p in model.parameters())

# metrics

criterion = torch.nn.CrossEntropyLoss()

def compute_accuracy(pred_clss, clss):
    pred_clss = (pred_clss[:, 1] >= pred_clss[:, 0]).long()
    accurate = np.logical_not(np.logical_xor(pred_clss, clss))
    return torch.sum(accurate).item()/len(accurate)

def compute_precision(pred_clss, clss):
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
    return 2 / ((1 / precision) + (1 / recall))

# data loading and transformation

def data_to_clss_parts(data):
    return data[2]

def load_dl(dpath):
    gen = SameConfigGen()
    gen.load(dpath)
    dl = DataLoader(
            gen.to_dataset(),
            shuffle=True,
            batch_size=B_SIZE,
            collate_fn=collate_fn)
    return dl

# def load_dl_overfit(dpath, n):
#     gen = SameConfigGen()
#     gen.load(dpath)
#     ds = gen.to_dataset()
#     gen.reset()
#     # select n positive examples
#     pos = []
#     i = 0
#     while len(pos) > n:
#         if gen.labels[i]:
#             pos.append()

# model saving and loading

def save_model(m, path):
    torch.save(m.state_dict(), path)

def load_model(m, path):
    m.load_state_dict(torch.load(path))
    return m

# data saving and viz

def save_results(data, path):
    if isinstance(data, torch.Tensor):
        data = data.detach().numpy()
    if isinstance(data, list):
        data = np.array(data)
    np.save(path, data)

def plot_metrics(losses, accs, i, path):
    """
    Utility for plotting training loss and accuracy in a run.
    Also saves the numpy arrays corresponding to the training metrics in the
    same folder. 
    """
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(losses)
    axs[0].set_title('loss')
    axs[0].set_xlabel('steps (batch size %s)' % B_SIZE)
    fig.suptitle('Training metrics for seed {}'.format(i))

    axs[1].plot(accs)
    axs[1].set_title('accuracy')
    axs[1].set_ylabel('steps (batch size %s)' % B_SIZE)

    filename = op.join(
        path,
        (str(i) + '.png'))
    plt.savefig(filename)
    plt.close()

# training loops

def one_step(model,
             optimizer,
             dl,
             criterion,
             metric=compute_accuracy,
             train=True,
             cuda=False,
             report_indices=False):
    accs = []
    losses = []
    indices = []
    n_passes = 0
    cum_loss = 0
    cum_acc = 0
    data_fn = data_to_graph_simple
    clss_fn = data_to_clss_parts
    if cuda:
        model.cuda()
    for data in dl:
        indices.append(list(data[3].numpy()))
        optimizer.zero_grad()
        # ground truth, model prediction
        clss = clss_fn(data)
        if cuda:
            clss = clss.cuda()
        if train:
            pred_clss = model(*data_fn(data))
        else:
            with torch.no_grad(): # saving memory
                pred_clss = model(*data_fn(data))
        if type(pred_clss) is list:
            # we sum the loss of all the outputs of the model
            loss = sum([criterion(pred, clss) for pred in pred_clss])
        else:
            loss = criterion(pred_clss, clss)
        if train:
            loss.backward()
            optimizer.step()
        l = loss.detach().cpu().item()
        if type(pred_clss) is list:
            # we evaluate accuracy on the last prediction
            a = metric(pred_clss[-1].detach().cpu(), clss.cpu())
        else:
            a = metric(pred_clss.detach().cpu(), clss.cpu())
        cum_loss += l
        cum_acc += a
        losses.append(l)
        accs.append(a)
        n_passes += 1
    if report_indices:
        return losses, accs, indices
    else: # backward compatibility
        return losses, accs

def one_run(dset, 
            seed,
            n,
            model,
            opt,
            dl,
            test_dl,
            prefix,
            criterion=criterion,
            cuda=False):
    """
    One complete training/testing run.
    """
    t0 = time.time()
    training_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []
    test_indices = []
    # train model
    for _ in range(n):
        l, a = one_step(
            model,
            opt,
            dl,
            criterion=criterion, 
            train=True, 
            cuda=cuda)
        training_losses += l
        training_accuracies += a
    save_results(
        training_losses,
        op.join(prefix, 'data', '{0}_{1}_train_loss.npy'.format(dset, seed)))
    save_results(
        training_accuracies,
        op.join(prefix, 'data', '{0}_{1}_train_acc.npy'.format(dset, seed)))
    # test
    l, a, i = one_step(
        model,
        opt,
        dl,
        criterion=criterion, 
        train=False,
        report_indices=True, 
        cuda=cuda)
    save_results(
        l, 
        op.join(prefix, 'data', '{0}_{1}_val_loss.npy'.format(dset, seed)))
    save_results(
        a,
        op.join(prefix, 'data', '{0}_{1}_val_acc.npy'.format(dset, seed)))
    save_results(
        i,
        op.join(prefix, 'data', '{0}_{1}_val_indices.npy'.format(dset, seed)))
    # save model
    save_model(
        model,
        op.join(prefix, 'models', 'ds{0}_seed{1}.pt'.format(dset, seed)))
    t = time.time()
    print('running time %s seconds' % str(t - t0))

# result navigation

def get_plot(model_idx, path):
    """
    Plots, one by one, the curves of the different models.
    """
    done = False
    directory = op.join(
        'experimental_results',
        'same_config_alt_norm',
        'run3',
        'model%s' % model_idx,
        'data')
    d_paths = os.listdir(directory)
    datalist = sorted(
        [p for p in d_paths if re.search(r'^((?!indices).)*$', p)])
    dit = iter(datalist)
    while True:
        try:
            filename = next(dit)
            path = op.join(directory, filename)
            data = np.load(path)
            plt.plot(data)
            plt.title(filename)
            plt.show()
        except StopIteration:
            break

# aggregate metrics

def model_metrics(dir_name):
    """
    Plots a histogram of accuracies, accuracies and stds for each dataset, for
    each model in the considered directory.
    """
    directory = op.join(
        'experimental_results',
        'same_config_alt_norm',
        dir_name)
    m_paths = sorted(os.listdir(directory))
    # fig, axs = plt.subplots(2, 4, constrained_layout=True)
    # fig = plt.figure()
    # outer = gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2)
    for i, mod_path in enumerate(m_paths):
        mod_idx = int(re.search(r'^model([0-9]+)$', mod_path)[1])
        print('mod idx %s' % mod_idx)
        path = op.join(directory, mod_path, 'data')
        d_paths = os.listdir(path)
        mdata = []
        for p in d_paths:
            s = re.search(r'^([0-9]+)_([0-9]+)_val_acc.npy$', p)
            if s:
                # file name, dataset number, seed number
                mdata.append((s[0], int(s[1]), int(s[2])))
        aa = [np.mean(np.load(op.join(path, m[0]))) for m in mdata]
        # inner = gridspec.GridSpecFromSubplotSpec(
        #     2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        # all accuracies
        # ax = plt.Subplot(fig, inner[0])
        # ax.hist(aa, bins=10)
        axs[0].hist(aa, bins=20)
        # accuracies for each dataset
        # ax = plt.Subplot(fig, inner[1])
        # get indices of the dataset
        d_indices = sorted([*{*[m[1] for m in mdata]}])
        means = np.array([np.mean(np.array(
            [np.mean(np.load(op.join(path, m[0]))) for m in mdata if m[1] == j])) \
                for j in d_indices])
        x = np.arange(len(d_indices)) * 2
        # ax.bar(means, x)
        axs[1].bar(means, x)
        # ax.set_xticklabels(d_indices)
        axs[1].set_xticklabels(d_indices)
        # ax.set_title(gm.model_names[mod_idx])
        axs[1].set_title(gm.model_names[mod_idx])
    plt.show()