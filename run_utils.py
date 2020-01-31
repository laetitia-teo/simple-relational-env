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
except:
    from utils import Data

# data utilities

from gen import SameConfigGen, CompareConfigGen
from dataset import collate_fn, make_collate_fn
from baseline_utils import data_to_obj_seq_parts
from graph_utils import tensor_to_graphs
from graph_utils import data_to_graph_simple, data_to_graph_double
from graph_utils import state_list_to_graph
from graph_utils import merge_graphs

# viz

from env import Env
from test_utils import ModelPlayground

# params

B_SIZE = 128

def nparams(model):
    return sum(p.numel() for p in model.parameters())

# some data utils

def data_to_state_lists(data):
    """
    Used to transform data given by the parts DataLoader into a list of state
    lists usable by env.Env to generate the configuration image corresponding
    to this state of the environment.
    There is one state list per graph in the batch, and there are two scenes
    per batch (the target, smaller scene, and the reference scene).
    """
    targets, refs, labels, _, t_batch, r_batch = data
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

def load_dl(dpath, double=False):
    if not double:
        gen = SameConfigGen()
    if double:
        gen = CompareConfigGen()
    gen.load(dpath)
    dl = DataLoader(
            gen.to_dataset(),
            shuffle=True,
            batch_size=B_SIZE,
            collate_fn=collate_fn)
    return dl

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
             criterion=criterion,
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
    data_fn = data_to_graph_double
    # if isinstance(model, gm.GraphModelSimple):
    #     data_fn = data_to_graph_simple
    # elif isinstance(model, gm.GraphModelDouble):
    #     data_fn = data_to_graph_double
    clss_fn = data_to_clss_parts
    if cuda:
        model.cuda()
    for data in tqdm(dl):
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

    dset : dataset index
    seed :  random seed
    n : number of epochs
    prefix : path of the result directory.
    """
    t0 = time.time()
    training_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []
    test_indices = []
    # train model
    if isinstance(dl, DataLoader):
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
    elif isinstance(dl, list):
        for d in dl:
            for _ in range(n):
                l, a = one_step(
                    model,
                    opt,
                    d,
                    criterion=criterion, 
                    train=True, 
                    cuda=cuda)
                training_losses += l
                training_accuracies += a
    else:
        print('invalid dl type, must be DataLoader or list')
        return
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
        test_dl,
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

def get_expe_res(run_idx, model_idx):
    """
    Returns the list of arrays of training accuracies.
    """
    path = op.join(
        'experimental_results',
        'same_config_alt',
        'run%s' % run_idx,
        'model%s' % model_idx,
        'data')
    d_paths = os.listdir(path)
    plist = sorted(
        [p for p in d_paths if re.search(r'^.*train_acc.npy$', p)])
    trainlist = [np.load(op.join(path, p)) for p in plist]
    return np.array(trainlist)

def get_stat_func(line='mean', err='std'):    
    
    if line == 'mean':
        def line_f(a):
            return np.nanmean(a, axis=0)
    elif line == 'median':
        def line_f(a):
            return np.nanmedian(a, axis=0)
    else:
        raise NotImplementedError    

    if err == 'std':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0)
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0)
    elif err == 'sem':
        def err_plus(a):
            return line_f(a) + 1.676 * np.nanstd(a, axis=0) \
                / np.sqrt(a.shape[0])
        def err_minus(a):
            return line_f(a) - 1.676 * np.nanstd(a, axis=0) \
                / np.sqrt(a.shape[0])
    elif err == 'range':
        def err_plus(a):
            return np.nanmax(a, axis=0)
        def err_minus(a):
            return np.nanmin(a, axis=0)
    elif err == 'interquartile':
        def err_plus(a):
            return np.nanpercentile(a, q=75, axis=0)
        def err_minus(a):
            return np.nanpercentile(a, q=25, axis=0)
    else:
        raise NotImplementedError    

    return line_f, err_minus, err_plus

def plot_train_acc(run_idx, model_idx):
    """
    Plots the training accuracy of the model over all runs.
    """
    a = get_expe_res(run_idx, model_idx)
    line_f, err_minus, err_plus = get_stat_func(err='std')
    m = line_f(a)
    mi = err_minus(a)
    ma = err_plus(a)
    plt.plot(m)
    plt.fill_between(np.arange(len(m)), mi, ma, alpha=0.2)
    plt.show()
    # return m, mi, ma

def get_plot(model_idx, path):
    """
    Plots, one by one, the curves of the different models.
    """
    done = False
    directory = op.join(
        'experimental_results',
        'compare_config_alt_cur',
        path,
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

def batch_to_images(data, path, mod='one'):
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
        img = env.render(show=False)
        if mod == 'two':
            env.reset()
            env.from_state_list(rsl, norm=True)
            r_img = env.render(show=False)
            # separator is gray for false examples and white for true examples
            sep = np.ones((2, img.shape[1], 3)) * 127.5  + (l * 127.5)
            img = np.concatenate((img, sep, r_img))
        img_name = str(hash(img.tostring())) + '.jpg'
        cv2.imwrite(op.join(path, img_name), img)

# aggregate metrics

def model_metrics(run_idx):
    """
    Plots a histogram of accuracies, accuracies and stds for each dataset, for
    each model in the considered directory.
    """
    directory = op.join(
        'experimental_results',
        'compare_config_alt',
        'run%s' % run_idx)
    m_paths = sorted(os.listdir(directory))
    m_paths = [p for p in m_paths if re.search(r'^model([0-9]+)$', p)]
    fig, axs = plt.subplots(2, 4, constrained_layout=True)
    # fig = plt.figure()
    # outer = gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2)
    for i, mod_path in enumerate(m_paths[:8]):
        mod_idx = int(re.search(r'^model([0-9]+)$', mod_path)[1])
        # print('mod idx %s' % mod_idx)
        path = op.join(directory, mod_path, 'data')
        d_paths = os.listdir(path)
        mdata = []
        for p in d_paths:
            s = re.search(r'^([0-9]+)_([0-9]+)_val_acc.npy$', p)
            if s:
                # file name, dataset number, seed number
                mdata.append((s[0], int(s[1]), int(s[2])))
        aa = [np.mean(np.load(op.join(path, m[0]))) for m in mdata]
        mean_acc = str(np.around(np.mean(aa), 2))[:4]
        j = i % 2
        k = i // 2
        axs[j, k].hist(aa, bins=20)
        s = gm.model_names[mod_idx] + '; acc : {}'.format(mean_acc)
        axs[j, k].set_title(s)
    plt.show()

def hardness_dsets(run_idx):
    """
    Plots, for each model, the mean accuracy (over the random seeds) over each
    dataset.
    """
    directory = op.join(
        'experimental_results',
        'same_config_alt',
        'run%s' % run_idx)
    m_paths = sorted(os.listdir(directory))
    m_paths = [p for p in m_paths if re.search(r'^model([0-9]+)$', p)]
    fig, axs = plt.subplots(2, 4, constrained_layout=True)
    for i, mod_path in enumerate(m_paths[:-1]):
        mod_idx = int(re.search(r'^model([0-9]+)$', mod_path)[1])
        path = op.join(directory, mod_path, 'data')
        d_paths = os.listdir(path)
        mdata = []
        for p in d_paths:
            s = re.search(r'^([0-9]+)_([0-9]+)_val_acc.npy$', p)
            if s:
                # file name, dataset number, seed number
                mdata.append((s[0], int(s[1]), int(s[2])))
        means = []
        for h in range(10):
            l = [np.load(op.join(path, m[0])) for m in mdata if m[1] == h]
            means.append(np.mean(np.array(l)))
        j = i % 2
        k = i // 2
        axs[j, k].bar(np.arange(len(means)), means)
        mean_acc = np.mean(means)
        mean_acc = str(np.around(mean_acc, 2))
        s = gm.model_names[mod_idx] + '; acc : {}'.format(mean_acc)
        axs[j, k].set_xticklabels(np.arange(len(means)))
        axs[j, k].set_title(s)
    plt.show()

def get_heat_map_simple(model, gen):
    """
    Plots the heat maps of the difference between the positive and negative
    classes as a function of the position of one of the objects in the scene,
    and does it for each object.

    Assumes the model is a simple model (one input graph) and that the
    generator gen has the reference configuration loaded in its ref_state_list
    attribute.
    """
    n = gen.env.envsize * gen.env.gridsize
    matlist = []
    s = gen.ref_state_list[2:] # fix the reading
    pos_idx = [gen.env.N_SH+4, gen.env.N_SH+5]
    for state in s:
        mem = state[pos_idx]
        mat = np.zeros((0, n))
        for x in tqdm(range(n)):
            glist = []
            t = time.time()
            for y in range(n):
                state[pos_idx] = np.array([x / gen.env.gridsize,
                                           y / gen.env.gridsize])
                glist.append(state_list_to_graph(s))
            g = merge_graphs(glist)
            with torch.no_grad():
                pred = model(g)
            if isinstance(pred, list):
                pred = pred[-1]
            pred = pred.numpy()
            pred = pred[..., 1] - pred[..., 0]
            pred = np.expand_dims(pred, 0)
            mat = np.concatenate((mat, pred), 0)
        state[pos_idx] = mem
        matlist.append(mat) # maybe change data format here
    poslist = [state[pos_idx] * gen.env.gridsize for state in s]
    return matlist, poslist

def plot_heat_map_simple(model, gen):
    matlist, poslist = get_heat_map_simple(model, gen)
    # careful, this works only for the first five objects
    fig, axs = plt.subplots(1, 5, constrained_layout=True)
    for i, mat in enumerate(matlist):
        axs[i].matshow(mat)
        for j, pos in enumerate(poslist):
            if j == i:
                axs[i].scatter(pos[1], pos[0], color='r')
            else:
                axs[i].scatter(pos[1], pos[0], color='b')
        axs[i].set_title('object %s' % i)
    plt.show()
