import os
import re
import os.path as op
import time
import cv2

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

import gen
import graph_models as gm
import baseline_models as bm

from tqdm import tqdm
from pydoc import locate
from pprint import pprint

from torch.utils.data import DataLoader
try:
    from torch_geometric.data import Data
except:
    from utils import Data

# data utilities

from gen import SameConfigGen, CompareConfigGen
from dataset import collate_fn, make_collate_fn
from graph_utils import tensor_to_graphs
from graph_utils import data_to_graph_simple, data_to_graph_double
from graph_utils import state_list_to_graph
from graph_utils import merge_graphs
from graph_utils import graph_to_data, graphs_to_data
from baseline_utils import make_data_to_mlp_inputs, make_data_to_seq
from generate_config import load_config, type_to_string

# viz

from env import Env

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

def load_dl(dpath, double=False, cut=None, multiply=None):
    if not double:
        g = SameConfigGen()
    if double:
        g = CompareConfigGen()
    g.load(dpath)

    if cut is not None:
        g.cut(cut)
    if multiply is not None:
        g.multiply(multiply)
    
    dl = DataLoader(
            g.to_dataset(),
            shuffle=True,
            batch_size=B_SIZE,
            collate_fn=collate_fn)
    return dl

# for testing

# dl = load_dl('data/double/rotcur4_3_8_100000', double=True)
# data = next(iter(dl))
# x1, x2, labels, indices, b1, b2 = data

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

def test_and_save(dset, seed, prefix, model, opt, test_dl, cuda, n_obj, test_idx=0):
    l, a, i = one_step(
        model,
        opt,
        test_dl,
        criterion=criterion, 
        train=False,
        report_indices=True, 
        cuda=cuda,
        n_obj=n_obj)
    save_results(
        l, op.join(prefix, 'data', '{}_{}_{}_test_loss.npy'.format(
            test_idx, dset, seed)))
    save_results(
        a, op.join(prefix, 'data', '{}_{}_{}_test_acc.npy'.format(
            test_idx, dset, seed)))
    save_results(
        i, op.join(prefix, 'data', '{}_{}_{}_test_indices.npy'.format(
            test_idx, dset, seed)))

def one_step(model,
             optimizer,
             dl,
             criterion=criterion,
             metric=compute_accuracy,
             train=True,
             cuda=False,
             report_indices=False,
             list_mode='all',
             n_obj=5,
             **kwargs
             ):
    """
    list mode can be 'all' or 'last' depending on how we want to backprop.
    """
    accs = []
    losses = []
    indices = []
    n_passes = 0
    cum_loss = 0
    cum_acc = 0
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
            pred_clss = model(data)
        else:
            with torch.no_grad(): # saving memory
                pred_clss = model(data)
        if type(pred_clss) is list:
            if list_mode == 'all':
                # we sum the loss of all the outputs of the model
                loss = sum([criterion(pred, clss) for pred in pred_clss])
            elif list_mode == 'last':
                loss = criterion(pred_clss[-1], clss)
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
            n_obj=5,
            criterion=criterion,
            cuda=False,
            list_mode='all',
            preload=False):
    """
    One complete training/testing run.

    dset : dataset index
    seed :  random seed
    n : number of epochs
    prefix : path of the result directory.
    dl and test_dl can be lists of dls:
        if dl is a list the datasets will be trained on as a curriculum, with n
        epochs on each.
        if test_dl is a list, independent testing and saving will be performed
        on each dl.
    if load_model is True, tries to load the model corresponding to the dataset
    and the seed. If it fails, an exception is raised.
    """
    t0 = time.time()
    training_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []
    test_indices = []
    if preload:
        model = load_model(
            model,
            op.join(prefix, 'models', f'ds{dset}_seed{seed}.pt'))
    # train model
    if isinstance(dl, DataLoader):
        for _ in range(n):
            l, a = one_step(
                model,
                opt,
                dl,
                criterion=criterion, 
                train=True, 
                cuda=cuda,
                list_mode=list_mode,
                n_obj=n_obj)
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
                    cuda=cuda,
                    list_mode=list_mode,
                    n_obj=n_obj)
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
    if isinstance(test_dl, DataLoader):
        test_and_save(
            dset,
            seed,
            prefix,
            model,
            opt,
            test_dl,
            n_obj=n_obj,
            cuda=cuda)
    elif isinstance(test_dl, list):
        for test_idx, test_d in enumerate(test_dl):
            test_and_save(
                dset,
                seed,
                prefix,
                model,
                opt,
                test_d,
                cuda=cuda,
                n_obj=n_obj,
                test_idx=test_idx)
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

def get_plot(model_name, path):
    """
    Plots, one by one, the curves of the different models.
    """
    done = False
    
    directory = op.join(
        'experimental_results',
        path,
        model_name,
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

def config_summary(prefix=''):
    """
    Prints to terminal a summary of all the experiments configurations
    available at the specified prefix.
    """
    listd = os.listdir(op.join('configs', prefix))

    rese = lambda p: re.search(r'^config[0-9]+$', p)
    configs = sorted([rese(p)[0] for p in listd if rese(p)])

    for cp in configs:
        c = load_config(op.join('configs', prefix, cp))
        print(f'\n\n################### {cp} #################\n\n')
        pprint(c)

        # number of parameters of models
        for i, model_name in enumerate(c['models']):
            model_class = locate('graph_models.' + model_name)
            if model_class is None:
                model_class = locate('baseline_models.' + model_name)
            model = model_class(*c['hparam_list'][i])

            print(f'n_params for {model_name}: {nparams(model)}')

def model_metrics_old(run_idx, double=True):
    """
    Plots a histogram of accuracies, accuracies and stds for each dataset, for
    each model in the considered directory.
    """
    if double:
        mlist = gm.model_list_double
    else:
        mlist = gm.model_list
    directory = op.join(
        'experimental_results',
        'same_config_alt',
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
        std = str(np.around(np.std(aa), 3))[:5]
        mean_acc = str(np.around(np.mean(aa), 2))[:4]
        j = i % 2
        k = i // 2
        axs[j, k].hist(aa, bins=20)
        s = mlist[mod_idx].__name__ + '; {0} +- {1}'.format(mean_acc, std)
        axs[j, k].set_title(s)
    plt.show()

def model_metrics(expe_idx, dir_prefix='', n_test=0, var='std'):
    """
    Print the mean accuracy of the model on the test data specified by the 
    expe_idx. dir_prefix allows us to use data/models in subdirectories of the
    standard data save and config directories.
    The variability reported can be 'std' for standard deviation or 'conf' for 
    95 percent confidence intervals.
    """
    config = load_config(op.join('configs', dir_prefix, 'config%s' % expe_idx))
    path = op.join(config['save_dir'], dir_prefix, 'expe%s' % expe_idx)
    res_dict = {}
    for m_str in config['models']:
        mpath = op.join(path, m_str, 'data')
        d_paths = os.listdir(mpath)
        mdata = []
        for p in d_paths:
            s = re.search(
                r'^{}_([0-9]+)_([0-9]+)_test_acc.npy$'.format(n_test), p)
            if s:
                # file name, dataset number, seed number
                mdata.append((s[0], int(s[1]), int(s[2])))
        aa = [np.mean(np.load(op.join(mpath, m[0]))) for m in mdata]
        print(len(aa))
        print()
        print(aa)
        # aaa = np.array()
        if var == 'std':
            v = np.std(aa)
        #     mean_acc = str(np.around(np.mean(aa), 2))[:4]
        elif var == 'conf':
            # mean_acc = np.mean(accs)
            print('conf')
            mean = np.mean(aa)
            se = st.sem(aa)
            n = len(aa)
            v = se * st.t.ppf((1 + 0.95) / 2., n-1)
            # inter = st.t.interval(
            #     0.95,
            #     len(aa) - 1,
            #     loc=np.mean(aa),
            #     scale=st.sem(aa))
            # v = np.mean(aa) - inter[0]
        else:
            raise ValueError(f'Invalid value for "var": {var}, must be one' \
                + 'of "std" or "conf"')
        # res_dict[m_str] = (mean_acc, v)
        # v_round = str(np.around(v, 3))[:5]
        # mean_acc_round = str(np.around(mean_acc, 2))[:4]
        v_round = str(np.around(v, 3))[:5]
        mean_acc_round = str(np.around(np.mean(aa), 2))[:4]
        plt.hist(aa, bins=20)
        title = m_str + '; {0} +- {1}'.format(mean_acc_round, v_round)
        plt.title(title)
        plt.show()
    return aa

def test_on(expe_idx, tpath, midx=0, dir_prefix='', var='std'):
    """
    Loads trained models as specified by the expe_idx, and tests them on all
    test data at path specified by tpath. The model tested corresponds to the
    one at index midx of the model list specified by the configuration file
    corresponding to expe_idx.
    The variability metric reported can be the standard deviation (var='std')
    or the 95 percent confidence interval (var='conf').
    """
    test_paths = os.listdir(tpath)
    config = load_config(op.join('configs', dir_prefix, 'config%s' % expe_idx))
    path = op.join(config['save_dir'], dir_prefix, 'expe%s' % expe_idx)
    hparams = config['hparam_list'][midx]
    res_dict = {}
    for test_path in test_paths:
        print('loading dl')
        test_dl = load_dl(op.join(tpath, test_path), double=True)
        print('done')
        model_name = config['models'][midx]
        mpaths = op.join(path, model_name, 'models')
        try:
            model = eval('gm.' + model_name + '(*hparams)')
        except:
            model = eval('bm.' + model_name + '(*hparams)')
        mpathlist = os.listdir(mpaths)
        accs = []
        mpathlist = [p for p in mpathlist \
                     if re.search(r'^ds0_seed[0-9]+.pt', p)]
        for mpath in mpathlist:
            mpath = op.join(mpaths, mpath)
            model = load_model(model, mpath)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            l, a, i = one_step(
                model,
                opt,
                test_dl,
                criterion=criterion,
                train=False,
                report_indices=True, 
            )
            accs += list(a)
        if var == 'std':
            v = np.std(accs)
        elif var == 'conf':
            v = st.t.interval(
                0.95,
                len(accs) - 1,
                loc=np.mean(accs),
                scale=st.sem(accs))
        else:
            raise ValueError(f'Invalid value for "var": {var}, must be one' \
                + 'of "std" or "conf"')
        a = np.mean(accs)
        print(f'dset : {test_path}, mean acc {a} +- {v}')
        res_dict[test_path] = (a, v)
    return res_dict

def hardness_dsets_old(run_idx):
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

def hardness_dsets(expe_idx, n_test=0):
    """
    Plots, for each model, the mean accuracy (over the random seeds) over each
    dataset.
    """
    config = load_config(op.join('configs', 'config%s' % expe_idx))
    path = op.join(config['save_dir'], 'expe%s' % expe_idx)
    for m_str in config['models']:
        mpath = op.join(path, m_str, 'data')
        d_paths = os.listdir(mpath)
        mdata = []
        for p in d_paths:
            s = re.search(
                r'^{}_([0-9]+)_([0-9]+)_test_acc.npy$'.format(n_test), p)
            if s:
                # file name, dataset number, seed number
                mdata.append((s[0], int(s[1]), int(s[2])))
        means = []
        Ndsets = max([m[1] for m in mdata])
        for h in range(Ndsets + 1):
            l = [np.load(op.join(mpath, m[0])) for m in mdata if m[1] == h]
            means.append(np.mean(np.array(l)))
        plt.bar(np.arange(len(means)), means)
        print(means)
        mean_acc = np.mean(means)
        mean_acc = str(np.around(mean_acc, 2))
        # title = m_str + '; acc : {}'.format(mean_acc)
        # plt.xticklabels(np.arange(len(means)))
        # plt.title(title)
        plt.show()

def models_from_config(config_idx, prefix=''):
    config = load_config(op.join('configs', prefix, f'config{config_idx}'))
    model_list = []
    for m_str, params in zip(config['models'], config['hparam_list']):
        m = locate('graph_models.' + m_str)(*params)
        model_list.append(m)
    return model_list

def load_models_from_config(dset, seed, config_idx, prefix=''):
    # index of experiment is same as index of config
    config = load_config(op.join('configs', prefix, 'config%s' % config_idx))
    path = op.join(
        config['save_dir'],
        prefix,
        'expe%s' % config_idx)
    model_list = models_from_config(config_idx, prefix)
    loaded_models = []
    for m_str, m in zip(config['models'], model_list):
        mpath = op.join(
            path,
            m_str,
            'models',
            'ds{}_seed{}.pt'.format(dset, seed))
        m = load_model(m, mpath)
        loaded_models.append(m)
    return loaded_models

#################### heatmaps for simple setting ##############################

def get_heat_map_simple(model, g):
    """
    Plots the heat maps of the difference between the positive and negative
    classes as a function of the position of one of the objects in the scene,
    and does it for each object.

    Assumes the model is a simple model (one input graph) and that the
    generator gen has the reference configuration loaded in its ref_state_list
    attribute.
    """
    a = 2
    n = g.env.envsize * a
    matlist = []
    s = g.ref_state_list[2:] # fix the reading
    pos_idx = [g.env.N_SH+4, g.env.N_SH+5]
    for state in s:
        mem = state[pos_idx]
        mat = np.zeros((0, n))
        for x in tqdm(range(n)):
            glist = []
            t = time.time()
            for y in range(n):
                state[pos_idx] = np.array([x / a,
                                           y / a])
                glist.append(state_list_to_graph(s))
            graph = merge_graphs(glist)
            with torch.no_grad():
                pred = model(graph_to_data(graph))
            if isinstance(pred, list):
                pred = pred[-1]
            pred = pred.numpy()
            pred = pred[..., 1] - pred[..., 0]
            pred = np.expand_dims(pred, 0)
            mat = np.concatenate((mat, pred), 0)
        state[pos_idx] = mem
        matlist.append(mat) # maybe change data format here
    poslist = [state[pos_idx] * a for state in s]
    return matlist, poslist

def plot_heat_map_simple(model, g, save=False, show=True, **kwargs):
    matlist, poslist = get_heat_map_simple(model, g)
    maxval = max([np.max(mat) for mat in matlist])
    minval = min([np.min(mat) for mat in matlist])
    maxval = min(5, maxval)
    minval = max(-5, minval)
    # careful, this works only for the first five objects
    fig, axs = plt.subplots(1, 5, constrained_layout=True, figsize=(14, 3))
    for i, mat in enumerate(matlist):
        axs[i].set_axis_off()
        im = axs[i].matshow(mat, vmin=minval, vmax=maxval, cmap='inferno')
        for j, pos in enumerate(poslist):
            if j == i:
                axs[i].scatter(pos[1], pos[0], color='cyan')
            else:
                axs[i].scatter(pos[1], pos[0], color='b')
            # fig.colorbar(im)
        # axs[i].set_title('object %s' % i)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    if show:
        plt.show()
    if save:
        directory = op.join('images', 'heatmaps', 'simple')
        seed = kwargs['seed']
        dset = kwargs['dset']
        expe = kwargs['expe']
        modl = type_to_string(type(model))
        name = 'expe{}_dset{}_seed{}_{}.png'.format(expe, dset, seed, modl)
        plt.savefig(op.join(directory, name))
        plt.close()

def plot_heat_map_simple_several_models(model_list,
                                        g,
                                        save=False,
                                        show=True,
                                        **kwargs):
    for m in model_list:
        print(type_to_string(type(m)))
        plot_heat_map_simple(m, g, save=save, show=show, **kwargs)

def plot_heatmap_simple_all(config_idx):
    g = gen.SameConfigGen()
    config = load_config(op.join('configs', 'config%s' % config_idx))
    for dset in range(5):
        g.load(op.join(config['load_dir'], config['test_datasets'][dset]))
        directory = op.join('images', 'heatmaps', 'simple')
        path = op.join(directory, 'dset%s.png' % dset)
        img = g.render_ref_state(show=False)
        img = np.flip(img, 0)
        cv2.imwrite(path, img)
        for seed in range(5):
            seed = seed
            model_list = load_models_from_config(dset, seed, config_idx)
            plot_heat_map_simple_several_models(
                model_list,
                g,
                save=True,
                show=False,
                seed=seed,
                expe=config_idx,
                dset=dset)
        g.reset()

#################### heatmaps for double setting ##############################

def get_heat_map_double(model, n_obj, s=None):
    a = 2
    env = Env()
    n = env.envsize * a
    matlist = []
    if s is None:
        env = Env()
        env.random_config(n_obj)
        s = env.to_state_list(norm=True)
    graph1 = merge_graphs([state_list_to_graph(s)] * n)
    pos_idx = [env.N_SH+4, env.N_SH+5]
    for state in s:
        mem = state[pos_idx]
        mat = np.zeros((0, n))
        for x in tqdm(range(n)):
            glist = []
            t = time.time()
            for y in range(n):
                state[pos_idx] = np.array([x / a,
                                           y / a])
                glist.append(state_list_to_graph(s))
            graph2 = merge_graphs(glist)
            with torch.no_grad():
                pred = model(graphs_to_data(graph1, graph2))
            if isinstance(pred, list):
                pred = pred[-1]
            pred = pred.numpy()
            pred = pred[..., 1] - pred[..., 0]
            pred = np.expand_dims(pred, 0)
            mat = np.concatenate((mat, pred), 0)
        state[pos_idx] = mem
        matlist.append(mat) # maybe change data format here
    poslist = [state[pos_idx] * a for state in s]
    return matlist, poslist

def plot_heat_map_double(model, n_obj, s, save=False, show=True, **kwargs):
    matlist, poslist = get_heat_map_double(model, n_obj, s)
    maxval = max([np.max(mat) for mat in matlist])
    minval = min([np.min(mat) for mat in matlist])
    maxval = min(5, maxval)
    minval = max(-5, minval)
    # careful, this works only for the first five objects
    fig, axs = plt.subplots(1, 5, constrained_layout=True, figsize=(14, 3))
    for i, mat in enumerate(matlist):
        axs[i].set_axis_off()
        im = axs[i].matshow(mat, vmin=minval, vmax=maxval, cmap='inferno')
        for j, pos in enumerate(poslist):
            if j == i:
                axs[i].scatter(pos[1], pos[0], color='cyan')
            else:
                axs[i].scatter(pos[1], pos[0], color='b')
            # fig.colorbar(im)
        # axs[i].set_title('object %s' % i)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    if show:
        plt.show()
    if save:
        directory = op.join('images', 'heatmaps', 'double')
        seed = kwargs['seed']
        draw = kwargs['draw']
        expe = kwargs['expe']
        modl = type_to_string(type(model))
        name = 'expe{}_draw{}_seed{}_{}.png'.format(expe, draw, seed, modl)
        plt.savefig(op.join(directory, name))
        plt.close()

def plot_heat_map_double_several_models(model_list,
                                        n_obj,
                                        s,
                                        save=False,
                                        show=True,
                                        **kwargs):
    for m in model_list:
        print(type_to_string(type(m)))
        plot_heat_map_double(m, n_obj, s, save=save, show=show, **kwargs)

def plot_heatmap_double_all(config_idx, n_obj=5, n_draws=5):
    config = load_config(op.join('configs', 'config%s' % config_idx))
    env = Env()
    for draw in range(n_draws):
        env.random_config(n_obj)
        s = env.to_state_list(norm=True)
        directory = op.join('images', 'heatmaps', 'double')
        path = op.join(directory, 'draw%s.png' % draw)
        img = env.render(show=False, mode='envsize')
        img = np.flip(img, 0)
        cv2.imwrite(path, img)
        for seed in range(5):
            seed = seed
            model_list = load_models_from_config(0, seed, config_idx)
            plot_heat_map_double_several_models(
                model_list,
                n_obj=n_obj,
                s=s,
                save=True,
                show=False,
                seed=seed,
                expe=config_idx,
                draw=draw)
        env.reset()