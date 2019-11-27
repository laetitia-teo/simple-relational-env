"""
This is an executable script for running the Parts experiment (may be updated 
to include other experiments as well).

In the Parts challenge, a model has to learn to discriminate when a
configuration of objects is present in a bigger scene. 
"""
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import torch

import baseline_models as bm
import graph_models as gm

from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from gen import PartsGen

# data utilities

from dataset import collate_fn
from baseline_utils import data_to_obj_seq_parts
from graph_utils import data_to_graph_parts

# training 

from training_utils import one_step
from training_utils import data_to_clss_parts, data_to_clss_simple
from training_utils import batch_to_images
from training_utils import load_dl_legacy
from training_utils import data_fn_graphs_three
from training_utils import load_dl_parts
from training_utils import save_model, load_model

# visualization/image generation

from env import Env
from test_utils import ModelPlayground

# data path

pretrain_path = op.join('data', 'simple_task', 'train.txt')
train_path = op.join('data', 'parts_task', 'train1.txt')
overfit_path = op.join('data', 'parts_task', 'overfit10000_32.txt')

# hparams

N_SH = 3
N_OBJ = 3
B_SIZE = 128
L_RATE = 1e-3
N_EPOCHS = 1
F_OBJ = 10
H = 16

f_dict = {
        'f_x': F_OBJ,
        'f_e': F_OBJ,
        'f_u': F_OBJ,
        'f_out': 2}

# script arguments

parser = ArgumentParser()
parser.add_argument('-n', '--nepochs',
                    dest='n',
                    help='number of epochs',
                    default='1')
args = parser.parse_args()

# load data

# print('loading pretraining data ...')
# pretrain_dl = load_dl_legacy('trainobject1')
# print('done')
# train_dl = load_dl_parts('train1.txt', bsize=B_SIZE)
# print('loading overfitting data ...')
# overfit_dl = load_dl_parts('overfit10000_32.txt', bsize=B_SIZE)

# model

# model = gm.Simplified_GraphEmbedding([16, 16], 16, f_dict)
# model = gm.AlternatingSimple([16, 16], 2, f_dict)
# model = gm.GraphMatchingSimple([16, 16, 16], 10, 1, f_dict)
model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
criterion = torch.nn.CrossEntropyLoss()

def pre_train(n):
    losses, accs = [], []
    for i in range(n):
        print('Epoch %s' % i)
        l, a = one_step(model,
                        pretrain_dl,
                        data_fn_graphs_three,
                        data_to_clss_simple,
                        opt, 
                        criterion)
        losses += l
        accs += a
    plt.figure()
    plt.plot(losses)
    plt.figure()
    plt.plot(accs)
    plt.show()

def overfit(n):
    overfit_dl = load_dl_parts('overfit10000_32.txt', bsize=B_SIZE)
    losses, accs = [], []
    for i in range(n):
        print('Epoch %s' % i)
        l, a = one_step(model,
                        overfit_dl,
                        data_to_graph_parts,
                        data_to_clss_parts,
                        opt, 
                        criterion)
        losses += l
        accs += a
    plt.figure()
    plt.plot(losses)
    plt.figure()
    plt.plot(accs)
    plt.show()
    return losses, accs

def run(n=int(args.n)):
    losses, accs = [], []
    for i in range(n):
        print('Epoch %s' % i)
        l, a = one_step(model,
                        train_dl,
                        data_to_graph_parts,
                        data_to_clss_parts,
                        opt, 
                        criterion)
        losses += l
        accs += a
    plt.figure()
    plt.plot(losses)
    plt.figure()
    plt.plot(accs)
    plt.show()

def several_steps(n, dl, model, opt):
    losses, accs = [], []
    for _ in range(n):
        l, a = one_step(model,
                        dl,
                        data_to_graph_parts,
                        data_to_clss_parts,
                        opt, 
                        criterion)
        losses += l
        accs += a
    return losses, accs

def run_curriculum(retrain=True):
    """
    In this experiment, we train on different datasets, parametrized by the 
    number of distractor objects in the reference image.

    The number of distractors goes from 0 to 5, so there are 6 different
    datasets to train on.

    The argument retrain controls whether to start a different network for 
    each dataset (in which case we control the learning ability of our model as
    a function of the number of distractors), or to keep the same parameters
    and optimizer throughout the different datasets (in which case we monitor
    the effect of a curriculum of training data on the optimization process).

    This last setting should be contrasted with learning on a dataset of 0-5
    distractors with no curriculum for the same amounts of steps.

    Don't forget to test the overfitting capacity of the model beforehand.

    We save the resulting plots in the directory specified at save_path.
    """
    ds_names = ['curriculum0.txt',
                'curriculum1.txt',
                'curriculum2.txt',
                'curriculum3.txt',
                'curriculum4.txt',
                'curriculum5.txt']
    model = None
    for name in ds_names:
        print('Training %s' % name)
        if retrain or (model is None):
            model = gm.GraphMatchingSimple([16, 16, 16], 10, 1, f_dict)
            opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
        c_dl = load_dl_parts(name, bsize=B_SIZE)
        l, a = one_step(model,
                        c_dl,
                        data_to_graph_parts,
                        data_to_clss_parts,
                        opt, 
                        criterion)
        # plot and save training metrics
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(l)
        axs[0].set_title('loss')
        axs[0].set_xlabel('steps (batch size %s)' % B_SIZE)
        fig.suptitle('Training metrics for {}'.format(name[:-4]))

        axs[1].plot(a)
        axs[1].set_title('accuracy')
        axs[1].set_ylabel('steps (batch size %s)' % B_SIZE)

        filename = op.join(
            'experimental_results',
            'parts_curriculum',
            'retrain',
            name[:-4] + '.png')
        plt.savefig(filename)
        plt.clf()
        # save losses and accuracies as numpy arrays
        np.save(
            op.join('experimental_results',
                    'parts_curriculum',
                    'retrain',
                    name[:-4] + 'loss.npy'),
            np.array(l))
        np.save(
            op.join('experimental_results',
                    'parts_curriculum',
                    'retrain',
                    name[:-4] + 'acc.npy'),
            np.array(a))
        # checkpoint model
        save_model(
            model,
            op.join(
                'parts_curriculum',
                'retrain',
                (name[:-4] + '.pt')))

def curriculum_diffseeds(n, s, cur_n=0, training=None):
    """
    n : number of epochs;
    s : number of seeds;
    cur_n : number of distractors
    """
    dl_train = load_dl_parts('curriculum%s.txt' % cur_n)
    dl_test = load_dl_parts('curriculum%stest.txt' % cur_n)
    for i in range(s):
        if training is None:
            model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
            opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
        else:
            model, opt = training
        l, a = several_steps(n, dl_train, model, opt)
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(l)
        axs[0].set_title('loss')
        axs[0].set_xlabel('steps (batch size %s)' % B_SIZE)
        fig.suptitle('Training metrics for seed {}'.format(i))

        axs[1].plot(a)
        axs[1].set_title('accuracy')
        axs[1].set_ylabel('steps (batch size %s)' % B_SIZE)

        filename = op.join(
            'experimental_results',
            'curriculum%s' % cur_n,
            (str(i) + '.png'))
        plt.savefig(filename)
        plt.clf()
        # save losses and accuracies as numpy arrays
        np.save(
            op.join('experimental_results',
                    'curriculum%s' % cur_n,
                    (str(i) + 'loss.npy')),
            np.array(l))
        np.save(
            op.join('experimental_results',
                    'curriculum%s' % cur_n,
                    (str(i) + 'acc.npy')),
            np.array(a))
        # checkpoint model
        save_model(
            model,
            op.join(
                'curriculum%s' % cur_n,
                (str(i) + '.pt')))
        # test 
        l_test, a_test = one_step(model,
                                  dl_test,
                                  data_to_graph_parts,
                                  data_to_clss_parts,
                                  opt, 
                                  criterion,
                                  train=False)
        print('Test accuracy %s' % np.mean(a_test))
# run_curriculum()

def try_all_cur_n(s, n):
    """
    Performs computation with different initializations on all possible numbers
    of distractors.

    The goal of this test 
    """
    cur_list = [1, 2, 3, 4, 5]
    for cur_n in cur_list:
        curriculum_diffseeds(n, s, cur_n=cur_n)

def try_full_cur(s, n):
    # try all seeds
    dl_train_list = []
    dl_test_list = []
    for cur_n in range(6):
        dl_train_list.append(load_dl_parts('curriculum%s.txt' % cur_n))
        dl_test_list.append(load_dl_parts('curriculum%stest.txt' % cur_n))
    for i in range(s):
        model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
        opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
        losses = []
        accs = []
        for cur_n in range(6):
            dl_train = dl_train_list[cur_n]
            # dl_test = dl_test_list[cur_n]
            l, a = several_steps(n, dl_train, model, opt)
            losses += l
            accs += a
        # save data
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(losses)
        axs[0].set_title('loss')
        axs[0].set_xlabel('steps (batch size %s)' % B_SIZE)
        fig.suptitle('Training metrics for seed {}'.format(i))

        axs[1].plot(accs)
        axs[1].set_title('accuracy')
        axs[1].set_ylabel('steps (batch size %s)' % B_SIZE)

        filename = op.join(
            'experimental_results',
            'curriculum_full',
            (str(i) + '.png'))
        plt.savefig(filename)
        plt.clf()
        # save losses and accuracies as numpy arrays
        np.save(
            op.join('experimental_results',
                    'curriculum_full',
                    (str(i) + 'loss.npy')),
            np.array(losses))
        np.save(
            op.join('experimental_results',
                    'curriculum_full',
                    (str(i) + 'acc.npy')),
            np.array(accs))
        # checkpoint model
        save_model(
            model,
            op.join(
                'curriculum_full',
                (str(i) + '.pt')))

        dl_test = dl_test_list[-1]
        l_test, a_test = one_step(model,
                                  dl_test,
                                  data_to_graph_parts,
                                  data_to_clss_parts,
                                  opt, 
                                  criterion,
                                  train=False)
        print('Test accuracy %s' % np.mean(a_test))

def mix_all_cur(s, n):
    """
    This is the experiment that examines the relevance of training with a
    curriculum on the number of distractors. Instead of presenting our model
    with first only 0 distractors, then 1, then 2, we mix all distractor
    numbers by merging all those datasets, and train for the same number of
    iterations.
    """
    names = ['curriculum%s.txt' % i for i in range(6)]
    test_name = 'curriculum5test.txt'
    p = PartsGen()
    for name in names:
        p.load(op.join('data', 'parts_task', name), replace=False)
    dl_test = load_dl_parts(test_name)
    # load all the datasets
    dl = DataLoader(p.to_dataset(),
                    batch_size=B_SIZE,
                    shuffle=True,
                    collate_fn=collate_fn)
    for i in range(s):
        model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
        opt = torch.optim.Adam(model.parameters(), lr=L_RATE)
        losses, accs = several_steps(n, dl, model, opt)
        # plot
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(losses)
        axs[0].set_title('loss')
        axs[0].set_xlabel('steps (batch size %s)' % B_SIZE)
        fig.suptitle('Training metrics for seed {}'.format(i))

        axs[1].plot(accs)
        axs[1].set_title('accuracy')
        axs[1].set_ylabel('steps (batch size %s)' % B_SIZE)

        filename = op.join(
            'experimental_results',
            'full',
            (str(i) + '.png'))
        plt.savefig(filename)
        plt.clf()
        # save losses and accuracies as numpy arrays
        np.save(
            op.join('experimental_results',
                    'full',
                    (str(i) + 'loss.npy')),
            np.array(losses))
        np.save(
            op.join('experimental_results',
                    'full',
                    (str(i) + 'acc.npy')),
            np.array(accs))
        # checkpoint model
        save_model(
            model,
            op.join(
                'full',
                (str(i) + '.pt')))
        # test 
        l_test, a_test = one_step(model,
                                  dl_test,
                                  data_to_graph_parts,
                                  data_to_clss_parts,
                                  opt, 
                                  criterion,
                                  train=False)
        print('Test accuracy %s' % np.mean(a_test))

def load_model_playground():
    model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
    model.load_state_dict(torch.load('saves/models/curriculum5/19.pt'))
    pg = ModelPlayground(16, 20, model)
    maps = pg.model_heat_map(4, show=True)
    return maps