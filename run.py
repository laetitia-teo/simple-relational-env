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
from training_utils import load_dl
from training_utils import data_fn_graphs_three

# visualization/image generation

from env import Env

# data path

pretrain_path = op.join('data', 'simple_task', 'train.txt')
train_path = op.join('data', 'parts_task', 'train1.txt')
overfit_path = op.join('data', 'parts_task', 'overfit10000_32.txt')

# hparams

N_SH = 3
N_OBJ = 3
B_SIZE = 128 # batch size for graphs is small, because of edges
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
# pretrain_dl = load_dl('trainobject1')
# print('done')
# print('loading data ...')
# p = PartsGen()
# p.load(train_path)
# train_dl = DataLoader(p.to_dataset(),
#                       batch_size=B_SIZE,
#                       shuffle=True,
#                       collate_fn=collate_fn)
# print('done')
print('loading overfitting data ...')
p = PartsGen()
p.load(overfit_path)
overfit_dl = DataLoader(p.to_dataset(),
                        batch_size=B_SIZE,
                        shuffle=True,
                        collate_fn=collate_fn)
print('done')

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
