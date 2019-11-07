"""
This file is a helper file to generate train and test datasets.
"""
import os.path as op
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from env import Env
from gen import SimpleTaskGen
from dataset import ObjectDataset

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-d', '--directory',
                    dest='directory',
                    help='directory for the generated datasets')
parser.add_argument('-n', '--name',
                     dest='name',
                     help='name of the dataset')

try:
    directory = parser.parse_args()['directory']
    name = parser.parse_args()['name']
except IndexError:
    print('please provide a directory for the generated dataset using' \
          + ' the -d flag and a name for the dataset using the -n flag')

path = op.join(directory, name)

n_obj = 3
n = 20 # number of variations/perturbations
No = 5 # number of distinct object configs, with the same objects
Ns = 10 # number of different object sets

env = Env(16, 20)
env.random_config(n_obj)

gen = SimpleTaskGen(env, n_obj)
rec = gen.generate_mix(Ns, No, n, rotations=False, record=False, shuffle=True)
print('saving objects')
gen.save(op.join(path, '.txt')) # we dont save the images
print('done')
ds = ObjectDataset(op.join(path, '.txt'), epsilon=1/(No*Ns))
ds.process()

# with open('data/simple_task/recording', 'wb') as f:
#     pickle.dump(rec, f)
print('dumping comparison dataset')
# see how to save this more modularly
with open('data/simple_task/dataset_binaries/shuffletest2', 'wb') as f:
    pickle.dump(ds, f)
print('done')
