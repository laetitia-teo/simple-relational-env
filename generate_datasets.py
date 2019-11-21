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
from gen import PartsGen
from dataset import ObjectDataset

from argparse import ArgumentParser

tasklist = ['simple', 'parts']

parser = ArgumentParser()
parser.add_argument('-d', '--directory',
                    dest='directory',
                    help='directory for the generated datasets',
                    default='data/parts_task')
parser.add_argument('-n', '--name',
                     dest='name',
                     help='name of the dataset')
parser.add_argument('-t', '--task',
                    dest='task',
                    help='task for which to generate a dataset, available' \
                    + 'tasks are "simple" and "parts". Default is "parts"',
                    default='parts')
parser.add_argument('-N', '--number',
                     dest='N',
                     help='number of objects in the dataset, ideally an' \
                     + 'even number',
                     default='300000')
# parser.add_argument('-D', '--distractors',
#                      dest='distractors',
#                      help='Number of distractor objects',
#                      default=None)

args = parser.parse_args()
if args.directory is None:
    print('Please provide a directory for the generated dataset using the -d '\
        + 'flag')
    quit()
if args.name is None:
    print('Please provide a name for the generated dataset using the -n flag')
    quit()

path = op.join(args.directory, args.name)

if args.task == 'simple':
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

if args.task == 'parts':

    gen = PartsGen()
    gen.generate(int(args.N))
    gen.save(path)

if args.task == 'curriculum':

    curr = [0, 1, 2, 3, 4, 5]
    for n_d in curr:
        print('Generating dataset with %s distractors' % n_d)
        gen = PartsGen(n_d=n_d)
        gen.generate(int(args.N))
        path = op.join(args.directory, (args.name + str(n_d) + '.txt'))
        gen.save(path)
