"""
This file is a helper file to generate train and test datasets.
"""
import os.path as op
import numpy as np

import gen

from argparse import ArgumentParser

tasklist = ['simple', 'parts']

parser = ArgumentParser()
parser.add_argument('-d', '--directory',
                    dest='directory',
                    help='directory for the generated datasets',
                    default='data/gen_same_config')
parser.add_argument('-Nc', '--number-configs',
                     dest='Nc',
                     help='number of different generated datasets',
                     default='20')
parser.add_argument('-Ns', '--number-samples',
                     dest='Ns',
                     help='number of samples in each dataset',
                     default='10000')
parser.add_argument('-No', '--number-objects',
                     dest='No',
                     help='number of objects in config',
                     default='5')
parser.add_argument('-Nt', '--number-test',
                     dest='Nt',
                     help='number of samples in test dataset',
                     default='5000')
parser.add_argument('-Nb', '--number-batch',
                     dest='Nb',
                     help='number of examples in one element of the test grid',
                     default='1')
parser.add_argument('-Ng', '--number-grid',
                     dest='Ng',
                     help='size of the test grid',
                     default='100')
parser.add_argument('-Ngt', '--number-grid-trans',
                     dest='Ngt',
                     help='size of the test grid for translations',
                     default='100')

args = parser.parse_args()

def gen_scale_train(g, ex_range, s, i):
    g.s_ex_range = ex_range
    g.reset()

    g.generate_alternative(int(args.Ns))
    a = (int(args.No), i, int(args.Ns), s)
    path = op.join(
        args.directory,
        '{0}_{1}_{2}_{3}'.format(*a))
    g.save(path)
    g.s_ex_range = None

def gen_scale_test(g, i):
    g.reset()
    g.generate_grid(int(args.Ng), int(args.Nb), mod='s')
    a = (int(args.No), i, int(args.Ns), int(args.Ng), int(args.Nb))
    path = op.join(
        args.directory,
        '{0}_{1}_{2}_sgrid_{3}_{4}'.format(*a))
    g.save(path)

def gen_rot_train(g, ex_range, s, i):
    g.r_ex_range = ex_range
    g.reset()
    g.generate_alternative(int(args.Ns))
    a = (int(args.No), i, int(args.Ns), s)
    path = op.join(
        args.directory,
        '{0}_{1}_{2}_{3}'.format(*a))
    g.save(path)
    g.r_ex_range = None

def gen_rot_test(g, i):
    g.reset()
    g.generate_grid(int(args.Ng), int(args.Nb), mod='r')
    a = (int(args.No), i, int(args.Ns), int(args.Ng), int(args.Nb))
    path = op.join(
        args.directory,
        '{0}_{1}_{2}_rgrid_{3}_{4}'.format(*a))
    g.save(path)

def gen_trans_train(g, ex_range, s, i):
    g.t_ex_range = ex_range
    g.reset()
    g.generate_alternative(int(args.Ns))
    a = (int(args.No), i, int(args.Ns), s)
    path = op.join(
        args.directory,
        '{0}_{1}_{2}_{3}'.format(*a))
    g.save(path)
    g.t_ex_range = None

def gen_trans_test(g, i):
    g.reset()
    g.generate_grid(int(args.Ngt), int(args.Nb), mod='t')
    a = (int(args.No), i, int(args.Ns), int(args.Ngt), int(args.Nb))
    path = op.join(
        args.directory,
        '{0}_{1}_{2}_tgrid_{3}_{4}'.format(*a))
    g.save(path)

for i in range(int(args.Nc)):
    g = gen.SameConfigGen(n=int(args.No))
    ref = g.ref_state_list # same ref state list for all the generated dsets
    s = 's_inter'
    print(s)
    ex_range = (0.8, 1.2)
    gen_scale_train(g, ex_range, s, i)
    s = 's_extra1'
    print(s)
    ex_range = (1.5, 2.0)
    gen_scale_train(g, ex_range, s, i)
    s = 's_extra2'
    print(s)
    ex_range = (1.0, 2.0)
    gen_scale_train(g, ex_range, s, i)
    # test grid (for all training datasets)
    gen_scale_test(g, i)

    s = 'r_inter'
    print(s)
    ex_range = (np.pi / 2, 3 * np.pi / 2)
    gen_rot_train(g, ex_range, s, i)
    # test grid
    gen_rot_test(g, i)

    # translation
    s = 't_inter1'
    print(s)
    ex_range = ((- 0.5, 0.5), (- 0.5, 0.5))
    gen_trans_train(g, ex_range, s, i)
    s = 't_inter2'
    print(s)
    ex_range = ((0.5, 1.0), (- 0.5, 0.5))
    gen_trans_train(g, ex_range, s, i)
    s = 't_inter3'
    print(s)
    ex_range = ((0.5, 1.0), (0.5, 1.0))
    gen_trans_train(g, ex_range, s, i)
    s = 't_extra1'
    print(s)
    ex_range = ((0.5, 1.0), (- 1.0, 1.0))
    gen_trans_train(g, ex_range, s, i)
    s = 't_extra2'
    print(s)
    ex_range = ((0.0, 1.0), (- 1.0, 1.0))
    gen_trans_train(g, ex_range, s, i)
    s = 't_extra3'
    print(s)
    ex_range = ((0.5, 1.0), (0.5, 1.0))
    gen_trans_train(g, ex_range, s, i)
    s = 't_extra4'
    print(s)
    ex_range = ((0.0, 1.0), (0.0, 1.0))
    gen_trans_train(g, ex_range, s, i)
    # test dataset
    gen_trans_test(g, i) # takes a long time
