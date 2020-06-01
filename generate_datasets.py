"""
This file is a helper file to generate train and test datasets.
"""
import os.path as op
import numpy as np

import gen

from argparse import ArgumentParser
from pathlib import Path

tasklist = ['simple', 'parts']

parser = ArgumentParser()
parser.add_argument('-d', '--directory',
                    dest='directory',
                    help='directory for the generated datasets',
                    default='data/double_perturb')
parser.add_argument('-Nc', '--number-configs',
                     dest='Nc',
                     help='number of different generated datasets',
                     default='1')
parser.add_argument('-Ns', '--number-samples',
                     dest='Ns',
                     help='number of samples in each dataset',
                     default='100000')
parser.add_argument('-No', '--number-objects',
                     dest='No',
                     help='number of objects in config',
                     default='5')
parser.add_argument('-Nm', '--number-objects-max',
                     dest='No_max',
                     help='max number of objects in config',
                     default=None)
parser.add_argument('-Nt', '--number-test',
                     dest='Nt',
                     help='number of samples in test dataset',
                     default='10000')
parser.add_argument('-m', '--mode',
                     dest='mode',
                     help='simple or double dataset generation',
                     default='rotcur_perturb')

args = parser.parse_args()
pi = np.pi

if args.mode == 'double':
    for i in range(6, 21):
        print(i)
        g = gen.CompareConfigGen(n=i)
        g.generate_alternative(int(args.Ns))
        path = op.join(
            save_dir,
            '%s_%s_%s' % (int(args.No), 0, int(args.Ns)))
        g.save(path)
        g.reset()
    # # validation
    # g.generate_alternative(int(args.Nt))
    # path = op.join(
    #     save_dir,
    #     '%s_%s_%s_val' % (int(args.No), 0, int(args.Nt)))
    # g.save(path)
    # # generate test dataset
    # g.reset()
    # g.generate_alternative(int(args.Nt))
    # path = op.join(
    #     save_dir,
    #     '%s_%s_%s_test' % (int(args.No), 0, int(args.Nt)))
    # g.save(path)

if args.mode == 'simple':
    save_dir = 'data/simple'
    No_range = range(3, 31) # dataset n_objs
    # No = int(args.No) # number of objects in a dataset
    Ns = 10000 # number of train examples
    Nt = 5000 # number of validation/test examples
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for No in No_range:

        g = gen.SameConfigGen(n=No)
        ref = g.ref_state_list
        g.generate_alternative(Ns)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}')
        g.save(path)

        # generate validation dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate_alternative(Nt)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}_val')
        g.save(path)
        
        # generate test dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate_alternative(Nt)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}_test')
        g.save(path)

if args.mode == 'simple_perturb':
    save_dir = 'data/simple_perturb'
    Nc = 10 # number of different datasets
    No = int(args.No) # number of objects in a dataset
    Ns = 10000 # number of train examples
    Nt = 5000 # number of validation/test examples
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i in range(Nc):
        g = gen.SameConfigGen(n=No)
        ref = g.ref_state_list
        g.generate(Ns)
        path = op.join(
            save_dir,
            '%s_%s_%s' % (No, i, Ns))
        g.save(path)
        # generate validation dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate(Nt)
        path = op.join(
            save_dir,
            '%s_%s_%s_val' % (No, i, Nt))
        g.save(path)
        # generate test dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate(Nt)
        path = op.join(
            save_dir,
            '%s_%s_%s_test' % (No, i, Nt))
        g.save(path)

if args.mode == 'rotcur':
    # double setting, rotation curriculum
    save_dir = 'data/double'
    No_min = int(args.No) # minimum number of objects in a dataset

    if args.No_max is None:
        No_max = int(args.No)
    else:
        No_max = int(args.No_max) # maximum number of objects in a dataset

    Ns = 100000 # number of train examples
    Nt = 10000 # number of validation/test examples
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cur_list = [
        (pi/10, 2*pi),
        (pi/2 + pi/10, 2*pi),
        (pi + pi/10, 2*pi),
        (3*pi/2 + pi/10, 2*pi),
        None]

    for i, cur in enumerate(cur_list):
        g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
        g.r_ex_range = cur
        g.generate_alternative(Ns)
        path = op.join(
            save_dir,
            f'rotcur{i}_{No_min}_{No_max}_{Ns}')
        g.save(path)

    # valid
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate_alternative(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_val')
    g.save(path)

    # test
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate_alternative(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_test')
    g.save(path)

if args.mode == 'rotcur_perturb':
    # double setting, rotation curriculum, perturbed version
    save_dir = 'data/double_perturb'
    Nc = 10 # number of different datasets
    No_min = int(args.No) # minimum number of objects in a dataset
    No_max = int(args.Nm) # maximum number of objects in a dataset
    Ns = 100000 # number of train examples
    Nt = 10000 # number of validation/test examples
    cur_list = [
        (pi/10, 2*pi),
        (pi/2 + pi/10, 2*pi),
        (pi + pi/10, 2*pi),
        (3*pi/2 + pi/10, 2*pi),
        None]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, cur in enumerate(cur_list):
        g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
        g.r_ex_range = cur
        g.generate(Ns)
        path = op.join(
            save_dir,
            f'rotcur{i}_{No_min}_{No_max}_{Ns}')
        g.save(path)
    # validation
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_val')
    g.save(path)
    # generate test dataset
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_test')
    g.save(path)

if args.mode == 'test_double':
    # generate test datasets for the double setting with different number of
    # objects
    Ns = 10000
    save_dir = 'data/test_double'
    n_obj_list = range(4, 8)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for n in n_obj_list:
        g = gen.CompareConfigGen(n_min=n, n_max=n)
        g.generate_alternative(Ns)
        path = op.join(
            save_dir,
            'testdouble_%s_%s_%s' % (n, 0, Ns))
        g.save(path)

if args.mode == 'simple_abstract':
    save_dir = 'data/simple_abstract'
    No_range = range(3, 31) # dataset n_objs
    # No = int(args.No) # number of objects in a dataset
    Ns = 10000 # number of train examples
    Nt = 5000 # number of validation/test examples
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for No in No_range:

        g = gen.SameConfigGen(n=No)
        ref = g.ref_state_list
        g.generate_abstract(Ns)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}')
        g.save(path)

        # generate validation dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate_abstract(Nt)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}_val')
        g.save(path)
        
        # generate test dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate_abstract(Nt)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}_test')
        g.save(path)

if args.mode == 'simple_distractors':
    save_dir = 'data/simple_distractors'
    No_range = range(3, 31) # dataset n_objs
    # No = int(args.No) # number of objects in a dataset
    Ns = 10000 # number of train examples
    Nt = 5000 # number of validation/test examples
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for No in No_range:

        g = gen.SameConfigGen(n=No)
        ref = g.ref_state_list
        g.generate_alternative_distractors(Ns)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}')
        g.save(path)

        # generate validation dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate_alternative_distractors(Nt)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}_val')
        g.save(path)
        
        # generate test dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate_alternative_distractors(Nt)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}_test')
        g.save(path)

if args.mode == 'simple_abstract_distractors':
    save_dir = 'data/simple_abstract_distractors'
    No_range = range(3, 31) # dataset n_objs
    # No = int(args.No) # number of objects in a dataset
    Ns = 10000 # number of train examples
    Nt = 5000 # number of validation/test examples
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for No in No_range:

        g = gen.SameConfigGen(n=No)
        ref = g.ref_state_list
        g.generate_abstract_distractors(Ns)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}')
        g.save(path)

        # generate validation dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate_abstract_distractors(Nt)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}_val')
        g.save(path)
        
        # generate test dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate_abstract_distractors(Nt)
        path = op.join(
            save_dir,
            f'{No}_0_{Ns}_test')
        g.save(path)

if args.mode == 'rotcur_abstract':
    # double setting, rotation curriculum, perturbed version
    save_dir = 'data/double_abstract'
    Nc = 10 # number of different datasets
    No_min = int(args.No) # minimum number of objects in a dataset
    if args.No_max is None:
        No_max = int(args.No)
    else:
        No_max = int(args.No_max) # maximum number of objects in a dataset
    Ns = 100000 # number of train examples
    Nt = 10000 # number of validation/test examples
    cur_list = [
        (pi/10, 2*pi),
        (pi/2 + pi/10, 2*pi),
        (pi + pi/10, 2*pi),
        (3*pi/2 + pi/10, 2*pi),
        None]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, cur in enumerate(cur_list):
        g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
        g.r_ex_range = cur
        g.generate_abstract(Ns)
        path = op.join(
            save_dir,
            f'rotcur{i}_{No_min}_{No_max}_{Ns}')
        g.save(path)
    # validation
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate_abstract(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_val')
    g.save(path)
    # generate test dataset
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate_abstract(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_test')
    g.save(path)

if args.mode == 'rotcur_distractors':
    # double setting, rotation curriculum, perturbed version
    save_dir = 'data/double_distractors'
    Nc = 10 # number of different datasets
    No_min = int(args.No) # minimum number of objects in a dataset
    if args.No_max is None:
        No_max = int(args.No)
    else:
        No_max = int(args.No_max) # maximum number of objects in a dataset
    Ns = 100000 # number of train examples
    Nt = 10000 # number of validation/test examples
    cur_list = [
        (pi/10, 2*pi),
        (pi/2 + pi/10, 2*pi),
        (pi + pi/10, 2*pi),
        (3*pi/2 + pi/10, 2*pi),
        None]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, cur in enumerate(cur_list):
        g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
        g.r_ex_range = cur
        g.generate_alternative_distractors(Ns)
        path = op.join(
            save_dir,
            f'rotcur{i}_{No_min}_{No_max}_{Ns}')
        g.save(path)
    # validation
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate_alternative_distractors(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_val')
    g.save(path)
    # generate test dataset
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate_alternative_distractors(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_test')
    g.save(path)

if args.mode == 'rotcur_abstract_distractors':
    # double setting, rotation curriculum, perturbed version
    save_dir = 'data/double_abstract_distractors'
    Nc = 10 # number of different datasets
    No_min = int(args.No) # minimum number of objects in a dataset
    if args.No_max is None:
        No_max = int(args.No)
    else:
        No_max = int(args.No_max) # maximum number of objects in a dataset
    Ns = 100000 # number of train examples
    Nt = 10000 # number of validation/test examples
    cur_list = [
        (pi/10, 2*pi),
        (pi/2 + pi/10, 2*pi),
        (pi + pi/10, 2*pi),
        (3*pi/2 + pi/10, 2*pi),
        None]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, cur in enumerate(cur_list):
        g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
        g.r_ex_range = cur
        g.generate_abstract_distractors(Ns)
        path = op.join(
            save_dir,
            f'rotcur{i}_{No_min}_{No_max}_{Ns}')
        g.save(path)
    # validation
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate_abstract_distractors(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_val')
    g.save(path)
    # generate test dataset
    cur = None
    g = gen.CompareConfigGen(n_min=No_min, n_max=No_max)
    g.r_ex_range = cur
    g.generate_abstract_distractors(Nt)
    path = op.join(
        save_dir,
        f'rotcur{i}_{No_min}_{No_max}_{Nt}_test')
    g.save(path)