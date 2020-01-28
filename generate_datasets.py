"""
This file is a helper file to generate train and test datasets.
"""
import os.path as op
import gen

from argparse import ArgumentParser

tasklist = ['simple', 'parts']

parser = ArgumentParser()
parser.add_argument('-d', '--directory',
                    dest='directory',
                    help='directory for the generated datasets',
                    default='data/compare_config_alt')
parser.add_argument('-Nc', '--number-configs',
                     dest='Nc',
                     help='number of different generated datasets',
                     default='10')
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
parser.add_argument('-m', '--mode',
                     dest='mode',
                     help='simple or double dataset generation',
                     default='double')

args = parser.parse_args()

if args.mode == 'simple':
    for i in range(int(args.Nc)):
        g = gen.SameConfigGen(n=int(args.No))
        ref = g.ref_state_list
        g.generate(int(args.Ns))
        path = op.join(
            args.directory,
            '%s_%s_%s' % (int(args.No), i, int(args.Ns)))
        g.save(path)
        # generate validation dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate(int(args.Nt))
        path = op.join(
            args.directory,
            '%s_%s_%s_val' % (int(args.No), i, int(args.Nt)))
        g.save(path)
        # generate test dataset
        g = gen.SameConfigGen(ref_state_list=ref)
        g.generate(int(args.Nt))
        path = op.join(
            args.directory,
            '%s_%s_%s_test' % (int(args.No), i, int(args.Nt)))
        g.save(path)
if args.mode == 'double':
    g = gen.CompareConfigGen(n=int(args.No))
    g.generate_alternative(int(args.Ns))
    path = op.join(
        args.directory,
        '%s_%s_%s' % (int(args.No), 0, int(args.Ns)))
    g.save(path)
    # validation
    g.reset()
    g.generate_alternative(int(args.Nt))
    path = op.join(
        args.directory,
        '%s_%s_%s_val' % (int(args.No), 0, int(args.Nt)))
    g.save(path)
    # generate test dataset
    g.reset()
    g.generate_alternative(int(args.Nt))
    path = op.join(
        args.directory,
        '%s_%s_%s_test' % (int(args.No), 0, int(args.Nt)))
    g.save(path)