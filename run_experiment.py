import os
import shutil
import os.path as op
import pathlib
import time, datetime
import numpy as np
import torch

import graph_models

from argparse import ArgumentParser
from pydoc import locate

from run_utils import one_run, load_dl, load_model
from generate_config import load_config

parser = ArgumentParser()
parser.add_argument('-c', '--config',
                    dest='config',
                    help='name of the config file',
                    default='')

if __name__ == '__main__':
    args = parser.parse_args()

config = load_config(op.join('configs', args.config))

expe_idx = config['expe_idx']
d = config['load_dir']
s = config['save_dir']
train_datasets = config['train_datasets']
train_indices = config['train_dataset_indices']
test_datasets = config['test_datasets']
test_indices = config['test_dataset_indices']
seeds = config['seeds']
log_file_output_dir = s
model_list = config['models']
hparam_list = config['hparam_list']
hparams = config['hparams']
cuda = config['cuda']
try:
    preload_model = config['preload_model']
    load_idx = config['load_idx']
except KeyError:
    preload_model = False

double = (config['setting'] == 'double')

def copytree(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

def log(f, message):
    print(message, end='')
    f.write(message)

if __name__ == '__main__':
    path = op.join(s, 'expe%s' % expe_idx)
    # open log file
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    logfile = open(op.join(path, 'log'), 'w')
    log(logfile, 'started experiment {} at {}.\n'.format(
        expe_idx,
        str(datetime.datetime.now())))
    if preload_model:
        srcpath = op.join(s, 'expe%s' % load_idx)
    log(logfile, 'config file at path : %s\n' % op.join('configs', args.config))
    log(logfile, 'experiment details :\n\n')
    for k, v in config.items():
        log(logfile, '{} : {}\n'.format(k, v))
    try:
        maxi = max(train_indices) + 1
    except ValueError:
        maxi = 1
    for i in range(maxi):
        log(logfile, '\ntraining round %s\n' % i)
        # data loading
        train_i = [idx for idx, e in enumerate(train_indices) if e == i]
        train_dls = [
            load_dl(op.join(d, train_datasets[idx]), double) for idx in train_i]
        test_i = [idx for idx, e in enumerate(test_indices) if e == i]
        test_dls = [
            load_dl(op.join(d, test_datasets[idx]), double) for idx in test_i]
        log(logfile, 'train dls : {}\n'.format(
            [train_datasets[idx] for idx in train_i]))
        log(logfile, 'test dls : {}\n'.format(
            [test_datasets[idx] for idx in test_i]))
        for seed in seeds:
            log(logfile, '\nseed %s\n' % seed)
            t0 = time.time()
            np.random.seed(seed)
            torch.manual_seed(seed)
            # models
            for m_idx, m_str in enumerate(model_list):
                log(logfile, 'model %s\n' % m_str)
                m = locate('graph_models.' + m_str)
                model = m(*hparam_list[m_idx])
                opt = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
                mpath = op.join(path, m_str)
                pathlib.Path(op.join(mpath, 'data')).mkdir(
                    parents=True, exist_ok=True)
                if not preload_model:
                    pathlib.Path(op.join(mpath, 'models')).mkdir(
                        parents=True, exist_ok=True)
                else:
                    srcmpath = op.join(srcpath, m_str, 'models')
                    copytree(srcmpath, op.join(mpath, 'models'))
                one_run(
                    i,
                    seed,
                    hparams['n_epochs'],
                    model,
                    opt,
                    train_dls,
                    test_dls,
                    mpath,
                    cuda=cuda,
                    preload=preload_model)
                log(logfile, 'run completed, results saved in {}\n'.format(
                    mpath))
            log(logfile, 'training time for one seed %s\n' % (time.time() - t0))
    # close log file
    log(logfile, 'finished experiment {} at {}'.format(
        expe_idx,
        str(datetime.datetime.now())))
    logfile.close()