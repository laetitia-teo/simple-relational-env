import os.path as op
import pathlib
import torch

import graph_models

from argparse import ArgumentParser
from pydoc import locate

from run_utils import one_run, load_dl
from generate_configs import load_config

parser = ArgumentParser()
parser.add_argument('-c', '--config',
                    dest='config',
                    help='name of the config file',
                    default=None)

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

if __name__ == '__main__':
    # open log file
    path = op.join(s, 'expe%s' % expe_idx)
    for i in range(max(train_indices)):
        # data loading
        indices = [idx for idx, e in enumerate(train_indices) if e == i]
        train_dls = [
            load_dl(op.join(d, train_datasets[idx])) for idx in indices]
        indices = [idx for idx, e in enumerate(test_indices) if e == i]
        test_dls = [
            load_dl(op.join(d, test_datasets[idx])) for idx in indices]
        ipath = op.join(path, 'run%s' % i)
        for seed in seeds:
            t0 = time.time()
            np.random.seed(seed)
            torch.manual_seed(seed)
            # models
            for m_idx, m_str in enumerate(model_list):
                m = locate('graph_models.' + m_str)
                model = m(*hparam_list[m_idx])
                opt = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
                mpath = op.join(path, m_str)
                pathlib.Path(op.join(mpath, 'data')).mkdir(
                    parents=True, exist_ok=True)
                pathlib.Path(op.join(mpath, 'models')).mkdir(
                    parents=True, exist_ok=True)
                one_run(
                    i,
                    seed,
                    hparams['n_epochs'],
                    model,
                    opt,
                    train_dls,
                    test_dls,
                    mpath,
                    cuda=cuda)
    # close log file