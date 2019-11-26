"""
A module for testing out models.

Defines various functions for testing stuff.
"""
import os.path as op
import torch

from glob import glob

from training_utils import load_dl_parts

def test_models(save_dir, test_list, taxon):
    """
    Tests all the models listed in save_dir on the datasets loaded from the
    test list (that lists the names of the datasets to be loaded).

    taxon is the model used.
    """
    test_dls = []
    for name in test_list:
        print(name)
        test_dls.append(load_dl_parts(name))
    save_list = sorted(glob(op.join(save_dir, '*.pt')))
    for model_path in save_list:
         ...