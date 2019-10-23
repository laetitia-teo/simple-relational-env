"""
This file is a helper file to generate train and test datasets.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from env import Env
from gen import SimpleTaskGen
from dataset import ObjectDataset

n_obj = 4
n = 100
No = 5
Ns = 2

env = Env(16, 20)
env.random_config(n_obj)

gen = SimpleTaskGen(env, n_obj)
rec = gen.generate_mix(Ns, No, n, rotations=False, record=False)
gen.save('data/simple_task/4objtest.txt') # we dont save the images
ds = ObjectDataset('data/simple_task/4objtest.txt', epsilon=1/(No*Ns))
ds.process()

# with open('data/simple_task/recording', 'wb') as f:
#     pickle.dump(rec, f)
with open('data/simple_task/dataset_binaries/4objtest', 'wb') as f:
    pickle.dump(ds, f)

