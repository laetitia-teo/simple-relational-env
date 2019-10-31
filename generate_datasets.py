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

n_obj = 3
n = 100
No = 5
Ns = 2

env = Env(16, 20)
env.random_config(n_obj)

gen = SimpleTaskGen(env, n_obj)
rec = gen.generate_mix(Ns, No, n, rotations=False, record=False, shuffle=True)
print('saving objects')
gen.save('data/simple_task/shuffletest.txt') # we dont save the images
print('done')
ds = ObjectDataset('data/simple_task/shuffletest.txt', epsilon=1/(No*Ns))
ds.process()

# with open('data/simple_task/recording', 'wb') as f:
#     pickle.dump(rec, f)
print('dumping comparison dataset')
with open('data/simple_task/dataset_binaries/shuffletest', 'wb') as f:
    pickle.dump(ds, f)
print('done')
