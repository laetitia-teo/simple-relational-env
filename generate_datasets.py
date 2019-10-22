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

n = 100
No = 5
Ns = 2

env = Env(16, 20)
env.random_config(3)

gen = SimpleTaskGen(env, 3)
rec = gen.generate_mix(Ns, No, n, rotations=True, record=True)
gen.save('data/simple_task/testrotations.txt') # we dont save the images
ds = ObjectDataset('data/simple_task/testrotations.txt', epsilon=1/(No*Ns))
ds.process()

with open('data/simple_task/recording', 'wb') as f:
    pickle.dump(rec, f)
with open('data/simple_task/dataset_binaries/testrotations', 'wb') as f:
    pickle.dump(ds, f)

