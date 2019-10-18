"""
This file is a helper file to generate train and test datasets.
"""
import pickle
import numpy as np
import torch

from env import Env
from gen import SimpleTaskGen
from dataset import ObjectDataset

n = 100
No = 5
Ns = 2

env = Env(16, 200)
env.random_config(3)

gen = SimpleTaskGen(env, 3)
gen.generate_mix(Ns, No, n)
gen.save('data/simple_task/test2.txt') # we dont save the images
ds = ObjectDataset('data/simple_task/test2.txt', epsilon=1/(No*Ns))
ds.process()

with open('data/simple_task/dataset_binaries/testobject2', 'wb') as f:
    pickle.dump(ds, f)