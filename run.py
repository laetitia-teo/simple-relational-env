"""
This is an executable script for running the Parts experiment (may be updated 
to include other experiments as well).

In the Parts challenge, a model has to learn to discriminate when a
configuration of objects is present in a bigger scene. 
"""
import os.path as op
import numpy as np
import torch

import baseline_models as bm
import graph_models as gm

from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser

N_SH = 3
N_OBJ = 3
B_SIZE = 8 # batch size for graphs is small, because of edges
L_RATE = 1e-3
N_EPOCHS = 10
F_OBJ = 10
H = 16


