"""
Small utilities.
"""
import re
import os.path as op
import numpy as np
import torch

from glob import glob

### File utilities

def to_file(data, path):
    """
    Writes a list of vectors to a file at path.

    Arguments :
        data : the state data. It is a list containing (state, index) tuples.
            States are a list of arrays, indices are ints.

    TODO : make this more optimized and more general, for several data types.
    """
    with open(path, 'w') as f:
        f.write('\n')
        for vec_list, idx in data:
            f.write(str(idx) + '\n')
            for vec in vec_list:
                for num in vec:
                    f.write(str(num) + ' ')
                f.write('\n')
            f.write('\n')

def from_file(path, dtype=float):
    """
    Reads a list of vectors from a file at path.

    Returns :
        - A list of (list of arrays, int) tuples representing all the
            configurations in tha data with their configuration index.
    """
    with open(path, 'r') as f:
        data = []
        vec_list = []
        idx = 0
        for l in f.readlines():
            str_list = l.split(' ')
            if str_list == ['\n']:
                if vec_list:
                    data.append((vec_list, idx))
                vec_list = []
            elif len(str_list) == 1:
                idx = int(str_list[0])
            else:
                num_list = []
                for num in str_list[0:-1]:
                    num_list.append(dtype(num))
                vec_list.append(np.array(num_list))
    return data

### Cosine similarity functions

def norm2(t, dim):
    return (torch.sum(t**2, dim)**0.5)

def cos_sim(t1, t2):
    """
    Assumes the feature dimension is the last one.
    """
    t1 = t1.unsqueeze(0)
    t2 = t2.unsqueeze(1)
    sprod = torch.sum(t1 * t2, -1)
    nprod = (norm2(t1, -1) * norm2(t2, -1))
    return sprod / nprod

def sim(v1, v2):
    """
    Acts on two individual vectors.
    """
    s = torch.sum(v1 * v2, -1)
    return s / (norm2(v1, -1) * norm2(v2, -1))

def cosine_similarity(t, v):
    s = torch.sum(t * v, -1)
    n = norm2(t, -1) * norm2(v, -1)
    return s / n

# data

class Data():
    def __init__(self, x, y, edge_attr, edge_index, batch):
        self.x = x
        self.y = y
        self.edge_attr = edge_attr
        self.edge_index = edge_index
        self.batch = batch

    def clean_size(self, s):
        s = str(s)
        size = re.search(r'^torch.Size\((.*)\)$', s)[1]
        return size

    def __repr__(self):
        s = 'Data(x={0},'.format(self.clean_size(self.x.shape)) \
            + ' y={0},'.format(self.clean_size(self.y.shape)) \
            + ' edge_attr={0},'.format(self.clean_size(self.edge_attr.shape)) \
            + ' edge_index={0},'.format(self.clean_size(
                self.edge_index.shape)) \
            + ' batch={0})'.format(self.clean_size(self.batch.shape))
        return s
