"""
Small utilities.
"""
import os.path as op
import numpy as np

from glob import glob

def to_file(data, path):
    """
    Writes a list of vectors to a file at path.

    TODO : make this more optimized and more general, for several data types.
    """
    with open(path, 'a') as f:
        for vec in data:
            for num in vec:
                f.write(str(num) + ' ')
            f.write('\n')

def from_file(path, dtype=float):
    """
    Reads a list of vectors from a file at path.
    """
    with open(path, 'r') as f:
        vec_list = []
        for l in f.readlines():
            str_list = l.split(' ')
            num_list = []
            for num in str_list[:-1]:
                num_list.append(dtype(num))
            vec_list.append(np.array(num_list))
    return vec_list