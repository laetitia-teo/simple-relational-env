"""
Small utilities.
"""
import os.path as op
import numpy as np

from glob import glob

def to_file(data, path):
    """
    Writes a list of vectors to a file at path.

    Arguments :
        data : the state data. It is a list containing (state, index) tuples.
            States are a list of arrays, indices are ints.

    TODO : make this more optimized and more general, for several data types.
    """
    with open(path, 'a') as f:
        f.write('\n')
        for vec_list in data:
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
        for l in f.readlines():
            str_list = l.split(' ')
            if str_list == ['\n']:
                if vec_list:
                    data.append(vec_list)
                vec_list = []
            else:
                num_list = []
                idx = int(str_list[0]) # config id
                for num in str_list[1:-1]:
                    num_list.append(dtype(num))
                vec_list.append((np.array(num_list), idx))
    return data