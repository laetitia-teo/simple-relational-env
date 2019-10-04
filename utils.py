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
            print(str_list)
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