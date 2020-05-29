"""
Utilities for the baseline models, such as data transformation functions.
"""

import numpy as np
import torch

from scipy.sparse import coo_matrix

def make_data_to_mlp_inputs(n, cuda=False):
    def data_to_mlp_inputs(data):
        """
        Transforms our data for mlp baselines.
        n is number of objects, that has to be specified in advance.
        The number of objects must be constant, equal to n.
        """
        x1, x2, labels, indices, batch1, batch2 = data
        f_x = x1.shape[-1]
        x1 = x1.reshape((-1, n * f_x))
        try:
            x2 = x2.reshape((-1, n * f_x))
        except AttributeError:
            x2 = []
        return x1, x2, labels, indices
    return data_to_mlp_inputs

def make_data_to_seq(n):
    def data_to_seq(data):
        """
        Transforms our data for mlp baselines.
        n is number of objects, that has to be specified in advance.
        The number of objects must be constant, equal to n.
        """
        x1, x2, labels, indices, batch1, batch2 = data
        f_x = x1.shape[-1]
        x1 = x1.reshape((-1, n, f_x)).permute(1, 0, 2)
        try:
            x2 = x2.reshape((-1, n, f_x)).permute(1, 0, 2)
        except AttributeError:
            x2 = []
        return x1, x2, labels, indices
    return data_to_seq

def var_tensor(x, batch, n):
    N = len(batch)
    bsize = x.shape[0]
    f_x = x.shape[1:]
    coo = coo_matrix((
        np.empty(N),
        (batch.numpy(),
        np.arange(N))))
    idxptr = torch.tensor(coo.tocsr().indptr)
    s = torch.sparse.FloatTensor(
        torch.cat([batch, torch.arange(n)], 0),
        x,
        torch.Size([bsize, N + n]))
    data = torch.empty((0, n) + f_x)
    for i in range(bsize):
        bidx = idxptr[i]
        eidx = idxptr[i + 1]
        data = torch.cat([data, x[bidx:eidx]], 0)
    return data

def densify(x, batch, n):
    """
    Takes in a tensor of data x, all listed in the 0th dimension, and the
    batch index corresponding to it, and returns a tensor of size [bsize, n, f_x]
    with scenes 0-padded to achieve 0 number of objects.
    """
    N = len(batch)
    bsize = batch[-1] + 1
    f_x = x.shape[-1]
    
    coo = coo_matrix((
        np.empty(N),
        (batch.numpy(),
        np.arange(N))))

    idxptr = torch.tensor(coo.tocsr().indptr)
    xd = torch.zeros(bsize, n * f_x)

    for i in range(bsize):
        idx1, idx2 = idxptr[i], idxptr[i+1]
        l = idx2 - idx1
        k = int(l * f_x) # length of current scene
        xd[i, :k] = x[idx1:idx2].reshape((k,))

    return xd

def data_to_baseline_var(n):
    def data_to_mlp_var(data):
        """
        Same as above, but with variable number of objects allowed.
        Inputs are zero-padded when they contain number of objects under the max.
        n is max number of objects.
        Maybe a bit slow.
        """
        x1, x2, labels, indices, batch1, batch2 = data
        data1 = densify(x1, batch1, n)
        if x2 is None:
            data2 = None
        else:
            data2 = densify(x2, batch2, n)
        return data1, data2, labels, indices
    return data_to_mlp_var

def data_to_obj_seq_parts(data):
    """
    Transforms the output of the Parts dataset DataLoader into sequences of 
    objects, to be fed to a LSTM.
    """
    x1, x2, labels, indices, batch1, batch2 = data
    f_x = x1.shape[-1]
    max_size1 = 4 # max number of objects in a target scene
    max_size2 = 13 # maximum number of objects in a ref scene
    bsize = batch1[-1]

    seq1 = torch.zeros((max_size1, 0, f_x))
    # indices of the objects in the corresponding scenes
    # i = (batch1.unsqueeze(1) == torch.arange(bsize)).float().T
    # seq1 = i.unsqueeze(-1) * x1.unsqueeze(0)
    # -> too fancy
    # quite suboptimal if batch size is too big
    for i in range(bsize):
        # ith batch
        ith = (batch1 == i).nonzero(as_tuple=True)
        s = torch.zeros((max_size1, f_x))
        s[:len(ith)] = x1[ith]
        seq1 = torch.cat([seq1, s], 1)

    # same for reference objects
    seq2 = torch.zeros((0, max_size2, f_x))
    for i in range(bsize):
        # ith batch
        ith = (batch2 == i).nonzero(as_tuple=True)
        s = torch.zeros((max_size2, f_x))
        s[:len(ith)] = x2[ith]
        seq2 = torch.cat([seq2, s], 1)

    return seq1, seq2
