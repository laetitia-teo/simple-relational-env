"""
Utilities for the baseline models, such as data transformation functions.
"""

import torch

def data_to_obj_seq_parts(data):
    """
    Transforms the output of the Parts dataset DataLoader into sequences of 
    objects, to be fed to a LSTM.
    """
    x1, x2, labels, batch1, batch2 = data
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
        s[:len(ith)] = x[ith]
        seq1 = torch.cat([seq1, s], 1)

    # same for reference objects
    seq2 = torch.zeros((0, max_size2, f_x))
    for i in range(bsize):
        # ith batch
        ith = (batch2 == i).nonzero(as_tuple=True)
        s = torch.zeros((max_size2, f_x))
        s[:len(ith)] = x[ith]
        seq2 = torch.cat([seq2, s], 1)

    return seq1, seq2
