"""
Class defining the Dataset classes, for handling of data of spatial
configurations of shapes in our environment.

The Datasets should allow saving and (dynamic) loading, and efficient batching
for downstream processing.
"""

import numpy as np
import torch

import utils as ut

from torch.utils.data import Dataset

class ObjectDataset(Dataset):
    """
    A Dataset class to hold our object data. Does not handle images of the
    state of the environment.
    """
    def __init__(self, data_path):
        """
        Initializes our object dataset. The data held by this dataset consists
        in state vectors for each object.

        Implementation notes : for now we'll yield vectors in batches of 1, 
        with (object vectors, index) tuples. index refers to the configuration
        index that is used to compare two different set of objects (equal 
        indices means equal configuartions).

        The configurations are stored as a list of (list of arrays, int)
        tuples, as is returned by ut.from_file().
        """
        self._configs = ut.from_file(data_path)
        self._nb_objects = 3

    def process(self):
        """
        Processes the configurations to produce the actual dataset.

        We iterate over all possible combinations of 2 configurations, and
        we build a tensor of all the objects (the three first for the first
        config and the three second for the second one), and we also return
        a tensor of size two which is equal to [1, 0] if the 2 configs are
        different, and [0, 1] if they are the same (as measured by the
        equality of the config indices).
        """
        self.data = []
        for vecs1, idx1 in self._configs:
            for vecs2, idx2 in self._configs:
                objects1 = torch.tensor(vecs1)
                objects2 = torch.tensor(vecs2)
                objects = torch.cat([objects1, objects2])
                print(objects.shape)
                clss = torch.zeros(2)
                if idx1 == idx2:
                    clss[1] = 1.
                else:
                    clss[0] = 1.
                self.data.append((objects, clss))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ImageDataset(Dataset):
    """
    A Dataset class to hold images of states.
    """
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getattr__(self, idx):
        return None