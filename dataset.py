"""
Class defining the Dataset classes, for handling of data of spatial
configurations of shapes in our environment.

The Datasets should allow saving and (dynamic) loading, and efficient batching
for downstream processing.
"""

import pickle
import numpy as np
import torch

import utils as ut

from tqdm import tqdm
from torch.utils.data import Dataset

N_SH = 3

class ObjectDataset(Dataset):
    """
    A Dataset class to hold our object data. Does not handle images of the
    state of the environment.
    """
    def __init__(self, data_path, epsilon=1, seed=42):
        """
        Initializes our object dataset. The data held by this dataset consists
        in state vectors for each object.

        Implementation notes : for now we'll yield vectors in batches of 1, 
        with (object vectors, index) tuples. index refers to the configuration
        index that is used to compare two different set of objects (equal 
        indices means equal configuartions).

        The configurations are stored as a list of (list of arrays, int)
        tuples, as is returned by ut.from_file().

        Arguments :
            - data_path : path to the data file
            - epsilon (float between 0 and 1) : proportion, for one
                configuration, of similar configurations in the dataset. This
                leads to a epsilon**2 to one imbalance in the comparison
                dataset for the positive ('same') class. To overcome this, we
                undersample the negative class by dropping negative examples
                with a probability of 1 - epsilon**2
        """
        self._configs = ut.from_file(data_path)
        self._nb_objects = 3
        self._seed = seed
        self.epsilon = epsilon
        np.random.seed(self._seed)

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
        rate = 1. / ((1 / self.epsilon) - 1.)
        print('building comparison dataset, %s configs' % len(self._configs))
        for vecs1, idx1 in tqdm(self._configs):
            for vecs2, idx2 in self._configs:
                clss = torch.zeros(2)
                if idx1 == idx2:
                    clss[1] = 1.
                else:
                    clss[0] = 1.
                    p = np.random.binomial(1, rate)
                    if not p:
                        continue # skip this negative sample
                objects1 = torch.tensor(vecs1, dtype=torch.float32)
                objects1[:, N_SH+1:N_SH+4] /= 255
                objects2 = torch.tensor(vecs2, dtype=torch.float32)
                objects2[:, N_SH+1:N_SH+4] /= 255
                objects = torch.cat([objects1, objects2])
                self.data.append((objects, clss))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, path):
        """
        Pickle the dataset for re-use.
        """
        with open(path, 'w') as f:
            pickle.dump(self, f)

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