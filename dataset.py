"""
Class defining the Dataset classes, for handling of data of spatial
configurations of shapes in our environment.

The Datasets should allow saving and (dynamic) loading, and efficient batching
for downstream processing.
"""
import time
import os.path as op
import pickle
import numpy as np
import torch

import utils as ut

from tqdm import tqdm
from torch.utils.data import Dataset

N_SH = 3
DTYPE = torch.float32
CUDA_DTYPE = torch.cuda.FloatTensor
ITYPE = torch.long
CUDA_ITYPE = torch.cuda.LongTensor

### Utils ###

def collate_fn(batch):
    """
    Custom collate_fn, based on the default one in pytorch, for concatenating
    data on the first dimension instead of adding a new dimension in which to
    batch data.

    Assumes the data is provided as a tuple of torch.Tensors, and concatenates
    along the first dimension on each tensor.

    When used in a pytorch DataLoader, returns batches that have the graph 
    nodes as first and second elements for both scenes, labels as third element
    and batches for the first and second graph as fourth and fifth element.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif isinstance(elem, tuple):
        transposed = list(zip(*batch)) # we lose memory here
        l = [collate_fn(samples) for samples in transposed]
        # t_batch
        l.append(
            collate_fn(
                [torch.ones(len(t), dtype=ITYPE) * i 
                    for i, t in enumerate(transposed[0])]))
        # r_batch
        l.append(
            collate_fn(
                [torch.ones(len(t), dtype=ITYPE) * i 
                    for i, t in enumerate(transposed[1])]))
        return l

### Dataset ###

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

class PartsDataset(Dataset):
    """
    Class for the Parts task.
    """
    def __init__(self,
                 targets,
                 t_batch,
                 refs,
                 r_batch,
                 labels,
                 indices=None,
                 task_type='scene',
                 label_type='long'):
        """
        Initializes the Parts Dataset.
        The inputs are the outputs of the Parts generator, defined in the gen
        module (as lists).

        When indices is not given, we compute them by hand. Since this is a
        costly operation, we prefer to write them to a file.
        """
        self.task_type = task_type
        if label_type == 'long':
            LABELTYPE = torch.long
        if label_type == 'float':
            LABELTYPE = torch.float

        self.targets = torch.tensor(targets, dtype=DTYPE)
        self.t_batch = torch.tensor(t_batch, dtype=ITYPE)
        self.refs = torch.tensor(refs, dtype=DTYPE)
        self.r_batch = torch.tensor(r_batch, dtype=ITYPE)
        self.labels = torch.tensor(labels, dtype=LABELTYPE)

        if indices is None:
            self.t_idx = []
            self.r_idx = []
            # build lists of all the indices corresponding to one set of
            # (target, reference, label) for efficient access
            # this is O(n^2) ! fix this
            for idx in range(len(self.labels)):
                t = (self.t_batch == idx).nonzero(as_tuple=True)[0]
                r = (self.r_batch == idx).nonzero(as_tuple=True)[0]
                self.t_idx.append(t)
                self.r_idx.append(r)
        else:
            t_idx, r_idx = indices
            self.t_idx = t_idx
            self.r_idx = r_idx

        self.get = self._get_maker()

    def _get_maker(self):
        get = None
        if self.task_type == 'scene':
            def get(idx):
                target = self.targets[self.t_idx[idx]]
                ref = self.refs[self.r_idx[idx]]
                label = self.labels[idx]
                return target, ref, label
        if self.task_type == 'object':
            def get(idx):
                target = self.targets[self.t_idx[idx]]
                ref = self.refs[self.r_idx[idx]]
                label = self.labels[self.r_idx[idx]]
                return target, ref, label
        return get

    def __len__(self):
        return self.t_batch[-1]

    def __getitem__(self, idx):
        return self.get(idx)

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

