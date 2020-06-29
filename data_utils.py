"""
Small module providing pytorch dataloaders for the different SpatialSIm 
datasets.
"""

import os.path as op
import json

import torch

from torch.utils.data import Dataset, DataLoader

DATAPATH = op.join('data', 'export')
BATCH_SIZE = 128

### collate_fn for DataLoader

def collate_fn(batch):
    """
    Custom collate_fn for the DataLoader: concatenates all data in a batch
    in an appropriate way.
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
                [torch.ones(len(t), dtype=torch.long) * i 
                    for i, t in enumerate(transposed[0])]))
        # r_batch
        if len(elem) == 3:
            l.append(
                collate_fn(
                    [torch.ones(len(t), dtype=torch.long) * i 
                        for i, t in enumerate(transposed[1])]))
        return l

### Dataset and DataLoader

class SpatialSimDataset(Dataset):
    """
    Dataset instance for the Identification and Comparison tasks.
    """
    def __init__(self, load_path):
        super().__init__()

        with open(load_path, 'r') as f:
            jsonstring = f.readlines()[0]
        datadict = json.loads(jsonstring)

        self.data = datadict['data']
        self.labels = datadict['labels']

        if len(self.data[0]) > 2:
            self.task = 'Identification'
        else:
            self.task = 'Comparison'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.task == 'Identification':
            return torch.Tensor(self.data[idx]), torch.Tensor([self.labels[idx]])
        elif self.task == 'Comparison':
            target, ref = self.data[idx]
            return (
                torch.Tensor(target),
                torch.Tensor(ref),
                torch.Tensor([self.labels[idx]]),
            )

class SpatialSimDataLoader(DataLoader):
    """
    Ready-made class for the SpatialSim DataLoader.
    """
    def __init__(self, dataset, shuffle=True, batch_size=BATCH_SIZE):

        super().__init__(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=collate_fn)

### Example use

# if __name__ == '__main__':

# Identification
dsname = 'IDS_5'
ds = SpatialSimDataset(op.join(DATAPATH, dsname))
dl = SpatialSimDataLoader(ds)

# Comparison
dsname = 'CDS_3_8_0'
ds2 = SpatialSimDataset(op.join(DATAPATH, dsname))
dl2 = SpatialSimDataLoader(ds2)