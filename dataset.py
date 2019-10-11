"""
Class defining the Dataset classes, for handling of data of spatial
configurations of shapes in our environment.

The Datasets should allow saving and (dynamic) loading, and efficient batching
for downstream processing.
"""

import numpy as np

import torch

from torch.utils.data import Dataset

class ObjectDataset(Dataset):
    """
    A Dataset class to hold our object data. Does not handle images of the
    state of the environment.
    """
    def __init__(self):
        pass