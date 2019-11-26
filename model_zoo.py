"""
This file defines all the models used in our experiments.

An individual model, with its set of hyperparameters, is called a taxon (by 
analogy to zoology). When performing experiments, we record the name of the 
taxon we used to learn, and when saving and loading the parameters of the model
we use this reference.
"""
import os.path as op
import torch

import graph_models as gm

from glob import glob

from graph_utils import data_to_graph_parts
from training_utils import data_to_clss_parts

class GraphTaxon():
    """
    Encapsulates a model, and all is functionnalities.
    """
    def __init__(self):
        self.model = NotImplemented
        self.opt = NotImplemented
        self.data_fn = data_to_graph_parts
        self.clss_fn = data_to_clss_parts

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def components(self):
        return self.model, self.opt

    def __call__(self, data):
        return self.model(*self.data_fn(data))

class GMNv2_16_10_1(GraphTaxon):
    def __init__(self):
        super(GMNv2_10, self).__init__()
        self.model = gm.GraphMatchingv2([16, 16], 10, 1, f_dict)
        self.opt = torch.optim.Adam(model.parameters(), lr=L_RATE)