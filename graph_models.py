"""
This file defines graph models for scene classification.

The model should be able to convert (maybe do this in a different class) the
object list it is given into a scene graph. The model should take two graphs as
input and output the probability of the two graphs being the same.

How to compute distance in the space of graphs ? Several approaches could be
considered : use global variable as a graph embedding, use graph-graph
comparisons with attention, or train a distance function on time-series of
graphs being jittred.
"""

import numpy as np

import torch

import torch_geometric

class GraphModel(object):
    """docstring for GraphModel"""
    def __init__(self, arg):
        super(GraphModel, self).__init__()
        self.arg = arg
