"""
This file defines simple baseline models for scene classification.

The models are :

A simple MLP taking in the environment state (how to adapt to different
number of objects ?)

A CNN-based model that works directly from pixels (with a pretrained 
embedding maybe - generic on other images, or maybe use the latent layer of a 
VAE trained to recognise the shapes)

Other interesting models to consider : LSTM with attention (taking in the 
objects as a sequence), transformer models. We'll see.
"""

import numpy as np

import torch

class MLPBaseline(object):
    """docstring for MLPBaseline"""
    def __init__(self, arg):
        super(MLPBaseline, self).__init__()
        self.arg = arg

class CNNBaseline(object):
    """docstring for CNNBaseline"""
    def __init__(self, arg):
        super(CNNBaseline, self).__init__()
        self.arg = arg
        