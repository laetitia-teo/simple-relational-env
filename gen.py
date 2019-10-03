"""
This file defines generating processes for data using the environments defined
in the env module.

The first simple classification task should be the following : with a fixed 
number of objects, train a model to recognise object configuration,
irrespective of object position.
"""

import numpy as np

import torch

class AbstractGenerator():
    """
    Generator abstract class.

    Defines all the methods that should be implemented.
    """

    def __init__(self):
        self.commands = []