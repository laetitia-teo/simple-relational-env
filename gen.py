"""
This file defines generating processes for data using the environments defined
in the env module.

The first simple classification task should be the following : with a fixed 
number of objects, train a model to recognise object configuration,
irrespective of object position.
"""

import numpy as np

import torch

class AbstractGen():
    """
    Generator abstract class.

    Defines all the methods that should be implemented.
    """

    def __init__(self):
        self.commands = []

class ConfigTaskGen(AbstractGen):
    """docstring for ConfigTaskGen"""
    def __init__(self, env):
        super(ConfigTaskGen, self).__init__()
        self.env = env
        self.configs = []

    def generate_configs(self):
        """
        Generates all the spatial configurations, with the associated ids.
        The configurations should be legal in the environment.
        """
        pass

    def perturb_configs(self):
        """
        Applies random spatial perturbations (translations, small angle
        rotations) to the configurations.
        """
        pass

    def generate_dataset(self):
        """
        Generates the dataset of configurations.
        Uses the dataset class.
        """
        pass