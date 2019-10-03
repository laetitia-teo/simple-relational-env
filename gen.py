"""
This file defines generating processes for data using the environments defined
in the env module.

The first simple classification task should be the following : with a fixed 
number of objects, train a model to recognise object configuration,
irrespective of object position.
"""
import pickle
import numpy as np

import torch

from dataset import Dataset

from env import SamplingTimeout

class AbstractGen():
    """
    Generator abstract class.

    Defines all the methods that should be implemented.
    """

    def __init__(self):
        self.commands = []

class SimpleTaskGen(AbstractGen):
    """
    docstring for SimpleTaskGen

    Change name for something more sexy.
    """
    def __init__(self, env, n_objects):
        super(SimpleTaskGen, self).__init__()
        self._env = env
        self._configs = []
        self.n_objects = n_objects

    def _generate_configs(self, n_configs):
        """
        Generates the reference spatial configuration
        """
        for idx in range(n_configs):
            try:
                self._env.reset()
                self._env.random_config()
                state_list = self._env.to_state_list()


    def _perturb_configs(self):
        """
        Applies random spatial perturbations (translations, small angle
        rotations) to the configurations.
        """
        pass

    def _generate_dataset(self, path):
        """
        Generates the dataset of configurations.
        Uses the dataset class.
        """
        dataset = Dataset()
        return dataset

    def generate(self, path):
        """
        Generates a dataset of shape configurations.
        """
        self._generate_configs()
        self._perturb_configs()
        return self._generate_dataset(path)