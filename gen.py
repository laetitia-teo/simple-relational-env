"""
This file defines generating processes for data using the environments defined
in the env module.

The first simple classification task should be the following : with a fixed 
number of objects, train a model to recognise object configuration,
irrespective of object position.
"""
import os.path as op
import pickle
import numpy as np

import torch

from glob import glob

from dataset import Dataset
from env import SamplingTimeout
from utils import to_file, from_file

class AbstractGen():
    """
    Generator abstract class.

    Defines all the methods that should be implemented.
    """

    def __init__(self):
        pass

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
        self._config_id = 0

    def _generate_configs(self, n):
        """
        Generates the reference spatial configuration and its perturbations.
        
        TODO : how to manage SamplingTimeouts ?

        Arguments :
            - n : number of output states
        """
        self._env.reset()
        # generate reference config
        self._env.random_config(self.n_objects)
        ref_state = self._env.to_state_list()
        self._configs.append((ref_state, self._config_id))
        for _ in range(n - 1):
            self._env.from_state_list(ref_state)
            self._env.random_transformation()
            self._configs.append((self._env.to_state_list(), self._config_id))
        self._config_id += 1

    def generate(self, n_configs, n, restart=True):
        """
        Genarates n_configs * n states.

        Arguments :
            - n_configs (int) : number of total different configurations.
            - n (int) : number of versions of the same configuration.
            - restart (bool) : Whether to start from 0 for the configuration
                indices. Default is True.
        """
        if restart:
            self._env.reset()
            self._config_id = 0
        for _ in range(n_configs):
            self._generate_configs(n)

    def save(self, path, save_images=True, img_path='images'):
        """
        Saves the current configurations to a text file at path.
        """
        to_file(self._configs, path)
        if save_images:
            img_count = 0
            for state, idx in self._configs:
                img_name = 'img' + str(img_count) + '.jpg'
                self._env.from_state_list(state)
                self._env.save(op.join(img_path, img_name), save_image=True)
                self._env.reset()
                img_count += 1

    def load(self, path):
        """
        Loads the configurations at path.
        """
        self._configs = from_file(path)
        self._config_id = len(self._configs) # this assumes the loaded data
                                             # begin at 0 and increment