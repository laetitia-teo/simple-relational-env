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
from tqdm import tqdm

from dataset import Dataset
from env import SamplingTimeout
from utils import to_file, from_file

N_SH = 3

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

    def _generate_configs(self,
                          n,
                          ref_state=None,
                          rotations=False,
                          record=False,
                          shuffle=False):
        """
        Generates the reference spatial configuration and its perturbations.
        
        TODO : how to manage SamplingTimeouts ?

        Arguments :
            - n : number of output states
            - ref_state (list of object vectors) : reference state. If not
                provided, a new one is generated at random.
        """
        self._env.reset()
        if record:
            rec = {'translations': [],
                   'scalings': [],
                   'rotations': []}
        # generate reference config
        if ref_state is None:
            self._env.random_config(self.n_objects)
            ref_state = self._env.to_state_list()
        self._configs.append((ref_state, self._config_id))
        for _ in range(n - 1):
            self._env.from_state_list(ref_state)
            if shuffle:
                self._env.shuffle_objects()
            amount, scale, phi = self._env.random_transformation(
                rotations=rotations)
            self._configs.append((self._env.to_state_list(), self._config_id))
            if record:
                rec['translations'].append(amount)
                rec['scalings'].append(scale)
                rec['rotations'].append(phi)
        self._config_id += 1
        if record:
            return rec

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

    def generate_mix(self,
                     n_obj_configs,
                     n_spatial_configs,
                     n,
                     restart=True,
                     rotations=False,
                     record=False,
                     shuffle=False):
        """
        Generates configs with object re-mixing.

        Arguments :
            - n_obj_configs (int) : number of different configurations for the
                size, color, and orientation of the objects.
            - n_spatial_configs (int) : number of different shufflings of the
                same objects (same color, size and orientation)
            - n (int) : number of transformations of the same spatial
                configuration.
            - restart (bool) : whether or not to restart from scratch,
                resetting the internal state of the generator. Defaults to True
            - rotations (bool) : whether or not to have rotations in our
                generating process. Defaults to False.
            - record (bool) : whether or not to record the translation vectors,
                scaling factors, and rotation angles used in the generating 
                process.
            - shuffle (bool) : whether or not to shuffle the order of objects
                in the generating process. If False, the objects are always in
                the same order, across configurations with the same objects.
        """
        if restart:
            self._env.reset()
            self._config_id = 0
        if record:
            recs = {'translations': [],
                   'scalings': [],
                   'rotations': []}
        print('Generating %s object configs :' % n_obj_configs)
        for i in range(n_obj_configs):
            # generate ref state
            self._env.reset()
            self._env.random_config(self.n_objects)
            ref_state = self._env.to_state_list()
            for j in tqdm(range(n_spatial_configs)):
                self._env.reset()
                self._env.from_state_list(ref_state)
                self._env.random_mix()
                state = self._env.to_state_list()
                rec = self._generate_configs(n,
                                             state,
                                             rotations,
                                             record,
                                             shuffle)
                if record and rec is not None:
                    recs['translations'] += rec['translations']
                    recs['scalings'] += rec['scalings']
                    recs['rotations'] += rec['rotations']
        if record:
            return recs

    def save(self, path, img_path=None):
        """
        Saves the current configurations to a text file at path.
        """
        to_file(self._configs, path)
        print('generating images')
        if img_path is not None:
            img_count = 0
            for state, idx in tqdm(self._configs):
                img_name = 'img' + str(img_count) + '.jpg'
                self._env.from_state_list(state)
                self._env.save_image(op.join(img_path, img_name))
                self._env.reset()
                img_count += 1

    def load(self, path):
        """
        Loads the configurations at path.
        """
        self._configs = from_file(path)
        self._config_id = len(self._configs) # this assumes the loaded data
                                             # begin at 0 and increment