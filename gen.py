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

from env import SamplingTimeout
from env import Env
from dataset import Dataset
from utils import to_file, from_file

N_SH = 3
DTYPE = torch.float32

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

class PartsGen():
    """
    Class for the generator of the Parts task.

    Can also be used as a dataset, to plug a pytorch Dataloader.
    """
    def __init__(self, env=None):
        """
        Initialize the Parts task generator. 

        The Parts dataset trains a model to recognize if a target configuration
        is present or not in a given reference. The target objects are always
        present in the reference, but may be present in a different spatial
        arrangement (this is a negative example).

        The constructor defines the range of the number of objects in the
        target range_t, and the range of the number of additional distractor
        objects range_d. If no distractors are provided, the tasks comes back
        to SimpleTask, judging the similarity of two scenes.
        """
        if env is None:
            env = Env(16, 20)
        self.range_t = [2, 5]
        self.range_d = [0, 10]

        # data
        self.reset()

    def reset(self):
        self.targets = []
        self.t_batch = []
        self.refs = []
        self.r_batch = []
        self.labels = []

    def gen_one(self):
        """
        Generates one pair of true-false examples associated with a given
        target.

        The targets are perturbed (similarity + small noise) before being
        completed with distractors.
        """
        self.env.reset()
        n_t = np.random.randint(*self.range_t)
        self.env.random_config(n_t)
        target = self.env.to_state_list()
        # generate positive example
        self.env.shuffle_objects()
        self.env.random_transformation()
        n_d = np.random.randint(*self.range_d)
        self.env.random_config(n_d) # add random objects
        trueref = self.env.to_state_list()
        # generate negative example
        self.env.reset()
        self.env.from_state_list(target)
        self.env.shuffle_objects() # shuffle order of the objects
        self.env.random_mix() # mix config
        self.random_transformation()
        n_d = np.random.randint(*self.range_d)
        self.env.random_config(n_d) # add random objects
        falseref = self.env.to_state_list()
        target = target
        return target, trueref, falseref

    def generate(self, N):
        """
        Generates a dataset of N positive and N negative examples.

        Arguments :
            - N (int) : half of the dataset length

        Generates:
            - targets (list of vectors): list of all the target objets;
            - t_batch (list of ints): list of indices linking the target
                objects to their corresponding scene index;
            - refs (list of object vectors): list of all the reference
                objects;
            - r_batch (list of ints): list of indices linking the reference
                objects to their corresponding scene index;
            - labels (list of ints): list of scene labels.
        """
        print('generating dataset of %s examples :' % 2 * N)
        for i in tqdm(range(N)):
            target, trueref, falseref = self.gen_one()
            n_t = len(target)
            n_r1 = len(trueref)
            n_r2 = len(falseref)
            self.targets += 2 * target
            self.t_batch += n_t * [2*i] + n_t * [2*i + 1]
            self.refs += trueref + falseref
            self.r_batch += n_r1 * [2*i] + n_r2 * [2*i + 1]
            self.labels += [1, 0]
    
    def write_targets(self, f):
        """
        Writes the targets, and the t_batch to file f. Every object vector
        is prepended its batch index.
        """
        f.write('targets\n')
        for i, obj in enumerate(self.targets):
            f.write(str(self.t_batch[i]) + ' ')
            for num in obj:
                f.write(str(num) + ' ')
            f.write('\n')

    def write_refs(self, f):
        """
        Writes the refs, and the r_batch to file f. Every object vector
        is prepended its batch index.
        """
        f.write('refs\n')
        for i, obj in enumerate(self.refs):
            f.write(str(self.r_batch[i]) + ' ')
            for num in obj:
                f.write(str(num) + ' ')
            f.write('\n')

    def write_labels(self, f):
        """
        Writes the labels.
        """
        f.write('labels\n')
        for label in self.labels:
            f.write(str(label) + '\n')

    def read_targets(self, lineit):
        """
        Takes in an iterator of the lines read.
        Reads the targets and t_batch from lines, returns targets, t_batch
        and stopping index.
        """
        targets = []
        t_batch = []
        line = next(lineit)
        while line != 'refs':
            if line == 'targets':
                pass
            else:
                linelist = line.split(' ')
                t_batch.append(linelist[0])
                targets.append(np.array(linelist[1:]))
            line = next(lineit)
        return targets, t_batch

    def read_refs(self, lineit):
        """
        Takes in an iterator of the lines read.
        Reads the refs and r_batch from lines, returns refs, r_batch
        and stopping index.
        """
        refs = []
        r_batch = []
        line = next(lineit)
        while line != 'labels':
            linelist = line.split(' ')
            r_batch.append(linelist[0])
            refs.append(np.array(linelist[1:], dtype=float))
            line = next(lineit)
        return refs, r_batch

    def read_labels(self, lineit):
        """
        Reads the label from an iterator on the file lines.
        """
        labels = []
        try:
            line = next(lineit)
            labels.append(int(line))
        except StopIteration:
            return labels

    def save(self, path):
        """
        Saves the dataset as a file. 
        """
        with open(path, 'w') as f:
            self.write_targets(f)
            self.write_refs(f)
            self.write_labels(f)

    def load(self, path):
        """
        Reads previously saved generator data.
        """
        with open(path, 'r') as f:
            # reads from line iterator
            lines = f.readlines()
            lineit = iter(lines)
            targets, t_batch = self.read_targets(lineit)
            refs, r_batch = self.read_refs(lineit)
            labels = self.read_labels(lineit)
        # stores the data
        self.targets = targets
        self.t_batch = t_batch
        self.refs = refs
        self.r_batch = r_batch
        self.labels = labels
            
