"""
This file defines generating processes for data using the environments defined
in the env module.

The first simple classification task should be the following : with a fixed 
number of objects, train a model to recognise object configuration,
irrespective of object position.
"""
import os.path as op
import random
import pickle
import numpy as np

import torch

from glob import glob
from tqdm import tqdm

from env import SamplingTimeout
from env import Env
from dataset import Dataset, PartsDataset
from utils import to_file, from_file

N_SH = 3
DTYPE = torch.float32

class Resample(Exception):
    """
    Raised when sampling of random transformations times out. This means the
    generated config is probably too big and we should drop it.
    """
    def __init__(self, message):
        self.message = message

class SimpleTaskGen():
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

class Gen():
    """
    Class for the generator of the Parts task.

    Can also be used as a dataset, to plug a pytorch Dataloader.
    """
    def __init__(self, env=None, n_d=None):
        """
        Initialize the Parts task generator. 

        The Parts dataset trains a model to recognize if a query configuration
        is present or not in a given reference. The query objects are always
        present in the reference, but may be present in a different spatial
        arrangement (this is a negative example).

        The constructor defines the range of the number of objects in the
        query range_t, and the range of the number of additional distractor
        objects range_d. If no distractors are provided, the tasks comes back
        to SimpleTask, judging the similarity of two scenes.

        This class defines no generating functions, which must be implemented
        according to the specific, concrete task at hand.
        """
        if env is None:
            self.env = Env(16, 20)
        self.range_t = [2, 5]
        self.range_d = [0, 6]

        self.n_d = n_d

        # data
        self.reset()

    def reset(self):
        self.queries = []
        self.q_batch = []
        self.worlds = []
        self.w_batch = []
        self.labels = []
        # careful, those are lists of torch.tensors
        self.t_idx = []
        self.r_idx = []

    def compute_access_indices(self):
        """
        Computes lists of target and reference indices for downsream efficient
        access. Computation-intensive.
        """
        t_batch = torch.tensor(self.t_batch, dtype=torch.float32)
        r_batch = torch.tensor(self.r_batch, dtype=torch.float32)
        for idx in range(len(self.labels)):
            self.t_idx.append((t_batch == idx).nonzero(as_tuple=True)[0])
            self.r_idx.append((r_batch == idx).nonzero(as_tuple=True)[0])
        self.t_idx = list(self.t)

    def gen_one(self):
        raise NotImplementedError()

    def generate(self, N):
        raise NotImplementedError()

    def generate_overfit(self, N, n):
        raise NotImplementedError()

    def write_queries(self, f):
        """
        Writes the queries, and the q_batch to file f. Every object vector
        is prepended its batch index.
        """
        f.write('targets\n')
        for i, obj in enumerate(self.queries):
            f.write(str(self.q_batch[i]) + ' ')
            for num in obj:
                f.write(str(num) + ' ')
            f.write('\n')

    def write_worlds(self, f):
        """
        Writes the worlds, and the w_batch to file f. Every object vector
        is prepended its batch index.
        """
        f.write('refs\n')
        for i, obj in enumerate(self.worlds):
            f.write(str(self.w_batch[i]) + ' ')
            for num in obj:
                f.write(str(num) + ' ')
            f.write('\n')

    def write_labels(self, f):
        """
        Writes the labels.
        """
        f.write('labels\n')
        for label in self.labels:
            f.write(str(label[0]) + '\n')

    def write_t_idx(self, f):
        """
        Writes the t-indices.
        """
        f.write('t_idx\n')
        for idx in self.t_idx:
            for i in idx:
                f.write(str(i.numpy()) + ' ')
            f.write('\n')

    def write_r_idx(self, f):
        """
        Writes the r-indices.
        """
        f.write('r_idx\n')
        for idx in self.r_idx:
            for i in idx:
                f.write(str(i.numpy()) + ' ')
            f.write('\n')

    def read_queries(self, lineit):
        """
        Takes in an iterator of the lines read.
        Reads the queries and q_batch from lines, returns queries, q_batch
        and stopping index.
        """
        queries = []
        q_batch = []
        line = next(lineit)
        while 'refs' not in line:
            if 'targets' in line:
                pass # first line
            else:
                linelist = line.split(' ')
                q_batch.append(int(linelist[0]))
                queries.append(np.array(linelist[1:-1], dtype=float))
            line = next(lineit)
        return queries, q_batch

    def read_worlds(self, lineit):
        """
        Takes in an iterator of the lines read.
        Reads the worlds and w_batch from lines, returns worlds, w_batch
        and stopping index.
        """
        worlds = []
        w_batch = []
        line = next(lineit)
        while 'labels' not in line:
            linelist = line.split(' ')
            w_batch.append(int(linelist[0]))
            worlds.append(np.array(linelist[1:-1], dtype=float))
            line = next(lineit)
        return worlds, w_batch

    def read_labels(self, lineit):
        """
        Reads the label from an iterator on the file lines.
        """
        labels = []
        try:
            line = next(lineit)
            while 't_idx' not in line:
                labels.append([int(line)])
                line = next(lineit)
        except StopIteration: # the file may also end here
            pass
        return labels

    def read_t_idx(self, lineit):
        """
        Reads the t-indices from the file line iterator.
        """
        t_idx = []
        line = next(lineit)
        while 'r_idx' not in line:
            linelist = line.split(' ')
            t_idx.append(torch.tensor(linelist[:-1], dtype=otrch.float32))
            line = next(lineit)
        return t_idx

    def read_r_idx(self, lineit):
        r_idx = []
        try:
            line = next(lineit)
            while 'some_other_stuff' not in line:
                linelist = line.split(' ')
                t_idx.append(torch.tensor(linelist[:-1], dtype=otrch.float32))
                line = next(lineit)
        except StopIteration:
            pass
        return r_idx

    def save(self, path, write_indices=True):
        """
        Saves the dataset as a file. 
        """
        with open(path, 'w') as f:
            self.write_queries(f)
            self.write_worlds(f)
            self.write_labels(f)
            if write_indices:
                self.write_t_idx(f)
                self.write_r_idx(f)

    def load(self, path, read_indices=True, replace=True):
        """
        Reads previously saved generator data.
        """
        with open(path, 'r') as f:
            # reads from line iterator
            lines = f.readlines()
            lineit = iter(lines)
            queries, q_batch = self.read_queries(lineit)
            worlds, w_batch = self.read_worlds(lineit)
            labels = self.read_labels(lineit)
            if read_indices:
                t_idx = self.read_t_idx(lineit)
                r_idx = self.read_r_idx(lineit)
        # stores the data
        if replace:
            self.queries = queries
            self.q_batch = q_batch
            self.worlds = worlds
            self.w_batch = w_batch
            self.labels = labels
            if read_indices:
                self.t_idx = t_idx
                self.r_idx = r_idx
        else:
            self.queries += queries
            self.q_batch += q_batch
            self.worlds += worlds
            self.w_batch += w_batch
            self.labels += labels

    def to_dataset(self, indices=True, n=None):
        """
        Creates a PartsDataset from the generated data and returns it.

        Arguments :
            - n (int) : allows to contol the dataset size for export.
        """
        if indices:
            ds = PartsDataset(self.queries[:n],
                              self.q_batch[:n],
                              self.worlds[:n],
                              self.w_batch[:n],
                              self.labels[:n],
                              (self.t_idx[:n], self.r_idx[:n]))
        else:
            ds = PartsDataset(self.queries[:n],
                              self.q_batch[:n],
                              self.worlds[:n],
                              self.w_batch[:n],
                              self.labels[:n])
        return ds

class PartsGen(Gen):
    """
    Generator for the Parts Task.
    """
    def __init__(self, env=None, n_d=None):
        """
        Initialize the Parts task generator. 

        The Parts dataset trains a model to recognize if a query configuration
        is present or not in a given reference. The query objects are always
        present in the reference, but may be present in a different spatial
        arrangement (this is a negative example).

        The constructor defines the range of the number of objects in the
        query range_t, and the range of the number of additional distractor
        objects range_d. If no distractors are provided, the tasks comes back
        to SimpleTask, judging the similarity of two scenes.

        This concrete class defines the generation functions.
        """
        super(PartsGen, self).__init__(env, n_d)

    def gen_one(self):
        """
        Generates one pair of true-false examples associated with a given
        query.

        The queries are perturbed (similarity + small noise) before being
        completed with distractors.
        """
        # Note : we could generate 4 by 4 with this code, by crossing queries
        # and worlds
        try:
            self.env.reset()
            n_t = np.random.randint(*self.range_t)
            if self.n_d is None:
                n_d = np.random.randint(*self.range_d)
            else:
                n_d = self.n_d
            self.env.random_config(n_t)
            query = self.env.to_state_list(norm=True)
            # generate positive example
            self.env.shuffle_objects()
            self.env.random_transformation()
            self.env.random_config(n_d) # add random objects
            trueworld = self.env.to_state_list(norm=True)
            # generate negative example
            if self.n_d is None:
                n_d = np.random.randint(*self.range_d)
            else:
                n_d = self.n_d
            self.env.reset()
            self.env.from_state_list(query, norm=True)
            self.env.shuffle_objects() # shuffle order of the objects
            self.env.random_mix() # mix config
            self.env.random_transformation()
            self.env.random_config(n_d) # add random objects
            falseworld = self.env.to_state_list(norm=True)
            query = query
            return query, trueworld, falseworld
        except SamplingTimeout:
            print('Sampling timed out, {} and {} objects'.format(n_t, n_d))
            raise Resample('Resample configuration')

    def generate(self, N):
        """
        Generates a dataset of N positive and N negative examples.

        Arguments :
            - N (int) : half of the dataset length

        Generates:
            - queries (list of vectors): list of all the query objets;
            - q_batch (list of ints): list of indices linking the query
                objects to their corresponding scene index;
            - worlds (list of object vectors): list of all the reference
                objects;
            - w_batch (list of ints): list of indices linking the reference
                objects to their corresponding scene index;
            - labels (list of ints): list of scene labels.
        """
        print('generating dataset of %s examples :' % (2 * N))
        for i in tqdm(range(N)):
            try:
                query, trueworld, falseworld = self.gen_one()
            except Resample:
                # We resample the config once
                # If there is a sampling timeout here, we let it pass
                query, trueworld, falseworld = self.gen_one()
            n_t = len(query)
            n_r1 = len(trueworld)
            n_r2 = len(falseworld)
            self.queries += 2 * query
            self.q_batch += n_t * [2*i] + n_t * [2*i + 1]
            self.worlds += trueworld + falseworld
            self.w_batch += n_r1 * [2*i] + n_r2 * [2*i + 1]
            self.labels += [[1], [0]]

    def generate_overfit(self, N, n):
        """
        Generates a dataset of size 2 * N with n positive and n negative
        (n << N) samples. Used for overfitting a model to check model capacity.
        """
        print('generating dataset of %s examples :' % (2 * N))
        mem = []
        for i in range(n):
            try:
                query, trueworld, falseworld = self.gen_one()
            except Resample:
                # We resample the config once
                # If there is a sampling timeout here, we let it pass
                query, trueworld, falseworld = self.gen_one()
            mem.append((query, trueworld, falseworld))
        for i in tqdm(range(N)):
            query, trueworld, falseworld = random.choice(mem)
            n_t = len(query)
            n_r1 = len(trueworld)
            n_r2 = len(falseworld)
            self.queries += 2 * query
            self.q_batch += n_t * [2*i] + n_t * [2*i + 1]
            self.worlds += trueworld + falseworld
            self.w_batch += n_r1 * [2*i] + n_r2 * [2*i + 1]
            self.labels += [[1], [0]]

class NumberGen(Gen):

    def __init__(self, env=None, n_d=None):
        """
        Initialize the Number class generator.

        blabla

        This concrete class defines the generation functions.
        """
        super(NumberGen, self).__init__(env, n_d)

        self.max_n = 10
        self.color_sigma = 0.05 # standard deviation for the color, test this

    def gen_one(self):
        """
        Generates one example.

        For now we consider only one object in the query, we'll see later for 
        greater number of objects.

        To generate objects that are 'the same', we sample their color from a 
        3-dimensional Gaussian centered on the color of the query object, and 
        with small standard deviation.
        """
        try:
            self.env.reset()
            # sample query object
            self.env.add_random_object()
            obj = self.env.objects[0]
            color = obj.color
            idx = obj.shape_index
            # sample number
            n = np.random.randint(0, self.max_n + 1)
            query = self.env.to_state_list(norm=True)
            # fill world with similar objects, as the number requires
            self.env.reset()
            for _ in range(n):
                sampled_color = np.random.normal(
                    color / 255,
                    self.color_sigma)
                sampled_color = (sampled_color * 255).astype(int)
                self.env.add_random_object(color=sampled_color, shape=idx)
            if self.n_d is None:
                n_d = np.random.randint(*self.range_d)
            else:
                n_d = self.n_d
            # fill with other stuff
            self.random_config(n_d)
            world = self.env.to_state_list(norm=True)
            return query, world, n
        except SamplingTimeout:
            print('Sampling timed out, {} and {} objects'.format(n_t, n_d))
            raise Resample('Resample configuration')

    def generate(self, N):
        """
        Generate a dataset for the task Number.
        """
        for i in range(N):
            try:
                query, world, n = gen_one()
            except Resample:
                query, world, n = gen_one()
            n_q = len(query)
            n_w = len(world)
            self.queries += query
            self.q_batch += n_q * [i]
            self.worlds += world
            self.w_batch += n_w * [i]
            self.labels += [[n]]
