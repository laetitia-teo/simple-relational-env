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
import cv2

import torch

from glob import glob
from tqdm import tqdm

from env import Env
from dataset import Dataset, PartsDataset
from utils import to_file, from_file

### quick testing ###

from torch.utils.data import DataLoader
from dataset import collate_fn

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

        # has metadata
        self.has_metadata = False

        # data
        self.reset()

    def reset(self):
        self.targets = []
        self.t_batch = []
        self.refs = []
        self.r_batch = []
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
        for idx in tqdm(range(len(self.labels))):
            self.t_idx.append((t_batch == idx).nonzero(as_tuple=True)[0])
            self.r_idx.append((r_batch == idx).nonzero(as_tuple=True)[0])

    def gen_one(self):
        raise NotImplementedError()

    def generate(self, N):
        raise NotImplementedError()

    def generate_overfit(self, N, n):
        raise NotImplementedError()

    # def add_one(self, targets, refs, labels):
    #     self.targets += targets
    #     self.refs += refs
    #     self.labels += labels
    def cut(self, n):
        """
        Cuts the data to the n first examples.
        """
        if not self.t_idx:
            self.compute_access_indices()
        t_stop_index = self.t_idx[n-1][-1] + 1
        if self.refs:
            r_stop_index = self.r_idx[n-1][-1] + 1
        self.targets = self.targets[:t_stop_index]
        self.t_batch = self.t_batch[:t_stop_index]
        if self.refs:
            self.refs = self.refs[:r_stop_index]
            self.r_batch = self.r_batch[:r_stop_index]
        self.labels = self.labels[:n]
        self.t_idx = self.t_idx[:n]
        self.r_idx = self.r_idx[:n]

    def multiply(self, n):
        """
        Duplicates the existing examples n times.
        """
        # if not self.t_idx:
        #     self.compute_access_indices()
        self.targets *= n
        self.refs *= n
        self.labels *= n
        self.t_batch
        self.r_batch *= n
        mem = list(self.t_batch)
        for i in range(n - 1):
            mem += [elem + mem[-1] + 1 for elem in self.t_batch]
        self.t_batch = mem
        mem = list(self.r_batch)
        for i in range(n - 1):
            mem += [elem + mem[-1] + 1 for elem in self.r_batch]
        self.r_batch = mem
        self.compute_access_indices()

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
            for i in label:
                f.write(str(i) + ' ')
            f.write('\n')

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

    def write_metadata(self, path):
        pass

    def read_targets(self, lineit):
        """
        Takes in an iterator of the lines read.
        Reads the targets and t_batch from lines, returns targets, t_batch
        and stopping index.
        """
        targets = []
        t_batch = []
        line = next(lineit)
        while 'refs' not in line:
            if 'targets' in line:
                pass # first line
            else:
                linelist = line.split(' ')
                t_batch.append(int(linelist[0]))
                targets.append(np.array(linelist[1:-1], dtype=float))
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
        while 'labels' not in line:
            linelist = line.split(' ')
            r_batch.append(int(linelist[0]))
            refs.append(np.array(linelist[1:-1], dtype=float))
            line = next(lineit)
        return refs, r_batch

    def read_labels(self, lineit):
        """
        Reads the label from an iterator on the file lines.
        """
        labels = []
        try:
            line = next(lineit)
            while 't_idx' not in line:
                labels.append([float(line)])
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
            to_array = np.array(linelist[:-1], dtype=int)
            t_idx.append(torch.tensor(to_array, dtype=torch.long))
            line = next(lineit)
        return t_idx

    def read_r_idx(self, lineit):
        r_idx = []
        try:
            line = next(lineit)
            while 'some_other_stuff' not in line:
                linelist = line.split(' ')
                # print(linelist)
                to_array = np.array(linelist[:-1], dtype=int)
                r_idx.append(torch.tensor(to_array, dtype=torch.long))
                line = next(lineit)
        except StopIteration:
            pass
        return r_idx

    def read_vectors(self, lineit, start_token, stop_token):
        """
        Takes in an iterator of the lines read.
        Reads the vectors from lines and returns a list of the read vectors.
        """
        try:
            vectors = []
            line = next(lineit)
            while stop_token not in line:
                if start_token in line:
                    pass # first line
                else:
                    linelist = line.split(' ')
                    vectors.append(np.array(linelist[:-1], dtype=float))
                line = next(lineit)
        except StopIteration:
            pass
        return vectors

    def read_scalars(self, lineit, start_token, stop_token):
        """
        Reads the scalars (one per line) and returns them as a list.
        """
        try:
            scalars = []
            line = next(lineit)
            while stop_token not in line:
                linelist = line.split(' ')
                scalars.append(float(linelist[0]))
                line = next(lineit)
        except StopIteration:
            pass
        return scalars

    def read_metadata(self, path):
        pass

    def save(self, path, write_indices=True):
        """
        Saves the dataset as a file. 
        """
        if write_indices and not self.t_idx:
            print('computing access indices...')
            self.compute_access_indices()
            print('done')
        with open(path, 'w') as f:
            self.write_targets(f)
            self.write_refs(f)
            self.write_labels(f)
            if write_indices:
                self.write_t_idx(f)
                self.write_r_idx(f)
        self.write_metadata(path)

    def load(self, path, read_indices=True, replace=True):
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
            if read_indices:
                try:
                    t_idx = self.read_t_idx(lineit)
                    r_idx = self.read_r_idx(lineit)
                except StopIteration:
                    print('No indices were found but the generator was '\
                        + 'asked to read them. Index lists are empty.')
                    t_idx = []
                    r_idx = []
        self.read_metadata(path)
        # stores the data
        if replace:
            self.targets = targets
            self.t_batch = t_batch
            self.refs = refs
            self.r_batch = r_batch
            self.labels = labels
            if read_indices:
                self.t_idx = t_idx
                self.r_idx = r_idx
        else:
            # this doesn't work as is, need to update indices
            self.targets += targets
            self.t_batch += t_batch
            self.refs += refs
            self.r_batch += r_batch
            self.labels += labels

    def to_dataset(self, n=None, label_type='long', device=torch.device('cpu')):
        """
        Creates a PartsDataset from the generated data and returns it.

        Arguments :
            - n (int) : allows to contol the dataset size for export.
        """
        if self.t_idx == []:
            indices = None
        else:
            indices = (self.t_idx, self.r_idx)
        ds = PartsDataset(self.targets[:n],
                          self.t_batch[:n],
                          self.refs[:n],
                          self.r_batch[:n],
                          self.labels[:n],
                          indices,
                          self.task_type,
                          self.label_type,
                          device=device)
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
        self.task = 'parts_task'
        self.task_type = 'scene'
        self.label_type='long'

    def gen_one(self):
        """
        Generates one pair of true-false examples associated with a given
        query.

        The targets are perturbed (similarity + small noise) before being
        completed with distractors.
        """
        # Note : we could generate 4 by 4 with this code, by crossing targets
        # and refs
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
            - targets (list of vectors): list of all the query objets;
            - t_batch (list of ints): list of indices linking the query
                objects to their corresponding scene index;
            - refs (list of object vectors): list of all the reference
                objects;
            - r_batch (list of ints): list of indices linking the reference
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
            self.targets += 2 * query
            self.t_batch += n_t * [2*i] + n_t * [2*i + 1]
            self.refs += trueworld + falseworld
            self.r_batch += n_r1 * [2*i] + n_r2 * [2*i + 1]
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
            self.targets += 2 * query
            self.t_batch += n_t * [2*i] + n_t * [2*i + 1]
            self.refs += trueworld + falseworld
            self.r_batch += n_r1 * [2*i] + n_r2 * [2*i + 1]
            self.labels += [[1], [0]]

class PartsGenv2(Gen):
    """
    A different version of the Parts Generator, to sample configurations that 
    only differ slightly, in terms of configuration of the objects that
    interests us. This is because in the previous generator, we only present
    to the model examples of samples that differ either completely, either that
    have the same configuration, up to similarity. This may make the task
    harder to learn, and the model more brittle when it learns.

    In this generator, we first sample the query, then for positive examples
    we translate/shift a bit (maybe also jitter object positions/angles/colors
    ?), and for negative examples we move one to all objects (number of objects
    moved sampled uniformly).

    Contrary to the previous generator, we generate one example per step.
    """
    def __init__(self, env=None, n_d=None):
        """
        Initialize the Parts task generator, version 2.
        """
        super(PartsGenv2, self).__init__(env, n_d)
        self.task = 'parts_task'
        self.task_type = 'scene'
        self.label_type='long'

    def gen_one(self):
        """
        Generates one training example.
        """
        try:
            self.env.reset()
            label = np.random.randint(0, 2)
            n_t = np.random.randint(*self.range_t)
            if self.n_d is None:
                n_d = np.random.randint(*self.range_d)
            else:
                n_d = self.n_d
            self.env.random_config(n_t)
            query = self.env.to_state_list(norm=True)
            if label:
                self.env.random_transformation()
                self.env.random_config(n_d)
                world = self.env.to_state_list(norm=True)
            else:
                n_p = np.random.randint(1, n_t + 1) # number of perturbed objects
                self.env.perturb_objects(n_p)
                self.env.random_transformation()
                self.env.random_config(n_d)
                world = self.env.to_state_list(norm=True)
            return query, world, label
        except SamplingTimeout:
            print('Sampling timed out, {} and {} objects'.format(n_t, n_d))
            raise Resample('Resample configuration')

    def generate(self, N):
        """
        Generates a dataset of N positive and N negative examples.

        Arguments :
            - N (int) : half of the dataset length

        Generates:
            - targets (list of vectors): list of all the query objets;
            - t_batch (list of ints): list of indices linking the query
                objects to their corresponding scene index;
            - refs (list of object vectors): list of all the reference
                objects;
            - r_batch (list of ints): list of indices linking the reference
                objects to their corresponding scene index;
            - labels (list of ints): list of scene labels.
        """
        print('generating dataset of %s examples :' % N)
        for i in tqdm(range(N)):
            try:
                query, world, label = self.gen_one()
            except Resample:
                # We resample the config once
                # If there is a sampling timeout here, we let it pass
                query, world, label = self.gen_one()
            n_t = len(query)
            n_r = len(world)
            self.targets += query
            self.t_batch += n_t * [i]
            self.refs += world
            self.r_batch += n_r * [i]
            self.labels += [[label]]
            
class SimilarityObjectsGen(Gen):
    """
    A generator for the Similarity-Object task.

    Similar to the generator for Similarity-Boolean task, except the labels are
    not 1 and 0 for each scene, but fir each object.
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
        super(SimilarityObjectsGen, self).__init__(env, n_d)
        self.task = 'similarity_object'
        self.task_type = 'object'
        self.label_type = 'long'

    def gen_one(self):
        """
        Generates one pair of true-false examples associated with a given
        query.

        The targets are perturbed (similarity + small noise) before being
        completed with distractors.
        """
        # Note : we could generate 4 by 4 with this code, by crossing targets
        # and refs
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
            self.env.shuffle_objects() # useless
            self.env.random_transformation()
            self.env.random_config(n_d) # add random objects
            trueworld = self.env.to_state_list(norm=True)
            truelabel = np.zeros(len(trueworld), dtype=int)
            truelabel[:len(query)] = 1 # distractors are appended
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
            falselabel = np.zeros(len(falseworld), dtype=int)
            query = query
            return query, trueworld, falseworld, truelabel, falselabel
        except SamplingTimeout:
            print('Sampling timed out, {} and {} objects'.format(n_t, n_d))
            raise Resample('Resample configuration')

    def generate(self, N):
        """
        Generates a dataset of N positive and N negative examples.

        Arguments :
            - N (int) : half of the dataset length

        Generates:
            - targets (list of vectors): list of all the query objets;
            - t_batch (list of ints): list of indices linking the query
                objects to their corresponding scene index;
            - refs (list of object vectors): list of all the reference
                objects;
            - r_batch (list of ints): list of indices linking the reference
                objects to their corresponding scene index;
            - labels (list of ints): list of scene labels.
        """
        print('generating dataset of %s examples :' % (2 * N))
        for i in tqdm(range(N)):
            try:
                query, trueworld, falseworld, truelabel, falselabel = \
                    self.gen_one()
            except Resample:
                # We resample the config once
                # If there is a sampling timeout here, we let it pass
                query, trueworld, falseworld, truelabel, falselabel = \
                    self.gen_one()
            n_t = len(query)
            n_r1 = len(trueworld)
            n_r2 = len(falseworld)
            self.targets += 2 * query
            self.t_batch += n_t * [2*i] + n_t * [2*i + 1]
            self.refs += trueworld + falseworld
            self.r_batch += n_r1 * [2*i] + n_r2 * [2*i + 1]
            self.labels += [[1]] * len(query)
            self.labels += [[0]] * (len(trueworld) - len(query))
            self.labels += [[0]] * len(falseworld)

class CountGen(Gen):
    """
    A generator for the conting task.
    """
    def __init__(self, env=None, n_d=None):
        """
        Initialize the Number class generator.

        blabla

        This concrete class defines the generation functions.
        """
        super(CountGen, self).__init__(env, n_d)

        self.task = 'count'
        self.task_type = 'scene'
        self.max_n = 10
        self.color_sigma = 0.05 # standard deviation for the color, test this
        self.label_type = 'float'

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
            n = np.random.randint(1, self.max_n + 1)
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
            self.env.random_config(n_d)
            world = self.env.to_state_list(norm=True)
            return query, world, n
        except SamplingTimeout:
            print('Sampling timed out, {} and {} objects'.format(n_t, n_d))
            raise Resample('Resample configuration')

    def generate(self, N):
        """
        Generate a dataset for the task Number.
        """
        for i in tqdm(range(N)):
            try:
                query, world, n = self.gen_one()
            except Resample:
                query, world, n = self.gen_one()
            n_q = len(query)
            n_w = len(world)
            self.targets += query
            self.t_batch += n_q * [i]
            self.refs += world
            self.r_batch += n_w * [i]
            self.labels += [[n]]

class SelectGen(Gen):
    """
    A generator for the object selection task. This is very similar to the
    object counting task, except the prediction is done on objects : 1 for 
    objects to select, 0 for distractors.
    """
    def __init__(self, env=None, n_d=None):
        super(SelectGen, self).__init__(env, n_d)

        self.task = 'select'
        self.task_type = 'object'
        self.max_n = 5
        self.color_sigma = 0.05 # standard deviation for the color, test this
        self.label_type = 'long'

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
            self.env.random_config(n_d)
            world = self.env.to_state_list(norm=True)
            label = [[1]] * n + [[0]] * n_d
            return query, world, label
        except SamplingTimeout:
            print('Sampling timed out, {} and {} objects'.format(n_t, n_d))
            raise Resample('Resample configuration')

    def generate(self, N):
        """
        Generate a dataset for the task Select.
        """
        for i in tqdm(range(N)):
            try:
                query, world, label = self.gen_one()
            except Resample:
                query, world, label = self.gen_one()
            n_q = len(query)
            n_w = len(world)
            self.targets += query
            self.t_batch += n_q * [i]
            self.refs += world
            self.r_batch += n_w * [i]
            self.labels += label

class AbstractRelationsGen(Gen):
    """
    Generator class for the Abstract Relations task.
    The queries are pairs of objects linked by a relation of the type
    'is left of'.

    There are four possible relations, one for every cardinal direction in 2d
    space.
    """
    def __init__(self, env=None, n_d=None):
        super(AbstractRelationsGen, self).__init__()
        self.task = 'abstract_relations'
        self.task_type = 'scene'
        self.label_type = 'long'

    def gen_one(self):
        """
        Generates one example.
        """
        try:
            self.env.reset()
            # sample query abstract relation
            self.env.add_random_object()
            rel = np.random.randint(0, 4)
            flipped = self.env.add_random_object_relation(rel)
            if flipped:
                # if we sampled the reference object too close to the wall and
                # had to flip the relation, change its number
                rel = rel + 1 * (1 - rel % 2) - 1 * (rel % 2)
            query = self.env.to_state_list(norm=True)
            # keep the objects in memory so as to remember their shape/color
            obj1 = self.env.objects[0]
            obj2 = self.env.objects[1]
            self.env.reset()
            color1 = obj1.color
            color2 = obj2.color
            idx1 = obj1.shape_index
            idx2 = obj2.shape_index
            label = np.random.randint(0, 2) # same or different
            # maybe slightly change the colors
            # sample the world abstract relation
            self.env.add_random_object(color=color1, shape=idx1)
            if label:
                rel2 = rel
                flipped2 = self.env.add_random_object_relation(
                    rel2,
                    color=color2,
                    shape=idx2)
            else:
                rel2 = (rel + np.random.randint(1, 4)) % 4 # different rel
                flipped2 = self.env.add_random_object_relation(
                    rel2,
                    color=color2,
                    shape=idx2)
            if flipped2:
                rel2 = rel2 + 1 * (1 - rel2 % 2) - 1 * (rel2 % 2)
            if self.n_d is None:
                n_d = np.random.randint(*self.range_d)
            else:
                n_d = self.n_d
            # fill with other stuff
            self.env.random_config(n_d)
            world = self.env.to_state_list(norm=True)
            label = int(rel == rel2)
            return query, world, label
        except SamplingTimeout:
            print('Sampling timed out, {} and {} objects'.format(n_t, n_d))
            raise Resample('Resample configuration')

    def generate(self, N):
        for i in tqdm(range(N)):
            try:
                query, world, label = self.gen_one()
            except Resample:
                # shouldn't happen really often here, if n_d is reasonable
                query, world, label = self.gen_one()
            n_q = len(query)
            n_w = len(world)
            self.targets += query
            self.t_batch += n_q * [i]
            self.refs += world
            self.r_batch += n_w * [i]
            self.labels += [[label]]

class SameConfigGen(Gen):
    """
    This generator generates a single configuration based on a reference
    configuration. To generate something that can be considered the same 
    configuration, we are allowed to perturb each element's attributes by a
    small amount, and we are allowed to apply translations and scalings to 
    the configurations.
    """
    def __init__(self, ref_state_list=None, n=None):
        super(SameConfigGen, self).__init__()
        self.task = 'same_config'
        self.task_type = 'scene'
        self.label_type = 'long'
        if ref_state_list:
            self.ref_state_list = ref_state_list
        else:
            if n is None:
                n = 5
            self.env.random_config(n)
            self.ref_state_list = self.env.to_state_list(norm=True)
            self.env.reset()

        # metadata
        self.has_metadata = True
        self.N = 0 # number of generated samples
        self.translation_vectors = []
        self.scalings = []
        self.rotation_angles = []
        self.n_objects = len(self.ref_state_list)
        self.eps = 0.01 # amplitude factor of the perturbations
        self.small_perturbations = []
        self.perturbations = []
        # ranges we exclude from generation
        self.t_ex_range = (np.array([0., 0.]), np.array([1., 1.])) # example
        self.s_ex_range = (1., 1.2)
        self.r_ex_range = (0., np.pi)

    def equal_cut(self, n):
        """
        Similar to the cut function, but ensures n positive and n negative
        samples are selected.
        """
        pass

    def write_metadata(self, path):
        """
        Writes the metadata in the file at path, with the scmd (same_config
        metadata) extension.
        """
        with open(path + '.scmd', 'w') as f:
            # write number of generated samples
            f.write('N\n')
            f.write(str(self.N) + '\n')
            # write reference state list
            f.write('ref_state_list\n')
            for obj in self.ref_state_list:
                for num in obj:
                    f.write(str(num) + ' ')
                f.write('\n')
            # write epsilon
            f.write('eps\n')
            f.write(str(self.eps) + '\n')
            # write translation vectors
            f.write('tvecs\n')
            for tvec in self.translation_vectors:
                for i in tvec:
                    f.write(str(i) + ' ')
                f.write('\n')
            # write scalings
            f.write('scalings\n')
            for scal in self.scalings:
                f.write(str(scal) + '\n')
            # write rotation angles
            f.write('phis\n')
            for phi in self.rotation_angles:
                f.write(str(phi) + '\n')
            # write small perturbations
            f.write('spert\n')
            for spert in self.small_perturbations:
                for i in spert:
                    f.write(str(i) + ' ')
                f.write('\n')
            # write perturbations
            f.write('pert\n')
            for pert in self.perturbations:
                for i in pert:
                    f.write(str(i) + ' ')
                f.write('\n')
            # write the ranges we exclude
            pass # TODO : not implemented yet

    def read_metadata(self, path):
        """
        Reads the metadata witten in the file specified at path with the scmd 
        extension.
        """
        with open(path + '.scmd', 'r') as f:
            lines = f.readlines()
            l = iter(lines)
            # read ref state list
            self.ref_state_list = self.read_vectors(l, 'ref_state_list', 'eps')
            # read epsilon
            self.eps = self.read_scalars(l, 'eps', 'tvecs')
            # read translation vectors
            self.translation_vectors = self.read_vectors(
                l, 'tvecs', 'scalings')
            # read scalings
            self.scalings = self.read_scalars(l, 'scalings', 'phis')
            # read rotation angles
            self.rotation_angles = self.read_scalars(l, 'phis', 'spert')
            # read perturbations
            self.small_perturbations = self.read_vectors(l, 'spert', 'pert')
            # read small perturbations
            self.perturbations = self.read_vectors(l, 'pert', 'end')

    def gen_one(self):
        """
        Generates one example, by perturbing a bit each object of the reference
        configuration.

        Also records metadata for the generation.
        """
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        label = np.random.randint(2) # positive or negative example
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            vec, scale, phi = self.env.random_transformation()
            pert = [np.zeros(2)] * len(self.ref_state_list)
        else:
            n_p = np.random.randint(len(self.env.objects))
            spert = self.env.small_perturb_objects(self.eps)
            pert = self.env.perturb_objects(n_p)
            vec, scale, phi = self.env.random_transformation()
        state = self.env.to_state_list(norm=True)
        return state, label, vec, scale, phi, spert, pert

    def alternative_gen_one(self):
        """
        Alternative, legacy way to generate dataset.
        """
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        label = np.random.randint(2) # positive or negative example
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            vec, scale, phi = self.env.random_transformation()
            pert = [np.zeros(2)] * len(self.ref_state_list)
        else:
            n_p = np.random.randint(len(self.env.objects))
            spert = self.env.small_perturb_objects(self.eps)
            pert = [np.zeros(2)] * len(self.ref_state_list)
            self.env.random_mix()
            # pert = self.env.perturb_objects(n_p)
            vec, scale, phi = self.env.random_transformation()
        state = self.env.to_state_list(norm=True)
        return state, label, vec, scale, phi, spert, pert

    # generate controlled test examples
    # custom positive examples
    def gen_pure_translation(self, vec=None):
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        if vec is None:
            vec = self.env.random_translation_vector()
        spert = self.env.small_perturb_objects(self.eps)
        scale = 0
        phi = 0
        pert = [np.zeros(2)] * len(self.ref_state_list)
        self.env.translate(vec)

    def gen_pure_scaling(self, scale=None):
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        if scale is None:
            _, scale = self.env.random_scaling()
        spert = self.env.small_perturb_objects(self.eps)
        vec = np.zeros(2)
        phi = 0
        pert = [np.zeros(2)] * len(self.ref_state_list)
        self.env.translate(scale)
        return state, label, vec, scale, phi, spert, pert

    def gen_pure_rotation(self, phi=None):
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        if phi is None:
            phi = self.env.random_rotation()
        spert = self.env.small_perturb_objects(self.eps)
        scale = 0
        vec = np.zeros(2)
        self.env.translate(vec)
        return state, label, vec, scale, phi, spert, pert

    # contolled epsilon
    def gen_controlled_eps(self, eps):
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        spert = self.env.small_perturb_objects(self.eps)
        phi = 0
        scale = 0
        vec = np.zeros(2)
        pert = [np.zeros(2)] * len(self.ref_state_list)
        return state, label, vec, scale, phi, spert, pert

    def generate_one(self, gen_fn, i):
        """
        Wrapper for the different generation functions.
        Generates a config according to the provided generation function,
        and records the generation trace in all the good member variables.
        """
        state, label, vec, scale, phi, spert, pert = gen_fn()
        n_s = len(state)
        self.targets += state
        self.t_batch += n_s * [i]
        self.refs += []
        self.r_batch += []
        self.labels += [[label]]
        # metadata
        self.translation_vectors += [vec]
        self.scalings += [scale]
        self.rotation_angles += [phi]
        self.small_perturbations += spert
        self.perturbations += pert

    def generate(self, N):
        I = len(self.labels)
        for i in tqdm(range(N)):
            self.generate_one(self.gen_one, i + I)
        self.N += N

    def generate_alternative(self, N):
        I = len(self.labels)
        for i in tqdm(range(N)):
            self.generate_one(self.alternative_gen_one, i + I)
        self.N += N

    # utils
    def render(self, path, mode='fixed'):
        """
        Renders the generated configurations at the desired path.
        For testing purposes, do not try with too big a dataset (not optimized)

        pqth : string to the directory of save.
        """
        self.compute_access_indices()
        vecs = np.array(self.targets)
        for i in range(self.t_batch[-1]):
            print('batch %s' % i)
            print('label %s' % self.labels[i])
            self.env.reset()
            state_list = list(vecs[self.t_idx[i].numpy()])
            self.env.from_state_list(state_list, norm=True)
            img = self.env.render(mode=mode)
            cv2.imwrite(op.join(path, 'img_%s.jpg' % i), img)
