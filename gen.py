"""
This file defines the dataset generators.
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

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import collate_fn

N_SH = 3
DTYPE = torch.float32

class Gen():
    """
    Base generator class.
    """
    def __init__(self, env=None, n_d=None):
        
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
        # reset env
        self.env.reset()
        # reset data
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
        pass

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
        # self.t_batch *= n
        # self.r_batch *= n
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
        with open(path, 'w') as f:
            self.write_targets(f)
            self.write_refs(f)
            self.write_labels(f)
            if write_indices:
                self.write_t_idx(f)
                self.write_r_idx(f)
        # self.write_metadata(path)

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
        # self.read_metadata(path)
        # stores the data
        if replace:
            self.targets = targets
            self.t_batch = t_batch
            self.refs = refs
            self.r_batch = r_batch
            self.labels = labels
            # if read_indices:
            #     self.t_idx = t_idx
            #     self.r_idx = r_idx
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
        ds = PartsDataset(self.targets,
                          self.t_batch,
                          self.refs,
                          self.r_batch,
                          self.labels,
                          self.task_type,
                          device=device)
        return ds

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
        self.t_ex_range = None # example
        self.s_ex_range = None
        self.r_ex_range = None

    def reset(self):
        super(SameConfigGen, self).reset()
        self.translation_vectors = []
        self.rotation_angles = []
        self.scalings = []
        self.small_perturbations = []
        self.perturbations = []
        self.t_ex_range = None
        self.s_ex_range = None
        self.r_ex_range = None

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
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation()
            pert = [np.zeros(2)] * len(self.ref_state_list)
        else:
            n_p = np.random.randint(len(self.env.objects))
            spert = self.env.small_perturb_objects(self.eps)
            pert = self.env.perturb_objects(n_p)
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation()
        state = self.env.to_state_list(norm=True)
        return state, label, vec, scale, phi, spert, pert

    def alternative_gen_one(self):
        """
        Negative examples are complete re_shufflings of the reference config.
        """
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        label = np.random.randint(2) # positive or negative example
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
            pert = [np.zeros(2)] * len(self.ref_state_list)
        else:
            n_p = np.random.randint(len(self.env.objects))
            spert = self.env.small_perturb_objects(self.eps)
            pert = [np.zeros(2)] * len(self.ref_state_list)
            self.env.random_mix()
            self.env.shuffle_objects()
            # pert = self.env.perturb_objects(n_p)
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
        state = self.env.to_state_list(norm=True)
        return state, label, vec, scale, phi, spert, pert

    def abstract_gen_one(self):
        """
        Generates one example where only the spatial positions of the objects
        matter, the other features are re-drawn randomly in the positive and
        negative samples.
        """
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        label = np.random.randint(2)
        # positive or negative example
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            self.env.non_spatial_perturb()
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
            pert = [np.zeros(2)] * len(self.ref_state_list)
        else:
            n_p = np.random.randint(len(self.env.objects))
            spert = self.env.small_perturb_objects(self.eps)
            pert = [np.zeros(2)] * len(self.ref_state_list)
            self.env.random_mix()
            self.env.non_spatial_perturb()
            self.env.shuffle_objects()
            # pert = self.env.perturb_objects(n_p)
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
        state = self.env.to_state_list(norm=True)
        return state, label, vec, scale, phi, spert, pert

    def abstract_gen_one_distractors(self, ndmin=1, ndmax=3):
        """
        Generates one example where only the spatial positions of the objects
        matter, the other features are re-drawn randomly in the positive and
        negative samples.
        """
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        label = np.random.randint(2)
        # positive or negative example
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            self.env.non_spatial_perturb()

            nd = np.random.randint(ndmin, ndmax)
            self.env.random_config(nd)

            self.env.shuffle_objects()

            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
            pert = [np.zeros(2)] * len(self.ref_state_list)
        else:
            n_p = np.random.randint(len(self.env.objects))
            spert = self.env.small_perturb_objects(self.eps)
            pert = [np.zeros(2)] * len(self.ref_state_list)
            self.env.random_mix()
            self.env.non_spatial_perturb()

            nd = np.random.randint(ndmin, ndmax)
            self.env.random_config(nd)

            self.env.shuffle_objects()

            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
        state = self.env.to_state_list(norm=True)

        return state, label, vec, scale, phi, spert, pert

    def controlled_gen_one(self,
                           scale=None,
                           tvec=None,
                           phi=None,
                           positive_only=True):
        """
        Allows to have a semi-random transformation, where some of the
        transformation parameters are randomly sampled and others are fixed
        in advance.
        """
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        if positive_only:
            label = 1
        else:
            label = np.random.randint(2) # positive or negative example
        # print('passed phi %s' % phi)
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            if tvec is None:
                tvec = self.env.random_translation_vector_cartesian_v2(
                    ex_range=self.t_ex_range)
            self.env.translate(tvec)
            if scale is None:
                center, scale = self.env.random_scaling(
                    ex_range=self.s_ex_range)
            else:
                center = None
            self.env.scale(scale, center=center)
            if phi is None:
                center, phi = self.env.random_rotation(
                    ex_range=self.r_ex_range)
            else:
                center = None
            self.env.rotate(phi, center=center)
            pert = [np.zeros(2)] * len(self.ref_state_list)
        else:
            n_p = np.random.randint(len(self.env.objects))
            spert = self.env.small_perturb_objects(self.eps)
            pert = [np.zeros(2)] * len(self.ref_state_list)
            self.env.random_mix()
            # pert = self.env.perturb_objects(n_p)
            tvec, scale, phi = self.env.random_transformation()
        state = self.env.to_state_list(norm=True)
        # print('result phi %s' % phi)
        return state, label, tvec, scale, phi, spert, pert

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
        return state, label, vec, scale, phi, spert, pert

    def gen_pure_scaling(self, scale=None):
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        if scale is None:
            _, scale = self.env.random_scaling()
        spert = self.env.small_perturb_objects(self.eps)
        vec = np.zeros(2)
        phi = 0
        pert = [np.zeros(2)] * len(self.ref_state_list)
        self.env.scale(scale)
        return state, label, vec, scale, phi, spert, pert

    def gen_pure_rotation(self, phi=None):
        self.env.reset()
        self.env.from_state_list(self.ref_state_list, norm=True)
        if phi is None:
            _, phi = self.env.random_rotation()
        spert = self.env.small_perturb_objects(self.eps)
        scale = 0
        vec = np.zeros(2)
        pert = [np.zeros(2)] * len(self.ref_state_list)
        self.env.rotate(phi)
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

    def generate_one(self, gen_fn, i, *args, **kwargs):
        """
        Wrapper for the different generation functions.
        Generates a config according to the provided generation function,
        and records the generation trace in all the good member variables.
        """
        state, label, vec, scale, phi, spert, pert = gen_fn(*args, **kwargs)
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
        # self.perturbations += pert

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

    def generate_abstract(self, N):
        I = len(self.labels)
        for i in tqdm(range(N)):
            self.generate_one(self.abstract_gen_one, i + I)
        self.N += N

    def generate_abstract_distractors(self, N):
        I = len(self.labels)
        for i in tqdm(range(N)):
            self.generate_one(self.abstract_gen_one_distractors, i + I)
        self.N += N

    # def generate_generalization(self, N, n, ex_range, mod, b_size):
    #     """
    #     Based on the mod ('s' for scalings, 't' for translations, 'r' for
    #     rotations, on the exclusion range, on the number n of elements in the
    #     test grid, on the batch size b_size), generates the test and
    #     """

    def generate_grid(self, n, b_size, mod='s'):
        I = len(self.labels)
        i = 0
        if mod == 's':
            minscale = 0.5
            maxscale = 2.0
            scales = np.arange(n).astype(float) / n
            scales = (1 - scales) * minscale + scales * maxscale
            for scale in tqdm(scales):
                kwargs = {'scale': scale}
                for _ in range(b_size):
                    self.generate_one(
                        self.controlled_gen_one,
                        i = I + i,
                        **kwargs)
                    i += 1
        if mod == 'r':
            minphi = 0
            maxphi = 2 * np.pi
            phis = np.arange(n).astype(float) / n
            phis = (1 - phis) * minphi + phis * maxphi
            for phi in tqdm(phis):
                kwargs = {'phi': phi}
                for _ in range(b_size):
                    self.generate_one(
                        self.controlled_gen_one,
                        i = I + i,
                        **kwargs)
                    i += 1
        if mod == 't':
            minvec = - self.env.envsize
            maxvec = self.env.envsize
            a = np.arange(n).astype(float) / n
            a = (1 - a) * minvec + a * maxvec
            m = np.array(np.meshgrid(a, a))
            for k in tqdm(range(m.shape[2])):
                for l in range(m.shape[1]):
                    kwargs = {'tvec': m[:, l, k]}
                    for _ in range(b_size):
                        self.generate_one(
                            self.controlled_gen_one,
                            i = I + i,
                            **kwargs)
                        i += 1

    # utils
    def render(self, path, mode='fixed'):
        """
        Renders the generated configurations at the desired path.
        For testing purposes, do not try with too big a dataset (not optimized)

        path : string to the directory of save.
        """
        self.compute_access_indices()
        vecs = np.array(self.targets)
        for i in tqdm(range(self.t_batch[-1])):
            # print('batch %s' % i)
            # print('label %s' % self.labels[i])
            self.env.reset()
            state_list = list(vecs[self.t_idx[i].numpy()])
            self.env.from_state_list(state_list, norm=True)
            img = self.env.render(mode=mode, show=False)
            cv2.imwrite(op.join(path, 'img_%s.jpg' % i), img)

    def render_ref_state(self, show=True):
        # buggy ref state list
        ref_state_list = [a for a in self.ref_state_list if a.any()]
        self.env.reset()
        self.env.from_state_list(ref_state_list, norm=True)
        self.env.scale(amount=0.7)
        img = self.env.render(show=show, mode='envsize')
        img = np.flip(img, -1)
        self.env.reset()
        return img

class CompareConfigGen(Gen):
    """
    Generator for the 'compare_config' setup.

    It has the same generating procedures as the SameConfigGen but generates
    pairs of configs instead of single configs.
    """
    def __init__(self, n_min=5, n_max=5):
        super(CompareConfigGen, self).__init__()
        self.task = 'same_config'
        self.task_type = 'scene'
        self.label_type = 'long'

        # metadata
        self.has_metadata = False
        self.N = 0 # number of generated samples
        self.translation_vectors = []
        self.scalings = []
        self.rotation_angles = []
        self.n_objects_min = n_min
        self.n_objects_max = n_max + 1
        self.eps = 0.01 # amplitude factor of the perturbations
        self.small_perturbations = []
        self.perturbations = []
        # ranges we exclude from generation
        self.t_ex_range = None # example
        self.s_ex_range = None
        self.r_ex_range = None

    def reset(self):
        super(CompareConfigGen, self).reset()
        self.translation_vectors = []
        self.rotations = []
        self.scalings = []
        self.small_perturbations = []
        self.perturbations = []
        self.t_ex_range = None
        self.s_ex_range = None
        self.r_ex_range = None

    def write_metadata(self, path):
        pass

    def read_metadata(self, path):
        pass

    def gen_one(self, n_max=1):
        # n_max is max number of perturbed objects
        self.env.reset()
        nobj = np.random.randint(self.n_objects_min, self.n_objects_max)
        self.env.random_config(nobj)
        ref = self.env.to_state_list(norm=True)
        label = np.random.randint(2)
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
            pert = [np.zeros(2)] * len(ref)
        else:
            spert = self.env.small_perturb_objects(self.eps)
            # number of objects to perturb
            n = np.random.randint(1, n_max + 1)
            self.env.perturb_objects(n)
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
            pert = [np.zeros(2)] * len(ref)
        state = self.env.to_state_list(norm=True)
        return state, ref, label, vec, scale, phi, spert, pert

    def alternative_gen_one(self):
        """
        Generates one example, by perturbing a bit each object of the reference
        configuration.

        Also records metadata for the generation.
        """
        # generate first example
        self.env.reset()
        nobj = np.random.randint(self.n_objects_min, self.n_objects_max)
        self.env.random_config(nobj)
        ref = self.env.to_state_list(norm=True)
        label = np.random.randint(2) # positive or negative example
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
            pert = [np.zeros(2)] * len(ref)
        else:
            spert = self.env.small_perturb_objects(self.eps)
            pert = [np.zeros(2)] * len(ref)
            self.env.random_mix()
            self.env.shuffle_objects()
            # pert = self.env.perturb_objects(n_p)
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
        state = self.env.to_state_list(norm=True)
        return state, ref, label, vec, scale, phi, spert, pert

    def abstract_gen_one(self):
        """
        Generates one example where only the spatial positions of the objects
        matter, the other features are re-drawn randomly in the positive and
        negative samples.
        """
        # generate first example
        self.env.reset()
        nobj = np.random.randint(self.n_objects_min, self.n_objects_max)
        self.env.random_config(nobj)
        ref = self.env.to_state_list(norm=True)
        label = np.random.randint(2)
        # generate second example
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            self.env.non_spatial_perturb()
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
            pert = [np.zeros(2)] * len(ref)
        else:
            spert = self.env.small_perturb_objects(self.eps)
            pert = [np.zeros(2)] * len(ref)
            self.env.random_mix()
            self.env.non_spatial_perturb()
            self.env.shuffle_objects()
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
        state = self.env.to_state_list(norm=True)
        return state, ref, label, vec, scale, phi, spert, pert

    def abstract_gen_one_distractors(self, ndmin=1, ndmax=3):
        """
        Same as abstract_gen_one, but with additional distractors in the
        second state.
        """
        self.env.reset()
        nobj = np.random.randint(self.n_objects_min, self.n_objects_max)
        self.env.random_config(nobj)
        ref = self.env.to_state_list(norm=True)
        label = np.random.randint(2)
        # generate second example
        if label:
            spert = self.env.small_perturb_objects(self.eps)
            self.env.non_spatial_perturb()
            
            nd = np.random.randint(ndmin, ndmax+1)
            # add nd distractors
            self.env.random_config(nd)
            
            self.env.shuffle_objects()
            
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
            pert = [np.zeros(2)] * len(ref)
        else:
            spert = self.env.small_perturb_objects(self.eps)
            pert = [np.zeros(2)] * len(ref)
            self.env.random_mix()
            self.env.non_spatial_perturb()

            nd = np.random.randint(ndmin, ndmax+1)
            # add nd distractors
            self.env.random_config(nd)
            
            self.env.shuffle_objects()
            
            vec, scale, phi = self.env.random_transformation(
                rotations=True,
                s_ex_range=self.s_ex_range,
                t_ex_range=self.t_ex_range,
                r_ex_range=self.r_ex_range)
        state = self.env.to_state_list(norm=True)
        return state, ref, label, vec, scale, phi, spert, pert

    def generate_one(self, gen_fn, i, *args, **kwargs):
        """
        Wrapper for the different generation functions.
        Generates a config according to the provided generation function,
        and records the generation trace in all the good member variables.
        """
        state, ref, label, vec, scale, phi, spert, pert = gen_fn(
            *args, **kwargs)
        n_s = len(state)
        self.targets += state
        self.t_batch += n_s * [i]
        self.refs += ref
        self.r_batch += n_s * [i]
        self.labels += [[label]]
        # metadata
        self.translation_vectors += [vec]
        self.scalings += [scale]
        self.rotation_angles += [phi]
        self.small_perturbations += spert
        # self.perturbations += pert

    def generate(self, N, *args, **kwargs):
        I = len(self.labels)
        for i in tqdm(range(N)):
            self.generate_one(self.gen_one, i + I, *args, **kwargs)
        self.N += N

    def generate_alternative(self, N, *args, **kwargs):
        I = len(self.labels)
        for i in tqdm(range(N)):
            self.generate_one(self.alternative_gen_one, i + I, *args, **kwargs)
        self.N += N

    def generate_abstract(self, N, *args, **kwargs):
        I = len(self.labels)
        for i in tqdm(range(N)):
            self.generate_one(self.abstract_gen_one, i + I, *args, **kwargs)
        self.N += N

    def generate_abstract_distractors(self, N, *args, **kwargs):
        I = len(self.labels)
        for i in tqdm(range(N)):
            self.generate_one(self.abstract_gen_one_distractors,
                              i + I,
                              *args,
                              **kwargs)
        self.N += N

    # utils
    def render(self, path, mode='fixed'):
        """
        Renders the generated configurations at the desired path.
        For testing purposes; do not try with too big a dataset (not optimized)
        """
        self.compute_access_indices()
        tvecs = np.array(self.targets)
        rvecs = np.array(self.refs)
        for i in range(self.N):
            print('batch %s' % i)
            print('label %s' % self.labels[i])
            self.env.reset()
            t_state_list = list(tvecs[self.t_idx[i].numpy()])
            r_state_list = list(rvecs[self.r_idx[i].numpy()])
            self.env.from_state_list(t_state_list, norm=True)
            t_img = self.env.render(show=False, mode=mode)
            self.env.reset()
            self.env.from_state_list(r_state_list, norm=True)
            r_img = self.env.render(show=False, mode=mode)
            l  = self.labels[i]
            sep = np.ones((2, t_img.shape[1], 3)) * 255
            img = np.concatenate((t_img, sep, r_img))
            cv2.imwrite(op.join(path, 'img_%s.jpg' % i), img)
