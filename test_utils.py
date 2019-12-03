"""
A module for testing out models.

Defines various functions for testing stuff.
"""
import time
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import torch
import pygame

import env

from glob import glob
from tqdm import tqdm

from training_utils import load_dl_parts
from graph_utils import state_list_to_graph
from graph_utils import merge_graphs

def test_models(save_dir, test_list, taxon):
    """
    Tests all the models listed in save_dir on the datasets loaded from the
    test list (that lists the names of the datasets to be loaded).

    Returns the 

    taxon is the model used.
    """
    test_dls = []
    for name in test_list:
        print(name)
        test_dls.append(load_dl_parts(name))
    save_list = sorted(glob(op.join(save_dir, '*.pt')))
    for model_path in save_list:
         ...

# def play(taxon, save_path):
#     """
#     Initializes an environment that a user can play with to test the model
#     predictons.
#     """
#     e = env.Env(16, 20)
#     e.random_config(3)
#     state_list = e.to_state_list(norm=True)
#     query_graph = state_list_to_graph(state_list)
#     # add distractors
#     n_d = 5
#     for _ in range(n_d):
#         env.add_random_object()
#     s = env.to_state_list(norm=False)
#     pg = env.Playground(16, 20, s)
#     

class ModelPlayground(env.Playground):
    """
    This class extends the Playground class in the env module.
    There is an additional model (loaded from a pretrained one) that produces
    an output at each action performed, allowing to explore the evolution of
    the trained model's prediction when the user deforms the base confifuration
    or adds objects.
    """
    def __init__(self, envsize, gridsize, model, state=None):
        """
        Initializes the Model Playground.
        """
        super(ModelPlayground, self).__init__(envsize, gridsize, state)
        self.model = model

    def get_graph(self):
        """
        Gets the graph associated with the current environment state. 
        """
        s = self._env.to_state_list(norm=True)
        return state_list_to_graph(s)

    def add_shape(self):
        try:
            self._env.add_random_object()
        except env.SamplingTimeout:
            print('Sampling timed out, environment is probably quite full')

    def interactive_run(self, reset=False, new_config=True):
        """
        Plays an interactive run.

        The user can select a shape with the space bar, move a shape using the 
        arrow keys, add a random shape using the enter bar, and evaluate the 
        output of the model using the shift key.

        Press esc to end.
        """
        if reset:
            self.reset()
        if new_config:
            self._env.reset()
            n = np.random.randint(2, 5)
            self._env.random_config(n)
        # save init state list as query, make a graph out of it
        g1 = self.get_graph()

        pygame.init()
        done = False
        X = self._env.L
        Y = self._env.L
        framename = 'images/frame.jpg'
        self._env.save_image(framename)
        q_framename = 'images/q_frame.jpg'
        self._env.save_image(q_framename)
        display = pygame.display.set_mode((X, Y))
        query_display = pygame.display.set_mode((X, Y))
        pygame.display.set_caption('Playground')
        query_display.fill((0, 0, 0))
        query_display.blit(pygame.image.load(q_framename), (0, 0))
        idx = 0
        while not done:
            display.fill((0, 0, 0))
            display.blit(pygame.image.load(framename), (0, 0))
            pygame.display.update()
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_shape(idx, 3)
                    if event.key == pygame.K_RIGHT:
                        self.move_shape(idx, 2)
                    if event.key == pygame.K_UP:
                        self.move_shape(idx, 0)
                    if event.key == pygame.K_DOWN:
                        self.move_shape(idx, 1)
                    if event.key == pygame.K_SPACE:
                        idx = (idx + 1) % len(self._env.objects)
                        print(idx)
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    if event.key == pygame.K_RETURN:
                        self.add_shape()
                    if event.key == pygame.K_RSHIFT:
                        g2 = self.get_graph()
                        # evaluate model and print ouput in the terminal
                        print(self.model(g1, g2))
            self._env.save_image(framename)
        pygame.quit()

    def breaking_point_n(self, n_max, s):
        """
        This evaluation metric samples, for each s, a random configuration for
        the model, which is then run on this config.

        We then add n_max objects (randomly sampled in the environment) and see
        when the model prediction breaks. (The prediction should always be 1,
        since the base config has not been moved).

        We record the prediction of the model for the true and false classes
        depending on the number of added objects.

        We use 3 objects as benchmark (change this ?)
        """
        falses = []
        trues = []
        for _ in range(s):
            self._env.reset()
            self._env.random_config(3)
            g1 = self.get_graph()
            false = []
            true = []
            for _ in range(n_max):
                self._env.add_random_object()
                g2 = self.get_graph()
                pred = self.model(g1, g2)
                false.append(pred[0, 0].detach().numpy())
                true.append(pred[0, 1].detach().numpy())
            falses.append(false)
            trues.append(true)
        return falses, trues

    def model_heat_map(self, n, show=False, save=None):
        """
        This function samples a random configuration and then explores what
        happens to the model prediction when we vary the position of each of
        the objects over the whole map.

        n is the number of objects in the randomly sampled config.

        Returns 2 * n heat maps of the value of each of the 2 components of the
        model prediction as a function of one object's position.
        """
        self._env.reset()
        self._env.random_config(n)
        g1 = self.get_graph()
        s = self._env.to_state_list(norm=True)
        maps = []
        pos_idx = [env.N_SH+4, env.N_SH+5]
        size = self._env.envsize * self._env.gridsize
        # for state in s:
        gq = merge_graphs([state_list_to_graph(s)] * size)
        matlist = []
        for state in s:
            mem = state[pos_idx]
            mat = np.zeros((0, size, 2))
            for x in tqdm(range(size)):
                glist = []
                t = time.time()
                for y in range(size):
                    state[pos_idx] = np.array([x / self._env.gridsize,
                                               y / self._env.gridsize])
                    glist.append(state_list_to_graph(s))
                gw = merge_graphs(glist)
                pred = self.model(gq, gw).detach().numpy()
                pred = np.expand_dims(pred, 0)
                mat = np.concatenate((mat, pred), 0)
            state[pos_idx] = mem
            matlist.append(mat) # maybe change data format here
            poslist = [state[pos_idx] * self._env.gridsize for state in s]
        if show:
            for i, (mat, pos) in enumerate(zip(matlist, poslist)):
                fig, axs = plt.subplots(1, 2, constrained_layout=True)
                fig.suptitle(
                    'Scores as a function of the red object\'s position')

                axs[0].matshow(mat[..., 0])
                axs[0].set_title('Score for the "false" class')
                for j, pos in enumerate(poslist):
                    if i == j:
                        c = 'r'
                    else:
                        c = 'b'
                    axs[0].scatter(pos[1], pos[0], color=c)

                axs[1].set_title('Score for the "true" class')
                axs[1].matshow(mat[..., 1])
                for j, pos in enumerate(poslist):
                    if i == j:
                        c = 'r'
                    else:
                        c = 'b'
                    axs[1].scatter(pos[1], pos[0], color=c)

                plt.show()
            plt.close()
        return matlist, poslist
