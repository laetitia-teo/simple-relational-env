"""
This file defines simple baseline models for scene comparison.

The task : two configurations are presented in as input, and the model has
to decide if they are the same or not.
We can imagine several versions of the task, with incrasing difficulty :
    - "Same configuration" means we need to have the same shapes, with the
    same colors, with the same orientation to be considered as same config.
    The transformations applied to the scene would be translations and small
    scalings.
    - "Same <attribute>" : in this version of the task, the models would be 
    trained to recognise if the objects in the scene all share the same
    attribute, such as color or orientation, and if the attribute they share
    is the same as in the other image.
    - "Same spatial configuration" : this task is similar to the first one,
    but the models are required to abstract from shape, orientation and color
    information, as well as from absolute positions, to concentrate only on
    relative distances. No scaling here, since the size of the shapes may vary.

It would also be interesting to see if we can generalize from one task to the 
next, or have a task-conditioned model that can achieve good results on all
tasks.

For the first experiments, we shall focus on the first task, the more intuitive
notion of "same configuration". We shall need an appropriate dataset that mixes
the same shapes with same colors/sizes/orientations in different spatial
configurations.

The models are :

A simple MLP taking in the environment state (how to adapt to different
number of objects ?)

A CNN-based model that works directly from pixels (with a pretrained 
embedding maybe - generic on other images, or maybe use the latent layer of a 
VAE trained to reconstruct the shapes)

Other interesting models to consider : LSTM with attention (taking in the 
objects as a sequence), transformer models. We'll see.
"""

###############################################################################
#                                                                             #
#                             Vector-based Baselines                          #
#                                                                             #
###############################################################################

import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU

class NaiveMLP():
    """
    The simplest, most unstructured model.
    """
    def __init__(self, 
                 n_objects,
                 f_obj,
                 layers):
        """
        Initializes the NaiveMLP.

        This model simply concatenates the input vectors corresponding to both
        configurations, and applies several linear layers.
        The output is a vector of size 2 giving the raw scores for each of the
        classes : "is not the same config" and "is the same config".

        Arguments :
            - n_objects (int) : number of objects to consider;
            - f_obj (int) : number of features of an object;
            - layers (iterable of ints) : number of hidden units of the
                different layers, excluding the output.
        """
        self.layer_list = []
        f_in = n_objects * f_obj
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLu())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(self.layer_list)

    def forward(self, data):
        return self.mlp(data)

class SceneMLP():
    """
    A model that is a bit more structured than NaiveMLP.
    """
    def __init__(self,
                 n_objects,
                 f_obj,
                 layers_scene,
                 f_scene,
                 layers_merge):
        """
        Initializes the SceneMLP.

        This model incorporates the following assumption : the two scenes
        should be treated the same. Consequently, the weights are shared 
        between the two scene-processing modules, and then the two scene
        feature vectors are used in the final processing module.

        Arguments :
            - n_objects (int) : number of objects to consider;
            - f_obj (int) : number of features of an object;
            - layers_scene (iterable of ints) : number of hidden units of the
                scene-processing layers
            - f_scene (int) : number of features for representing the scene.
            - layers_merge (iterable of ints) : number of hidden units of the
                final merging layers.
        """
        # scene mlp
        self.layer_list = []
        f_in = f_obj * n_objects
        for f_out in layers_scene:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLu())
            f_in = f_out
        self.layer_list.append(Linear(f_in, f_scene))
        self.scene_mlp = Sequential(self.layer_list)
        # merge mlp
        f_in = 2 * f_scene # tho scenes as input to merge
        self.layer_list = []
        for f_out in layers_merge:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLu())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.merge_mlp = Sequential(self.layer_list)

    def forward(self, data1, data2):
        """
        The forward pass of SceneMLP assumes that the states corresponding to
        the two scenes (the concatenated features of the objects) come 
        separetely.
        """
        scene1 = self.scene_mlp(data1)
        scene2 = self.scene_mlp(data2)
        return self.merge_mlp(torch.cat([scene1, scene2]))


###############################################################################
#                                                                             #
#                              Image-based Baselines                          #
#                                                                             #
###############################################################################


class CNNBaseline(object):
    """docstring for CNNBaseline"""
    def __init__(self, arg):
        super(CNNBaseline, self).__init__()
        self.arg = arg

class BetaVAE():
    """
    Beta-VAE used to learn embeddings of the scenes, to be used subsequently 
    for defining a distance between two scenes in embedding space.
    """
    def __init__(self,
                 image_size,
                 f_z,
                 layers):
        """
        Initializes the BetaVAE.
        """
        pass

    def forward(self, image):
        pass

class EmbeddingComparison():
    """
    This class uses the difference in embedding space to compare the two
    scenes.
    """
    def __init__(self,
                 embedding,
                 f_embed
                 layers):
        """
        Initializes the EmbeddingComparison.

        The model works in the following way : the two scenes are embedded by 
        the provided embedding, their difference in this space is computed, and
        this difference is then processes by an MLP.

        Arguments :
            - embedding (model) : the embedding model.
            - f_embed (int) : the number of features in the output of the
                embedding
            - layers (iterable of ints) : number of hidden units of the
                mlp layers, excluding the output.
        """
        self.embedding = embedding
        self.layer_list = []
        f_in = f_embed
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLu())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(self.layer_list)

    def forward(self, image1, image2):
        """
        Gives a score for each class : "different configuration" and "same
        configuration"
        """
        z1 = self.embedding(image1)
        z2 = self.embedding(image2)
        z = z1 - z2
        return self.mlp(z)