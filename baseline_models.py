"""
This file defines simple baseline models for scene comparison.
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU

class NaiveMLP(torch.nn.Module):
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
        super(NaiveMLP, self).__init__()
        self.layer_list = []
        f_in = 2 * n_objects * f_obj
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(*self.layer_list)

    def forward(self, data):
        return self.mlp(data)

class SceneMLP(torch.nn.Module):
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
        super(SceneMLP, self).__init__()
        # scene mlp
        self.layer_list = []
        f_in = f_obj * n_objects
        for f_out in layers_scene:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, f_scene))
        self.scene_mlp = Sequential(*self.layer_list)
        # merge mlp
        f_in = 2 * f_scene # two scenes as input to merge
        self.layer_list = []
        for f_out in layers_merge:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.merge_mlp = Sequential(*self.layer_list)

    def forward(self, data1, data2):
        """
        The forward pass of SceneMLP assumes that the states corresponding to
        the two scenes (the concatenated features of the objects) come 
        separetely.
        """
        scene1 = self.scene_mlp(data1)
        scene2 = self.scene_mlp(data2)
        return self.merge_mlp(torch.cat([scene1, scene2], 1))


class NaiveLSTM(torch.nn.Module):
    """
    LSTM Baseline.
    """
    def __init__(self,
                 f_obj,
                 h,
                 layers,
                 n_layers=1):
        """
        This baseline is based on the Long Short-Term Memory units. It
        considers the set of objects as a sequence, that is gradually fed into
        the LSTM. The sequence is the set of all objects in both scenes to 
        compare.

        It the simplest LSTM-based baseline, in that it does not separate the
        two scenes in parallel processing steps.

        Arguments :

            - f_obj (int) : number of features of the objects.
            - h (int) : size of the hidden state
            - f_out (int) : number of output features, defaults to 2.
            _ layers (int) : number of layers in the LSTM, defaults to 1.
        """
        super(NaiveLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(f_obj, h, n_layers)
        self.layer_list = []
        f_in = h
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(*self.layer_list)

    def forward(self, data):
        """
        Forward pass. Expects the data to be have as size :
        [seq_len, b_size, f_obj]

        We use the last hidden state as the latent vector we then decode using
        am mlp.
        """
        out = self.lstm(data)[0][-1]
        return self.mlp(out)

class SceneLSTM(torch.nn.Module):
    """
    LSTM baseline, with scene separation.
    """
    def __init__(self,
                 f_obj,
                 h,
                 layers,
                 f_out=2,
                 n_layers=1):
        """
        Arguments :


        """
        super(SceneLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(f_obj, h, n_layers)
        self.layer_list = []
        f_in = 2 * h
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(*self.layer_list)

    def forward(self, data1, data2):
        """
        Forward pass.
        """
        h1 = self.lstm(data1)[0][-1]
        h2 = self.lstm(data2)[0][-1]

        return self.mlp(torch.cat([h1, h2], 1))

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

class BetaVAE(torch.nn.Module):
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
        super(BetaVAE, self).__init__()
        pass

    def forward(self, image):
        pass

class EmbeddingComparison(torch.nn.Module):
    """
    This class uses the difference in embedding space to compare the two
    scenes.
    """
    def __init__(self,
                 embedding,
                 f_embed,
                 layers):
        """
        Initializes the EmbeddingComparison.

        The model works in the following way : the two scenes are embedded by 
        the provided embedding, their difference in this space is computed, and
        this difference is then processes by an MLP.

        Arguments :
            - embedding (nn model) : the embedding model.
            - f_embed (int) : the number of features in the output of the
                embedding
            - layers (iterable of ints) : number of hidden units of the
                mlp layers, excluding the output.
        """
        super(EmbeddingComparison, self).__init__()
        self.embedding = embedding
        self.layer_list = []
        f_in = f_embed
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(*self.layer_list)

    def forward(self, image1, image2):
        """
        Gives a score for each class : "different configuration" and "same
        configuration"
        """
        z1 = self.embedding(image1)
        z2 = self.embedding(image2)
        z = z1 - z2
        return self.mlp(z)