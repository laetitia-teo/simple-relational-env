"""
This file defines simple baseline models for scene comparison.
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, ReLU

from baseline_utils import data_to_baseline_var, make_data_to_mlp_inputs

class MLPBaseline(torch.nn.Module):
    def __init__(self):
        super().__init__()

class RNNBaseline(torch.nn.Module):
    def __init__(self):
        super().__init__()

class MLPSimple(torch.nn.Module):
    def __init__(self, n_obj):

        super().__init__()
        self.n_obj = n_obj

        self.data_fn = data_to_baseline_var(self.n_obj)

    def forward(self, data):
        
        x1, _, _, _ = self.data_fn(data)

        return x1

class MLPDouble(torch.nn.Module):
    def __init__(self, n_obj):

        super().__init__()
        self.n_obj = n_obj

        self.data_fn = data_to_baseline_var(self.n_obj)

    def forward(self, data):
        
        x1, x2, _, _ = self.data_fn(data)

        return x1, x2

# simple models

class NaiveMLP(MLPSimple):

    def __init__(self, 
                 n_objects,
                 f_obj,
                 layers,
                 **kwargs):
    
        super().__init__(n_objects)

        self.layer_list = []
        f_in = n_objects * f_obj
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(*self.layer_list)

    def forward(self, data):

        x = super().forward(data)
        return self.mlp(x)

class NaiveLSTM(RNNBaseline):

    def __init__(self,
                 f_obj,
                 h,
                 layers,
                 n_layers=1,
                 **kwargs):

        super().__init__()
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
        raise NotImplementedError
        x1, _, _, _ = inputs
        out = self.lstm(x1)[0][-1]
        return self.mlp(out)

# double models

class DoubleNaiveMLP(MLPDouble):

    def __init__(self, 
                 n_objects,
                 f_obj,
                 layers,
                 **kwargs):

        super().__init__(n_objects)

        self.layer_list = []
        f_in = 2 * n_objects * f_obj
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(*self.layer_list)

    def forward(self, data):
        x1, x2, = super().forward(data)

        return self.mlp(torch.cat([x1, x2], 1))

class SceneMLP(MLPDouble):

    def __init__(self,
                 n_objects,
                 f_obj,
                 layers_scene,
                 f_scene,
                 layers_merge,
                 **kwargs):

        super().__init__(n_objects)

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

    def forward(self, data):
        
        x1, x2 = super().forward(data)

        scene1 = self.scene_mlp(x1)
        scene2 = self.scene_mlp(x2)
        return self.merge_mlp(torch.cat([scene1, scene2], 1))

class DoubleNaiveLSTM(RNNBaseline):
    """
    LSTM Baseline for double setting.
    """
    def __init__(self,
                 f_obj,
                 h,
                 layers,
                 n_layers=1,
                 **kwargs):

        super().__init__()
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
        raise NotImplementedError
        x1, x2, _, _ = inputs
        out = self.lstm(torch.cat([x1, x2], 0))[0][-1]
        return self.mlp(out)

class SceneLSTM(RNNBaseline):
    """
    LSTM baseline, with scene separation.
    """
    def __init__(self,
                 f_obj,
                 h,
                 layers,
                 f_out=2,
                 n_layers=1,
                 **kwargs):

        super().__init__()
        self.lstm = torch.nn.LSTM(f_obj, h, n_layers)
        self.layer_list = []
        f_in = 2 * h
        for f_out in layers:
            self.layer_list.append(Linear(f_in, f_out))
            self.layer_list.append(ReLU())
            f_in = f_out
        self.layer_list.append(Linear(f_in, 2))
        self.mlp = Sequential(*self.layer_list)

    def forward(self, data):
        raise NotImplementedError
        x1, x2, _, _ = inputs
        h1 = self.lstm(x1)[0][-1]
        h2 = self.lstm(x2)[0][-1]
        return self.mlp(torch.cat([h1, h2], 1))

### Image baselines

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