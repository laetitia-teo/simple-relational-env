"""
This file defines graph models for scene classification.

The model should be able to convert (maybe do this in a different class) the
object list it is given into a scene graph. The model should take two graphs as
input and output the probability of the two graphs being the same.

How to compute distance in the space of graphs ? Several approaches could be
considered : use global variable as a graph embedding, use graph-graph
comparisons with attention, or train a distance function on time-series of
graphs being jittred.
"""

import torch
import torch_geometric

import graph_nets as gn

from torch_geometric import MetaLayer
from torch_geometric.data import Data

def scene_to_complete_graph(state_list, f_e, f_u):
    """
    Takes in a scene state and returns a complete graph.

    Arguments :
        - state_list (list of vectors) : the state list generated from the
            environment.
        - f_e (int) : number of edge features
        - f_u (int) : number of global features.

    Returns :
        - x, edge_index, edge_attr, y; respectively the node features, the 
            connectivity, the edge features, the grobal features.
    """
    x = torch.tensor([state for state in state_list])
    edge_index = [[i, j] for i in range(len(x)) for j in range(len(x))]
    edge_index = torch.tensor(edge_index)
    # we initialize the edge and global features with zeros
    edge_attr = [torch.zeros(f_e) for _ in range(len(edge_index))]
    y = torch.zeros(f_u)
    return x, edge_index, edge_attr, y

class GraphModel(torch.nn.Module):
    """
    Base class for all models operating on graphs.
    """
    def __init__(self):
        super(GraphModel, self).__init__()

    def get_features(self, f_dict):
        """
        Gets the input and output features for graph processing.
        """
        f_e = f_dict['f_e']
        f_x = f_dict['f_x']
        f_u = f_dict['f_u']
        f_out = f_dict['f_out']
        return f_e, f_x, f_u, f_out

class GraphEmbedding(torch.nn.Module):
    """
    GraphEmbedding model.
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 N,
                 f_dict):
        """
        This model processes the inputs into two graphs that undergo a certain
        number of recurrent message-passing rounds that yield two embeddings
        that can then serve as a basis for comparison.

        One graph is processed to give a latent representation for each node
        and edge, and also to give an attention graph over nodes and edges.
        This attention graph is used as weights for the edge and node features
        to be aggregated in the embedding.
        """
        super(GraphEmbedding, self).__init__(self)
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.encoder = MetaLayer(
            DirectEdgeModel(f_e, model_fn, h),
            DirectNodeModel(f_x, model_fn, h),
            DirectGlobalModel(f_u, model_fn, h))

        # set the different parameters
        self.reccurent = MetaLayer(
            gn.EdgeModelDiff(2*h, 2*h, 2*h, model_fn, h),
            gn.NodeModel(h, 2*h, 2*h, model_fn, h),
            gn.GlobalModel(h, h, 2*h, model_fn, h))

        self.attention_maker = MetaLayer(
            gn.EdgeModelDiff(h, h, h, model_fn, h),
            gn.NodeModel(h, h, h, model_fn, h),
            gn.GlobalModel(h, h, h, model_fn, h))

        # maybe change final embedding size
        self.aggregator = gn.GlobalModel(h, h, h, model_fn, h)

        self.final_mlp = model_fn(h, f_out)

    def graph_embedding(self, x, edge_index, e, u, batch):
        """
        Graph Embedding.
        """
        x_h, e_h, u_h = self.encoder(
            x, edge_index, e, u, batch)

        for _ in range(N):
            x_cat = torch.cat([x, x_h], 1)
            e_cat = torch.cat([e, e_h], 1)
            u_cat = torch.cat([u, u_h], 1)

            x_h, e_h, u_h = self.reccurent(
                x_cat,
                edge_index,
                e_cat,
                u_cat,
                batch)

        x_a, e_a, u_a = self.attention_maker(
            x_h, edge_index, e_h, u_h, batch)

        x, e, u = x_a * x_h, e_a * e_h, u_a * u_h
        return self.aggregator(x, edge_index, e, u, batch)

    def forward (self, graph1, graph2):
        """
        Difference between embeddings.

        TODO optimize this.
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        u1 = self.graph_embedding(x1, edge_index1, e1, u1, batch1)
        u2 = self.graph_embedding(x2, edge_index2, e2, u2, batch2)

        diff = u1 - u2

        return final_mlp(diff)


class GraphDifference(torch.nn.Module):
    """
    GraphModel.
    """
    def __init__(self,
                 mlp_layers):
        """
        Initialization of a GraphModel.
        """
        super(GraphModel, self).__init__(self)
        self.mlp_maker = gn.mlp_fn(mlp_layers)
