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
import numpy as np
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

def data_from_graph(graph):
    x = graph.x
    edge_index = graph.edge_index
    e = graph.edge_attr
    u = graph.y
    batch = graph.batch
    return x, edge_index, e, u, batch

def identity_mapping(graph1, graph2):
    n = len(graph1.x)
    if len(graph2.x) != n:
        raise ValueError('The two graphs to map have different node numbers')
    return torch.tensor(np.arange(n))

def permute_graph(graph, mapping):
    """
    Takes as input a graph and a mapping, and permutes all the indices of
    features of nodes of the first graph to match the indices of the mapping.

    Arguments :
        - graph : (Data) graph to permute.
        - mapping : (torch tensor) permutation tensor
    """
    graph.x = graph.x[mapping]
    graph.edge_index = mapping[graph.edge_index]
    # no need to permute edge attributes, global attributes or batch
    return graph


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
        self.N = N
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

        for _ in range(self.N):
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

    def forward(self, graph1, graph2):
        """
        Difference between embeddings.

        TODO optimize this.
        """
        x1, edge_index1, e1, u1, batch1 = data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = data_from_graph(graph2)

        u1 = self.graph_embedding(x1, edge_index1, e1, u1, batch1)
        u2 = self.graph_embedding(x2, edge_index2, e2, u2, batch2)

        diff = u1 - u2

        return final_mlp(diff)


class GraphDifference(torch.nn.Module):
    """
    GraphModel.
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 N,
                 f_dict,
                 mapping_fn):
        """
        Initialization of a GraphDifference model.

        The model works by encoding and then processing the inputs of the
        graphs, then mapping with a provided mapping, the two graphs 
        together node by node and merging them in a single graph where the
        features are the node-wise and edge-wise features for nodes and edges 
        respectively. This difference graph is then processed by a final graph
        network, and the resulting global feature is fed to the final mlp.

        Arguments :
            - mlp_layers (list of ints): the number of units in each hidden
                layer in the mlps used in the graph networks.
            - h (int): the size of the latent graph features (for edges, nodes
                and global)
            - N (int): number of passes for the GN blocks
            - f_dict : dictionnary of feature sizes
            - mapping_fn (function): a function that takes in two graphs and 
                returns a tensor representing a permutation of the indices
                to map the second graph on the first one.
        """
        super(GraphModel, self).__init__(self)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.mapping_fn = mapping_fn
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

        self.encoder_final = MetaLayer(
            DirectEdgeModel(h, model_fn, h),
            DirectNodeModel(h, model_fn, h),
            DirectGlobalModel(h, model_fn, h))

        self.reccurent_final = MetaLayer(
            gn.EdgeModelDiff(2*h, 2*h, 2*h, model_fn, h),
            gn.NodeModel(h, 2*h, 2*h, model_fn, h),
            gn.GlobalModel(h, h, 2*h, model_fn, h))

        self.final_mlp = model_fn(h, f_out)

    def processing(self, x, edge_index, edge_attr, u, batch):
        """
        Processing a single graph, before merging.
        """
        x_h, e_h, u_h = self.encoder(
            x, edge_index, e, u, batch)

        for _ in range(self.N):
            x_cat = torch.cat([x, x_h], 1)
            e_cat = torch.cat([e, e_h], 1)
            u_cat = torch.cat([u, u_h], 1)

            x_h, e_h, u_h = self.reccurent(
                x_cat,
                edge_index,
                e_cat,
                u_cat,
                batch)

        return x_h, edge_index, e_h, u_h, batch

    def processing_final(self, x, edge_index, edge_attr, u, batch):
        """
        Processing a single graph, before merging.
        """
        x_h, e_h, u_h = self.encoder_final(
            x, edge_index, e, u, batch)

        for _ in range(self.N):
            x_cat = torch.cat([x, x_h], 1)
            e_cat = torch.cat([e, e_h], 1)
            u_cat = torch.cat([u, u_h], 1)

            x_h, e_h, u_h = self.reccurent_final(
                x_cat,
                edge_index,
                e_cat,
                u_cat,
                batch)

        return x_h, edge_index, e_h, u_h, batch

    def forward(self, graph1, graph2):
        """
        Forward pass.

        Arguments :
            - graph1, graph2 : the two graphs to compare.
        """
        mapping = self.mapping_fn(graph2, graph1)
        graph2 = permute_graph(graph2, mapping)

        x1, edge_index1, e1, u1, batch1 = data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = data_from_graph(graph2)

        x1, edge_index1, e1, u1, batch1 = self.processing(
            x1, edge_index1, e1, u1, batch1)
        x2, edge_index2, e2, u2, batch2 = self.processing(
            x2, edge_index2, e2, u2, batch2)

        assert(edge_index1 == edge_index2) # this should pass easily

        x, e, u = x1 - x2, e1 - e2, u1 - u2

        x, edge_index, e, u, batch = self.processing_final(
            x, edge_index, e, u, batch)

        return self.final_mlp(u)

class Alternating(torch.nn.Module):
    """
    Class for the alternating graph model.
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 N,
                 f_dict,
                 n=1):
        """
        Intitialize the alternating model.

        This model alternates between processing one graph and processing the
        other, with communication between the two processing operations done 
        by conditionning the models involved with a shared global vector.

        After N rounds of processing, alternating on each graph, the shared
        vector is decoded by a final model.

        Arguments:
            - mlp_layers (list of ints): the number of units in each hidden
                layer in the mlps used in the graph networks.
            - h (int): the size of the latent graph features (for edges, nodes
                and global)
            - N (int): number of passes for the GN blocks
            - f_dict : dictionnary of feature sizes
            - n (int) : number of passes to do in each processing step.
        """
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        # encoding the graphs to give the first latent vectors in the
        # processing step
        self.encoder = MetaLayer(
            DirectEdgeModel(f_e, model_fn, h),
            DirectNodeModel(f_x, model_fn, h),
            DirectGlobalModel(f_u, model_fn, h))

        # attention GN Block
        self.reccurent = MetaLayer(
            gn.EdgeModelDiff(2*h, 2*h, 2*h, model_fn, h),
            gn.NodeModel(h, 2*h, 2*h, model_fn, h),
            gn.GlobalModel(h, h, 2*h, model_fn, h))

        self.decoder = model_fn(h, f_out)