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
import time
import numpy as np
import torch
import torch_geometric

import graph_nets as gn

from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data

###############################################################################
#                                                                             #
#                            Utility functions                                #
#                                                                             #
###############################################################################

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
    # this should work ?
    edge_index = [
        [i, j] for i in range(len(x)) if i != j for j in range(len(x))]
    edge_index = torch.tensor(edge_index)
    # we initialize the edge and global features with zeros
    edge_attr = [torch.zeros(f_e) for _ in range(len(edge_index))]
    y = torch.zeros(f_u)
    return x, edge_index, edge_attr, y

def tensor_to_graphs(t):
    """
    Turns a tensor containing the objects into a complete graph.
    The tensor is of shape [batch_size, nb_objects * 2, f_x].

    Returns two graphs of torch_geometric.data.Data type.
    """
    f_x = t.shape[-1]
    n_obj = t.shape[1] // 2
    b_size = t.shape[0]

    x1 = t[:, :n_obj, :]
    x2 = t[:, n_obj:, :]

    x1 = torch.reshape(x1, (-1, f_x))
    x2 = torch.reshape(x2, (-1, f_x))
    # defining edge_index
    e = torch.zeros(n_obj, dtype=torch.long).unsqueeze(0)
    for i in range(n_obj - 1):
        e = torch.cat(
            (e, (1 + i) * torch.ones(n_obj, dtype=torch.long).unsqueeze(0)), 0)

    ei = torch.stack(
        (torch.reshape(e, (-1,)), torch.reshape(e.T, (-1,))))
    ei = ei[:, ei[0] != ei[1]]
    ei1 = ei
    for i in range(b_size - 1):
        ei1 = torch.cat((ei1, n_obj * (i + 1) + ei), 1)
    ei2 = ei1
    # edge features : initialize with difference
    e1 = x1[ei1[1]] - x1[ei1[0]]
    e2 = x2[ei2[1]] - x2[ei2[0]]
    # batches
    batch1 = torch.zeros(n_obj, dtype=torch.long)
    for i in range(b_size - 1):
        batch1 = torch.cat((batch1,
                            (i + 1) * torch.ones(n_obj, dtype=torch.long)))
        
    batch2 = batch1
    # global features : initialize with mean of node features
    u1 = scatter_mean(x1, batch1, dim=0)
    u2 = scatter_mean(x2, batch2, dim=0)
    # build graphs
    graph1 = Data(x=x1, edge_index=ei1, edge_attr=e1, y=u1, batch=batch1)
    graph2 = Data(x=x2, edge_index=ei2, edge_attr=e2, y=u2, batch=batch2)
    return graph1, graph2

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

def complete_edge_index(n):
    """
    This function creates an edge_index tensor corresponding to the
    connectivity of a complete graph of n nodes, including self-edges.
    """
    e = torch.zeros(n, dtype=torch.long).unsqueeze(0)
    for i in range(n - 1):
        e = torch.cat(
            (e, (1 + i) * torch.ones(n, dtype=torch.long).unsqueeze(0)), 0)

    ei = torch.stack(
        (torch.reshape(e, (-1,)), torch.reshape(e.T, (-1,))))

###############################################################################
#                                                                             #
#                                  Models                                     #
#                                                                             #
###############################################################################


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

class ObjectMean(GraphModel):
    """
    Simple object-based embedding model.
    """
    def __init__(self,
                 mlp_layers,
                 f_dict):
        """
        This is one of the simplest graph models we can imagine, acting on
        objects.

        There is no encoder, graph features are taken as is, and the mean
        of all objects is used as embedding. The concatenation of the two
        embeddings of the two scenes is then fed to an MLP that produces the
        final prediction.

        The model is equivalent to doing the mean of all objects present in
        the scene, which is a very rough approximation. If this performs
        well, it means that our task is really too easy.

        This model shall be used as a baseline to quantify the effects of
        aggregating with attention, and of message-passing, in graph-based
        embedding models, whose second part of the architecture remains
        constant.
        """
        super(ObjectMean, self).__init__()
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.final_mlp = model_fn(2*f_x, f_out)

    def forward(self, graph1, graph2):

        x1, edge_index1, e1, u1, batch1 = data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = data_from_graph(graph2)

        return self.final_mlp(torch.cat([u1, u2], 1))

class ObjectMeanDirectAttention(GraphModel):
    """
    Graph Embedding model, where aggregation is done by a weighted mean, and
    the attention vectors are computed on a separate basis for each node.

    We only use the node features for aggregation (?).
    """
    def __init__(self,
                 mlp_layers,
                 f_dict):
        """
        This model is also a very simple one : we use the given node features
        compute scalar attentions over nodes, and use those as weights in the
        node feature aggregation. This aggregation is then used as an
        embedding for the graph. 

        We test this model agains the simpler one with no attention, and as a
        baseline (or ablation model) for the message-passing embeddings.
        """
        super(ObjectMeanDirectAttention, self).__init__()
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)
        f_a = 1 # attentions are scalars

        self.attention_model = gn.DirectNodeModel(f_x, model_fn, f_a)

        self.final_mlp = model_fn(2*f_x, f_out)

    def embedding(self, graph):
        """
        Returns the embedding of the graph, as a weighted mean of the node
        features with attention scalars computed by an mlp on all node features
        successively.
        """
        x, edge_index, e, u, batch = data_from_graph(graph)
        a_x = self.attention_model(x, edge_index, e, u, batch)

        return scatter_mean(x * a_x, batch, dim=0) # see if this works

    def forward(self, graph1, graph2):
        """
        Forward pass.
        """
        u1 = self.embedding(graph1)
        u2 = self.embedding(graph2)

        return self.final_mlp(torch.cat([u1, u2], 1))

class GraphEmbedding(GraphModel):
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
        super(GraphEmbedding, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        # self.time_dict = {} # dict for logging computation duration

        self.encoder = MetaLayer(
            gn.DirectEdgeModel(f_e, model_fn, h),
            gn.DirectNodeModel(f_x, model_fn, h),
            gn.DirectGlobalModel(f_u, model_fn, h))

        # set the different parameters
        self.reccurent = MetaLayer(
            gn.EdgeModelDiff(f_e + h, f_x + h, f_u + h, model_fn, h),
            gn.NodeModel(h, f_x + h, f_u + h, model_fn, h),
            gn.GlobalModel(h, h, f_u + h, model_fn, h))

        self.attention_maker = MetaLayer(
            gn.EdgeModelDiff(h, h, h, model_fn, h),
            gn.NodeModel(h, h, h, model_fn, h),
            gn.GlobalModel(h, h, h, model_fn, h))

        # maybe change final embedding size
        self.aggregator = gn.GlobalModel(h, h, h, model_fn, h)

        self.final_mlp = model_fn(2 * h, f_out)

    def graph_embedding(self, x, edge_index, e, u, batch):
        """
        Graph Embedding.
        """
        out_list = []

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

            x_agg, e_agg, u_agg = x_a * x_h, e_a * e_h, u_a * u_h
            out_list.append(self.aggregator(x_agg,
                                            edge_index,
                                            e_agg,
                                            u_agg,
                                            batch))

        return out_list

    def forward(self, graph1, graph2):
        """
        Difference between embeddings.

        TODO optimize this.
        """
        x1, edge_index1, e1, u1, batch1 = data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = data_from_graph(graph2)

        l1 = self.graph_embedding(x1, edge_index1, e1, u1, batch1)
        l2 = self.graph_embedding(x2, edge_index2, e2, u2, batch2)

        # diff = u1 - u2

        return [self.final_mlp(torch.cat([u1, u2], 1))
            for u1, u2 in zip(l1, l2)]

class Simplified_GraphEmbedding(GraphModel):
    """
    A variation on the original GraphEmbedding model.
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 f_dict):
        """
        This model is a simpler, cleaner version of the GraphEmbedding model.
        In this model, all aggregations are done only on nodes, and not
        on edges : edges do not carry features useful for the aggregation any
        more, but only serve for passing messages between nodes.

        In this simple model, we do not have an encoder, the node, edge and
        global features are used as such. There is only one layer of GNN.
        """
        super(Simplified_GraphEmbedding, self).__init__()
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        # aggregation with attention
        self.gnn = MetaLayer(
            gn.EdgeModelDiff(f_e, f_x, f_u, model_fn, h),
            gn.NodeModel(h, f_x, f_u, model_fn, h),
            gn.NodeGlobalModelAttention(h, h, f_u, model_fn, h))

        self.mlp = model_fn(2 * h, f_out)
        

    def graph_embedding(self, x, edge_index, e, u, batch):
        """
        In this case the embedding is simple : we apply the gnn once, and use
        the global vector as embedding.
        """
        x_h, e_h, u_h = self.gnn(x, edge_index, e, u, batch)
        return u_h

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = data_from_graph(graph2)

        u1 = self.graph_embedding(x1, edge_index1, e1, u1, batch1)
        u2 = self.graph_embedding(x2, edge_index2, e2, u2, batch2)

        return self.mlp(torch.cat([u1, u2], 1))

class GraphDifference(GraphModel):
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
        super(GraphModel, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.mapping_fn = mapping_fn
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.encoder = MetaLayer(
            gn.DirectEdgeModel(f_e, model_fn, h),
            gn.DirectNodeModel(f_x, model_fn, h),
            gn.DirectGlobalModel(f_u, model_fn, h))

        # set the different parameters
        self.reccurent = MetaLayer(
            gn.EdgeModelDiff(f_e + h, f_x + h, f_u + h, model_fn, h),
            gn.NodeModel(h, f_x + h, f_u + h, model_fn, h),
            gn.GlobalModel(h, h, f_u + h, model_fn, h))

        self.encoder_final = MetaLayer(
            gn.DirectEdgeModel(h, model_fn, h),
            gn.DirectNodeModel(h, model_fn, h),
            gn.DirectGlobalModel(h, model_fn, h))

        self.reccurent_final = MetaLayer(
            gn.EdgeModelDiff(2*h, 2*h, 2*h, model_fn, h),
            gn.NodeModel(h, 2*h, 2*h, model_fn, h),
            gn.GlobalModel(h, h, 2*h, model_fn, h))

        self.final_mlp = model_fn(h, f_out)

    def processing(self, x, edge_index, e, u, batch):
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

        return x_h, e_h, u_h

    def processing_final(self, x, edge_index, e, u, batch):
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

        return x_h, e_h, u_h

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

        x1, e1, u1 = self.processing(
            x1, edge_index1, e1, u1, batch1)
        x2, e2, u2 = self.processing(
            x2, edge_index2, e2, u2, batch2)

        assert((edge_index1 == edge_index2).all()) # this should pass easily

        x, e, u = x1 - x2, e1 - e2, u1 - u2

        x, e, u = self.processing_final(
            x, edge_index1, e, u, batch1)

        return self.final_mlp(u)

class Alternating(GraphModel):
    """
    Class for the alternating graph model (v1)
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 N,
                 f_dict,
                 n=2):
        """
        Intitialize the alternating model.

        This model alternates between processing one graph and processing the
        other, with communication between the two processing operations done 
        by conditionning the models involved with a shared global vector.

        After N rounds of processing, alternating on each graph, the shared
        vector is decoded by a final model.

        See if training on all outputs is not better than training only on the
        final one.

        Arguments:
            - mlp_layers (list of ints): the number of units in each hidden
                layer in the mlps used in the graph networks.
            - h (int): the size of the latent graph features (for edges, nodes
                and global)
            - N (int): number of passes for the GN blocks
            - f_dict : dictionnary of feature sizes
            - n (int) : number of passes to do in each processing step.
        """
        super(Alternating, self).__init__()
        self.N = N
        self.n = n
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        # encoding the graphs to give the first latent vectors in the
        # processing step
        self.encoder = MetaLayer(
            gn.DirectEdgeModel(f_e, model_fn, h),
            gn.DirectNodeModel(f_x, model_fn, h),
            gn.DirectGlobalModel(f_u, model_fn, h))

        # attention GN Block
        self.reccurent = gn.MetaLayer(
            gn.EdgeModelDiff(f_e + h, f_x + h, f_u + h, model_fn, h),
            gn.NodeModel(h, f_x + h, f_u + h, model_fn, h),
            gn.GlobalModel(h, h, f_u + h, model_fn, h))

        self.attention_maker = MetaLayer(
            gn.EdgeModelDiff(h, h, h, model_fn, h),
            gn.NodeModel(h, h, h, model_fn, h),
            gn.GlobalModel(h, h, h, model_fn, h))

        self.aggregator = gn.GlobalModel(h, h, h, model_fn, h)

        self.decoder = model_fn(h, f_out)

    def encode(self, x, edge_index, e, u, batch, shared):
        """
        Encodes a graph, for subsequent processing.
        """
        x_h, e_h, u_h = self.encoder(
            x, edge_index, e, u, batch)


    def processing(self,
                   x,
                   x_h,
                   edge_index,
                   e,
                   e_h,
                   u,
                   u_h,
                   batch,
                   shared):
        """
        Processing step.

        shared is the shared vector.
        """
        src, dest = edge_index

        for _ in range(self.n):
            x_cat = torch.cat([x, x_h], 1)
            e_cat = torch.cat([e, e_h], 1)
            u_cat = torch.cat([u, shared], 1)

            x_h, e_h, u_h = self.reccurent(
                x_cat,
                edge_index,
                e_cat,
                u_cat,
                batch)

        x_a, e_a, u_a = self.attention_maker(
            x_h, edge_index, e_h, u_h, batch)

        x, e, u = x_a * x_h, e_a * e_h, u_a * u_h
        u = self.aggregator(x, edge_index, e, u, batch)

        return x, e, u

    def forward(self, graph1, graph2):
        """
        Forward pass

        Returns la list of self.N outputs, corresponding to each alternating
        step.
        """
        outputs = []

        x1, edge_index1, e1, u1, batch1 = data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = data_from_graph(graph2)

        x1h, e1h, u1h = self.encoder(
            x1, edge_index1, e1, u1, batch1)
        x2h, e2h, u2h = self.encoder(
            x2, edge_index2, e2, u2, batch2)

        for _ in range(self.N):
            # alternative processing
            x1h, e1h, u1h = self.processing(
                x1, x1h, edge_index1, e1, e1h, u1, u1h, batch1, u2h)
            x2h, e2h, u2h = self.processing(
                x2, x2h, edge_index2, e2, e2h, u2, u2h, batch2, u1h)
            outputs.append(self.decoder(u2h))

        return outputs

class Alternatingv2(GraphModel):
    """
    Class for the alternating graph model (v2)
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 N,
                 f_dict,
                 n=2):
        """
        Intitialize the alternating model.

        This model alternates between processing one graph and processing the
        other, with communication between the two processing operations done 
        by conditionning the models involved with a shared global vector.

        After N rounds of processing, alternating on each graph, the shared
        vector is decoded by a final model.

        See if training on all outputs is not better than training only on the
        final one.

        Arguments:
            - mlp_layers (list of ints): the number of units in each hidden
                layer in the mlps used in the graph networks.
            - h (int): the size of the latent graph features (for edges, nodes
                and global)
            - N (int): number of passes for the GN blocks
            - f_dict : dictionnary of feature sizes
            - n (int) : number of passes to do in each processing step.
        """
        super(Alternating, self).__init__()
        self.N = N
        self.n = n
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        # encoding the graphs to give the first latent vectors in the
        # processing step
        self.encoder = MetaLayer(
            gn.DirectEdgeModel(f_e, model_fn, h),
            gn.DirectNodeModel(f_x, model_fn, h),
            gn.DirectGlobalModel(f_u, model_fn, h))

        # attention GN Block
        self.reccurent = gn.AttentionLayer(
            gn.EdgeModelDiff(f_e + h, f_x + h, f_u + h, model_fn, h),
            gn.EdgeModelDiff(f_e + h, f_x + h, f_u + h, model_fn, h),
            gn.NodeModel(h, f_x + h, f_u + h, model_fn, h),
            gn.NodeOnlyGlobalModel(h, h, f_u + h, model_fn, h))

        self.decoder = model_fn(h, f_out)

    def processing(self,
                   x,
                   x_h,
                   edge_index,
                   e,
                   e_h,
                   u,
                   u_h,
                   batch,
                   shared):
        """
        Processing step.

        shared is the shared vector.
        """
        src, dest = edge_index

        for _ in range(self.n):
            x_cat = torch.cat([x, x_h, shared[batch]], 1)
            e_cat = torch.cat([e, e_h, shared[batch[src]]], 1)
            u_cat = torch.cat([u, u_h, shared], 1)

            x_h, e_h, u_h = self.reccurent(
                x_cat,
                edge_index,
                e_cat,
                u_cat,
                batch)

        return x_h, e_h, u_h

    def forward(self, graph1, graph2):
        """
        Forward pass

        Returns la list of self.N outputs, corresponding to each alternating
        step.
        """
        outputs = []

        x1, edge_index1, e1, u1, batch1 = data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = data_from_graph(graph2)

        x1h, e1h, u1h = self.encoder(
            x1, edge_index1, e1, u1, batch1)
        x2h, e2h, u2h = self.encoder(
            x2, edge_index2, e2, u2, batch2)

        for _ in range(self.N):
            # alternative processing
            x1h, e1h, u1h = self.processing(
                x1, x1h, edge_index1, e1, e1h, u1, u1h, batch1, u2h)
            x2h, e2h, u2h = self.processing(
                x2, x2h, edge_index2, e2, e2h, u2, u2h, batch2, u1h)
            outputs.append(self.decoder(u2h))

        return outputs

class GraphMatchingNetwork(GraphModel):
    """
    Graph Merging network.
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 N,
                 f_dict):
        """
        This model takes two graphs as input and merges them by forming a
        single graph where all the nodes of the first graph are connected to
        all the nodes of the second.
        """
        super(GraphMatching, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.encoder = MetaLayer(
            gn.DirectEdgeModel(f_e, model_fn, h),
            gn.DirectNodeModel(f_x, model_fn, h),
            gn.DirectGlobalModel(f_u, model_fn, h))

        self.reccurent = gn.CosineAttentionLayer(
            gn.EdgeModelDiff(f_e + h, f_x + h, f_u + h, model_fn, h),
            gn.CosineSimNodeModel(h, f_x + h, f_u + h, model_fn, h),
            gn.GlobalModel(h, h, f_u + h, model_fn, h))

        self.attention_maker = MetaLayer(
            gn.EdgeModelDiff(h, h, h, model_fn, h),
            gn.NodeModel(h, h, h, model_fn, h),
            gn.GlobalModel(h, h, h, model_fn, h))

        self.cosine_attention = gn.CosineAttention(h, h, h)

        # maybe change final embedding size
        self.aggregator = gn.GlobalModel(h, h, h, model_fn, h)

        self.final_mlp = model_fn(2 * h, f_out)

    def processing(self,
                   x,
                   x_src,
                   edge_index,
                   e,
                   u,
                   batch,):
        pass

    def forward(self, graph1, graph2):
        """
        Forward pass.

        
        """
        x1, edge_index1, e1, u1, batch1 = data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = data_from_graph(graph2)

        # encode first 
        x1h, e1h, u1h = self.encoder(
            x1, edge_index1, e1, u1, batch1)
        x2h, e2h, u2h = self.encoder(
            x2, edge_index2, e2, u2, batch2)

        for _ in range(self.N):
            # prepare vectors for first graph processing
            x1_cat = torch.cat([x1, x1_h])
            x_src_cat = torch.cat([x2, x2h]) # source is graph2
            e1_cat = torch.cat([e1, e1_h])
            u1_cat = torch.cat([u1, u1_h])

            x1h, e1h, u1h = self.reccurent(x1_cat,
                                           x_src_cat,
                                           e1_cat,
                                           u1_cat)

            # prepare vectors for second graph processing
            x2_cat = torch.cat([x2, x2_h])
            x_src_cat = torch.cat([x1, x1h]) # source is graph1 
            e2_cat = torch.cat([e2, e2_h])
            u2_cat = torch.cat([u2, u2_h])

            x2h, e2h, u2h = self.reccurent(x2_cat,
                                           x_src_cat,
                                           e2_cat,
                                           u2_cat)

        x1_a, e1_a, u1_a = self.attention_maker(
            x1_h, edge_index, e1_h, u1_h, batch)
        x1, e1, u1 = x1_h * x1_a, e1_h * e1_a, u1_h * u1_a
        u1 = self.aggregator(x1, edge_index, e1, u1, batch)

        x2_a, e2_a, u2_a = self.attention_maker(
            x2_h, edge_index, e2_h, u2_h, batch)
        x2, e2, u2 = x2_h * x2_a, e2_h * e2_a, u2_h * u2_a
        u2 = self.aggregator(x2, edge_index, e2, u2, batch)

        return self.final_mlp(torch.cat([u1, u2]))