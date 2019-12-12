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

from graph_utils import data_from_graph_maker
from graph_utils import cross_graph_ei_maker


###############################################################################
#                                                                             #
#                                  Models                                     #
#                                                                             #
###############################################################################


class GraphModel(torch.nn.Module):
    """
    Base class for all models operating on graphs.

    Graph models are on CPU by default, call .cuda() to pass computation on
    GPU.
    """
    def __init__(self):
        super(GraphModel, self).__init__()
        self.GPU = False

        self.data_from_graph = data_from_graph_maker()
        # not all models use this, maybe subclass into CrossGraphModel
        self.cross_graph_ei = cross_graph_ei_maker()

        self.task_type = 'parts_task' # by default

    def get_features(self, f_dict):
        """
        Gets the input and output features for graph processing.
        """
        f_e = f_dict['f_e']
        f_x = f_dict['f_x']
        f_u = f_dict['f_u']
        f_out = f_dict['f_out']
        return f_e, f_x, f_u, f_out

    def cuda(self):
        super(GraphModel, self).cuda()
        self.GPU = True

        self.data_from_graph = data_from_graph_maker(cuda=True)
        self.cross_graph_ei = cross_graph_ei_maker(cuda=True)

    def cpu(self):
        super(GraphModel, self).cpu()
        self.GPU = False

        self.data_from_graph = data_from_graph_maker(cuda=False)
        self.cross_graph_ei = cross_graph_ei_maker(cuda=False)

    def reset_parameters(self):
        """
        Resets all the parameters of the neural networks in the model.
        """
        # for keys in 
        pass

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

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

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
        x, edge_index, e, u, batch = self.data_from_graph(graph)
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
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

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
            gn.EdgeModelConcat(f_e, f_x, f_u, model_fn, h),
            gn.NodeModel(h, f_x, f_u, model_fn, h),
            gn.GlobalModelNodeAttention(h, h, f_u, model_fn, h))

        self.mlp = model_fn(2 * h, f_out)
        

    def graph_embedding(self, x, edge_index, e, u, batch):
        """
        In this case the embedding is simple : we apply the gnn once, and use
        the global vector as embedding.
        """
        x_h, e_h, u_h = self.gnn(x, edge_index, e, u, batch)
        return u_h

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

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

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

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

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

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
        super(Alternatingv2, self).__init__()
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

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

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

class AlternatingSimple(GraphModel):
    """
    Simple version of the Altrenating model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        """
        Simpler version of the alternating model. In this model there is no
        encoder network, we only have 1 layer of GNN on each processing step.

        We condition on the output global embedding from the processing on the
        previous graph, and we only condition the node computations since there
        are less nodes than edges (this is a choice that can be discussed).

        We aggregate nodes with attention in the global model.

        We use the same gnn for processing both inputs.
        In this model, since we may want to chain the passes, we let the number
        of input features unchanged.
        """
        super(AlternatingSimple, self).__init__()
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = MetaLayer(
            gn.EdgeModelDiff(f_e, f_x + f_u, f_u, model_fn, f_e),
            gn.NodeModel(f_e, f_x + f_u, f_u, model_fn, f_x),
            gn.GlobalModelNodeAttention(f_e, f_x, f_u, model_fn, f_u))

        self.mlp = model_fn(2 * f_u, f_out)

    def forward(self, graph1, graph2):
        """
        Forward pass. We alternate computing on 1 graph and then on the other.
        We initialize the conditioning vector at 0.
        At each step we concatenate the global vectors to the node vectors.
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            # we can do N passes of this
            x1 = torch.cat([x1, u2[batch1]], 1)
            x1, e1, u1 = self.gnn(x1, edge_index1, e1, u1, batch1)
            x2 = torch.cat([x2, u1[batch2]], 1)
            x2, e2, u2 = self.gnn(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list


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
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

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

class GraphMatchingSimple(GraphModel):
    """
    Simpler version of the Graph Matching Network.
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 N,
                 f_dict):
        """
        This simpler version of the GMN has only one layer of internal
        propagation.
        """
        super(GraphMatchingSimple, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = gn.CosineAttentionLayer(
            gn.EdgeModelDiff(f_e, f_x, f_u, model_fn, h),
            gn.CosineAttention(h, f_x, f_u),
            gn.CosineSimNodeModel(h, f_x, f_u, model_fn, h),
            gn.GlobalModel(h, h, f_u, model_fn, h))

        self.mlp = model_fn(2 * h, f_out)

    def forward(self, graph1, graph2):
        """
        Forward pass.
        """
        # if self.cg_ei is None or self.b_size != len(graph1.y):
        #     # artificially constructing this reduces the model's generality
        #     # we want to have models that can also reason on different
        #     # graphs
        #     # there should be a way to compute thsi for any two graphs
        #     # at the cost of some generality
        #     n_obj = len(graph1.x) // len(graph1.y)
        #     self.b_size = len(graph1.y)
        #     ei = complete_edge_index(n_obj)
        #     self.cg_ei = ei
        #     for i in range(self.b_size - 1):
        #         self.cg_ei = torch.cat([self.cg_ei, ei + (i + 1) * n_obj], 1)

        # build complete cross-graph edge index tensor
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        cg_ei = self.cross_graph_ei(batch1, batch2)

        x1, e1, u1 = self.gnn(x1,
                              x2,
                              edge_index1,
                              torch.flip(cg_ei, (0,)),
                              e1,
                              u1,
                              batch1,
                              batch2)

        x2, e2, u2 = self.gnn(x2,
                              x1,
                              edge_index2,
                              cg_ei,
                              e2,
                              u2,
                              batch2,
                              batch1)

        return self.mlp(torch.cat([u1, u2], 1))

class GraphMatchingv2(GraphModel):
    """
    New version of the GMN, with learned cross-graph attentions.
    We use the nodes and edges for aggregation.
    """
    def __init__(self, 
                 mlp_layers,
                 h,
                 N,
                 f_dict):
        """
        Initialization.

        The motivation for this model is that in the standard implementation of
        the GMN model, the cross graph attentions between two nodes decreases
        when the nodes are similar (to implement matching : the less the
        graph match, the more cross-graph attention there is, because the 
        learned distance function should be higher).
        The intuition behind our model is somewhat the opposite : our model 
        should be able to select, in the bigger scene, the objects that are
        also present in the smaller one, and not attend to the other ones.
        """
        super(GraphMatchingv2, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = gn.CrossGraphAttentionLayer(
            gn.LearnedCrossGraphAttention(f_x, model_fn),
            gn.EdgeModelDiff(f_e, f_x, f_u, model_fn, h), # edgemodelconcat here ?
            gn.NodeModel(h, f_x, f_u, model_fn, h),
            gn.GlobalModel(h, h, f_u, model_fn, h))

        self.mlp = model_fn(2 * h, f_out)

    def forward(self, graph1, graph2):
        """
        Forward pass.

        Same outer logic as the regular GMN model, only the intra-layer logic
        differs.
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        cg_ei = self.cross_graph_ei(batch1, batch2)

        _, _, u1 = self.gnn(x1,
                            x2,
                            edge_index1,
                            torch.flip(cg_ei, (0,)),
                            e1,
                            u1,
                            batch1,
                            batch2)
        # we use the original features to compute the cross-graph attentions
        _, _, u2 = self.gnn(x2,
                            x1,
                            edge_index2,
                            cg_ei,
                            e2,
                            u2,
                            batch2,
                            batch1)

        return self.mlp(torch.cat([u1, u2], 1))

class GraphMatchingv2_EdgeConcat(GraphModel):
    """
    Upgraded version of the preceding model.
    Replaced EdgeModelDiff with EdgeModelConcat;
    """
    def __init__(self, 
                 mlp_layers,
                 h,
                 N,
                 f_dict):
        """
        Init.
        """
        super(GraphMatchingv2_EdgeConcat, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = gn.CrossGraphAttentionLayer(
            gn.LearnedCrossGraphAttention(f_x, model_fn),
            gn.EdgeModelConcat(f_e, f_x, f_u, model_fn, h), # edgemodelconcat here ?
            gn.NodeModel(h, f_x, f_u, model_fn, h),
            gn.GlobalModel(h, h, f_u, model_fn, h))

        self.mlp = model_fn(2 * h, f_out)

    def forward(self, graph1, graph2):
        """
        Forward pass.

        Same outer logic as the regular GMN model, only the intra-layer logic
        differs.

        The ordering of graphs matters here : the first graph is the query
        graph, the second one is the world graph.
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        cg_ei = self.cross_graph_ei(batch1, batch2)

        _, _, u1 = self.gnn(x1,
                            x2,
                            edge_index1,
                            torch.flip(cg_ei, (0,)),
                            e1,
                            u1,
                            batch1,
                            batch2)
        # we use the original features to compute the cross-graph attentions
        _, _, u2 = self.gnn(x2,
                            x1,
                            edge_index2,
                            cg_ei,
                            e2,
                            u2,
                            batch2,
                            batch1)

        return self.mlp(torch.cat([u1, u2], 1))

class GraphMatchingv2_DiffGNNs(GraphModel):
    def __init__(self, 
                 mlp_layers,
                 h,
                 N,
                 f_dict):
        """
        Same as previous model, but we use 2 different GNNs for the 2 different
        input graphs.
        """
        super(GraphMatchingv2_DiffGNNs, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn_q = gn.CrossGraphAttentionLayer(
            gn.LearnedCrossGraphAttention(f_x, model_fn),
            gn.EdgeModelDiff(f_e, f_x, f_u, model_fn, h), # edgemodelconcat here ?
            gn.NodeModel(h, f_x, f_u, model_fn, h),
            gn.GlobalModel(h, h, f_u, model_fn, h))

        self.gnn_w = gn.CrossGraphAttentionLayer(
            gn.LearnedCrossGraphAttention(f_x, model_fn),
            gn.EdgeModelDiff(f_e, f_x, f_u, model_fn, h), # edgemodelconcat here ?
            gn.NodeModel(h, f_x, f_u, model_fn, h),
            gn.GlobalModel(h, h, f_u, model_fn, h))

        self.mlp = model_fn(2 * h, f_out)

    def forward(self, graph1, graph2):
        """
        Forward pass.

        Same outer logic as the regular GMN model, only the intra-layer logic
        differs.

        The ordering of graphs matters here : the first graph is the query
        graph, the second one is the world graph.
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        cg_ei = self.cross_graph_ei(batch1, batch2)

        _, _, u1 = self.gnn_q(x1,
                              x2,
                              edge_index1,
                              torch.flip(cg_ei, (0,)),
                              e1,
                              u1,
                              batch1,
                              batch2)
        # we use the original features to compute the cross-graph attentions
        _, _, u2 = self.gnn_w(x2,
                              x1,
                              edge_index2,
                              cg_ei,
                              e2,
                              u2,
                              batch2,
                              batch1)

        return self.mlp(torch.cat([u1, u2], 1))

class GraphMatchingv2_NoMem(GraphModel):
    """
    Change Edge Model : the edges keep no memory of their previous state.
    """
    def __init__(self, 
                 mlp_layers,
                 h,
                 N,
                 f_dict):
        """
        """
        super(GraphMatchingv2_NoMem, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = gn.CrossGraphAttentionLayer(
            gn.LearnedCrossGraphAttention(f_x, model_fn),
            gn.EdgeModelConcatNoMem(f_e, f_x, f_u, model_fn, h),
            gn.NodeModel(h, f_x, f_u, model_fn, h),
            gn.GlobalModel(h, h, f_u, model_fn, h))

        self.mlp = model_fn(2 * h, f_out)

    def forward(self, graph1, graph2):
        """
        Forward pass.

        Same outer logic as the regular GMN model, only the intra-layer logic
        differs.
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        cg_ei = self.cross_graph_ei(batch1, batch2)

        _, _, u1 = self.gnn(x1,
                            x2,
                            edge_index1,
                            torch.flip(cg_ei, (0,)),
                            e1,
                            u1,
                            batch1,
                            batch2)
        # we use the original features to compute the cross-graph attentions
        _, _, u2 = self.gnn(x2,
                            x1,
                            edge_index2,
                            cg_ei,
                            e2,
                            u2,
                            batch2,
                            batch1)

        return self.mlp(torch.cat([u1, u2], 1))

class ConditionByGraphEmbedding(GraphModel):
    """
    In this model, there is no symmetry between the two graphs anymore; the 
    query graph is processed, and an embedding is made out of it.

    I'm not sure this model 
    """
    def __init__(self, 
                 mlp_layers,
                 h,
                 h_u,
                 N,
                 f_dict):
        """
        Description.

        h has to have same dimension as the input features.
        """
        super(ConditionByGraphEmbedding, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)

        # simple 1-time-step GNN layer for 
        self.gnn_q = gn.MetaLayer(
            gn.EdgeModelDiff(f_e, f_x, f_u, model_fn, h),
            gn.NodeModel(h, f_x, f_u, model_fn, h),
            gn.GlobalModel(h, h, f_u, model_fn, h_u))

        self.gnn_w = gn.MetaLayer(
            gn.EdgeModelDiff(f_e + h_u, f_x + h_u, f_u + h_u, model_fn, h),
            gn.NodeModel(h, f_x + h_u, f_u + h_u, model_fn, h),
            gn.GlobalModel(h, h, f_u + h_u, model_fn, h))

        self.resize_u = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(h_u, f_u))

        self.mlp = model_fn(h, f_out)

    def forward(self, graph_q, graph_w):
        """
        Description...
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph_q)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph_w)

        out_list = []

        for _ in range(self.N):
            x1, e1, u1 = self.gnn_q(x1, edge_index1, e1, u1, batch1)
            # condition with the query graph's embedding
            x2 = torch.cat([x2, u1[batch2]], 1)
            # compute the batch to which each edge belongs
            src, dest = edge_index2
            e2 = torch.cat([e2, u1[batch2[src]]], 1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn_w(x2, edge_index2, e2, u2, batch2)

            # transform the global to have the same size
            u1 = self.resize_u(u1)

            out_list.append(self.mlp(u2))

        return out_list

### A try at universal GMs ###

class GraphMatchingv2_U(GraphModel):
    """
    Universal version of the GraphMatchingv2 model.
    """
    def __init__(self,
                 mlp_layers,
                 h,
                 N,
                 f_dict,
                 task_type='scene'):
        """
        Description.

        h has to have same dimension as the input features.
        """
        super(GraphMatchingv2_U, self).__init__()
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        f_e, f_x, f_u, f_out = self.get_features(f_dict)
        self.task_type = task_type

        self.gnn = gn.CrossGraphAttentionLayer(
            gn.LearnedCrossGraphAttention(f_x, model_fn),
            gn.EdgeModelConcatNoMem(f_e, f_x, f_u, model_fn, h),
            gn.NodeModelAdd(h, f_x, f_u, model_fn, h),
            gn.GlobalModel(h, h, f_u, model_fn, h))

        if self.task_type == 'scene':
            self.mlp = model_fn(2 * h, f_out)
        if self.task_type == 'objects':
            self.mlp = model_fn(h, f_out) 

    def forward(self, graph1, graph2):
        """
        
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        cg_ei = self.cross_graph_ei(batch1, batch2)

        x1_, e1_, u1 = self.gnn(x1,
                                x2,
                                edge_index1,
                                torch.flip(cg_ei, (0,)),
                                e1,
                                u1,
                                batch1,
                                batch2)
        # we use the original features to compute the cross-graph attentions
        x2_, e2_, u2 = self.gnn(x2,
                                x1,
                                edge_index2,
                                cg_ei,
                                e2,
                                u2,
                                batch2,
                                batch1)
        if self.task_type == 'objects':
            return self.mlp(x2_)
        if self.task_type == 'scene':
            return self.mlp(torch.cat([u1, u2], 1))