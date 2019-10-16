"""
Small library for creating arbitrary Graph Networks.
"""

import torch
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU
from torch.nn import Sigmoid, LayerNorm, Dropout

from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

###############################################################################
#                                                                             #
#                               MLP function                                  #
#                                                                             #
###############################################################################

def mlp_fn(hidden_layer_sizes):
    def mlp(f_in, f_out):
        """
        This function returns a Multi-Layer Perceptron with ReLU
        non-linearities with num_layers layers and h hidden nodes in each
        layer, with f_in input features and f_out output features.
        """
        layers = []
        f1 = f_in
        for f2 in hidden_layer_sizes:
            layers.append(Linear(f1, f2))
            layers.append(ReLU())
            f1 = f2
        layers.append(Linear(f1, f_out))
        # layers.append(ReLU())
        layers.append(LayerNorm(f_out))
        return Sequential(*layers)
    return mlp

###############################################################################
#                                                                             #
#                                GN Models                                    #
#                                                                             #
###############################################################################

class EdgeModelConcat(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_e_out=None):
        """
        Edge model : for each edge, computes the result as a function of the
        edge attribute, the sender and receiver node attribute, and the global
        attribute.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(EdgeModelConcat, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
        self.phi_e = model_fn(f_e + 2*f_x + f_u, f_e_out)

    def forward(self, src, dest, edge_attr, u, batch):
        """
        src [E, f_x] where E is number of edges and f_x is number of vertex
            features : source node tensor
        dest [E, f_x] : destination node tensor
        edge_attr [E, f_e] where f_e is number of edge features : edge tensor
        u [B, f_u] where B is number of batches (graphs) and f_u is number of
            global features : global tensor
        batch [E] : edge-batch mapping
        """
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.phi_e(out)

class EdgeModelDiff(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_e_out=None):
        """
        Edge model : for each edge, computes the result as a function of the
        edge attribute, the sender and receiver node attribute, and the global
        attribute.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(EdgeModelDiff, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
        self.phi_e = model_fn(f_e + f_x + f_u, f_e_out)

    def forward(self, src, dest, edge_attr, u, batch):
        """
        src [E, f_x] where E is number of edges and f_x is number of vertex
            features : source node tensor
        dest [E, f_x] : destination node tensor
        edge_attr [E, f_e] where f_e is number of edge features : edge tensor
        u [B, f_u] where B is number of batches (graphs) and f_u is number of
            global features : global tensor
        batch [E] : edge-batch mapping
        """
        out = torch.cat([dest - src, edge_attr, u[batch]], 1)
        return self.phi_e(out)

class NodeModel(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_x_out=None):
        """
        Node model : for each node, first computes the mean of every incoming
        edge attibute tensor, then uses this, in addition to the node features
        and the global features to compute the updated node attributes

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        if f_x_out is None:
            f_x_out = f_x
        super(NodeModel, self).__init__()
        self.phi_x = model_fn(f_e + f_x + f_u, f_x_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        src, dest = edge_index
        # aggregate all edges which have the same destination
        e_agg_node = scatter_mean(edge_attr, dest, dim=0)
        out = torch.cat([x, e_agg_node, u[batch]], 1)
        return self.phi_x(out)

class GlobalModel(torch.nn.Module):
    def  __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_u_out=None):
        """
        Global model : aggregates the edge attributes over the whole graph,
        the node attributes over the whole graph, and uses those to compute
        the next global value.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(GlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_e + f_x + f_u, f_u_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        src, dest = edge_index
        # compute the batch index for all edges
        e_batch = batch[src]
        # aggregate all edges in the graph
        e_agg = scatter_mean(edge_attr, e_batch, dim=0)
        # aggregate all nodes in the graph
        x_agg = scatter_mean(x, batch, dim=0)
        out = torch.cat([x_agg, e_agg, u], 1)
        return self.phi_u(out)

class NodeOnlyGlobalModel(torch.nn.Module):
    def  __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_u_out=None):
        """
        Global model : aggregates the edge attributes over the whole graph,
        the node attributes over the whole graph, and uses those to compute
        the next global value.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(NodeOnlyGlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_x + f_u, f_u_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        # aggregate all nodes in the graph
        x_agg = scatter_mean(x, batch, dim=0)
        out = torch.cat([x_agg, u], 1)
        return self.phi_u(out)

###############################################################################
#                                                                             #
#                              Direct GN Models                               #
#                                                                             #
###############################################################################

class DirectEdgeModel(torch.nn.Module):
    def __init__(self,
                 f_e,
                 model_fn,
                 f_e_out=None):
        """
        Arguments :
            - f_e (int): number of edge features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(DirectEdgeModel, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
            print(model_fn)
        self.phi_e = model_fn(f_e, f_e_out)

    def forward(self, src, dest, edge_attr, u, batch):
        """
        src [E, f_x] where E is number of edges and f_x is number of vertex
            features : source node tensor
        dest [E, f_x] : destination node tensor
        edge_attr [E, f_e] where f_e is number of edge features : edge tensor
        u [B, f_u] where B is number of batches (graphs) and f_u is number of
            global features : global tensor
        batch [E] : edge-batch mapping
        """
        return self.phi_e(edge_attr)

class DirectNodeModel(torch.nn.Module):
    def __init__(self,
                 f_x,
                 model_fn,
                 f_x_out=None):
        """
        Arguments :
            - f_x (int): number of vertex features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(DirectNodeModel, self).__init__()
        if f_x_out is None:
            f_x_out = f_x
        self.phi_x = model_fn(f_x, f_x_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        return self.phi_x(x)

class DirectGlobalModel(torch.nn.Module):
    def  __init__(self,
                 f_u,
                 model_fn,
                 f_u_out=None):
        """
        Arguments :
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(DirectGlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_u, f_u_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.phi_u(u)


###############################################################################
#                                                                             #
#                                   GN Blocks                                 #
#                                                                             #
###############################################################################

class AttentionLayer(torch.nn.Module):
    """
    GN block with attentive messages, inpired from GRANs.
    This code is based upon the code for the MetaLayer from rusty1s' PyTorch
    Geometric library, check it out here : 

    https://github.com/rusty1s/pytorch_geometric 
    """
    def __init__(self,
                 edge_model,
                 attention_model,
                 node_model,
                 global_model):
        """
        Initialize the AttentionLayer.

        This layer performs a message-passing round, but with attention weights
        on the edge features in the node computation step.

        Maybe complexify the node model when using this, in terms of what
        information it uses to update the nodes, if we only aggregate the node
        features in the global attributes.

        Arguments:
            - edge_model : model that takes as input the source and destination
                node features of each edge, and the previous edge features, and
                returns the next edge features.
            - attention_model : model that takes the same input as the edge
                model, and outputs attention vectors for each edge
            - node_model : model that takes as input the updated edges, the
                attention features, and the previous node features, computes
                the sum of all edge features flowing into the considered node
                weighted by their attention vectors, and uses this sum to
                update the node features
            - global_model : model that computes the global attribute by
                aggregating all edges and nodes. (no attention here ? or maybe
                only aggregate the nodes ?)
        """
        super(AttentionLayer, self).__init__()
        self.edge_model = edge_model
        self.attention_model = attention_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [
            self.edge_model, 
            self.attention_model, 
            self.node_model, 
            self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters

    def forward(self, x, edge_index, e, u, batch):
        """
        Forward pass
        """
        src, dest = edge_index

        e = self.edge_model(x[src], x[dest], e, u, batch[src])
        a = self.attention_model(x[src], x[dest], e, u, batch[src])
        x = self.node_model(x, edge_index, e * a, u, batch)
        u = self.global_model(x, edge_index, e, u, batch)

        return x, e, u

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    attention_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__,
                            self.edge_model,
                            self.attention_model, 
                            self.node_model,
                            self.global_model)