"""
Cleaner version of the graph_nets module.
"""

import torch
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU
from torch.nn import Sigmoid, LayerNorm, Dropout
from torch.nn import BatchNorm1d

from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from torch_geometric.nn import MetaLayer

from utils import cosine_similarity
from utils import cos_sim
from utils import sim

# mlp function

def mlp_fn(hidden_layer_sizes, normalize=False):
    def mlp(f_in, f_out):
        """
        This function returns a Multi-Layer Perceptron with ReLU
        non-linearities with num_layers layers and h hidden nodes in each
        layer, with f_in input features and f_out output features.
        """
        layers = []
        f1 = f_in
        for i, f2 in enumerate(hidden_layer_sizes):
            layers.append(Linear(f1, f2))
            if (i == len(hidden_layer_sizes) - 1) and normalize:
                layers.append(LayerNorm(f2))
            layers.append(ReLU())
            f1 = f2
        layers.append(Linear(f1, f_out))
        # layers.append(ReLU())
        # layers.append(LayerNorm(f_out))
        return Sequential(*layers)
    return mlp

# Node and Global models for Deep Sets

class DS_NodeModel(torch.nn.Module):
    def __init__(self,
                 f_x,
                 f_u,
                 model_fn,
                 f_x_out=None):
        super(DS_NodeModel, self).__init__()
        if f_x_out is None:
            f_x_out = f_x
        self.phi_x = model_fn(f_x + f_u, f_x_out)

    def forward(self, x, u, batch):
        return self.phi_x(torch.cat([x, u], 1))

class DS_GlobalModel(torch.nn.Module):
    def  __init__(self,
                  f_x,
                  f_u,
                  model_fn,
                  f_u_out=None):
        super(DS_GlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_u, f_u_out)

    def forward(self, x, u, batch):
        x_agg = scatter_add(x, batch, dim=0)
        return self.phi_u(torch.cat([x_agg, u], 1))

class DS_GlobalModel_A(torch.nn.Module):
    """
    With attention.
    """
    def  __init__(self,
                  f_x,
                  f_u,
                  model_fn,
                  f_u_out=None):
        super(DS_GlobalModel_A, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_x + f_u, f_u_out)

    def forward(self, x, u, batch):
        a = ... # TODO : define attention function
        x_agg = scatter_add(a * x, batch, dim=0)
        return self.phi_u(torch.cat([x_agg, u], 1))

# Edge, Node and Global models for GNNs

class EdgeModelConcat(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_e_out=None):
        super(EdgeModelConcat, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
        self.phi_e = model_fn(f_e + 2*f_x + f_u, f_e_out)

    def forward(self, src, dest, e, u, batch):
        out = torch.cat([src, dest, e, u[batch]], 1)
        return self.phi_e(out)

class NodeModel(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_x_out=None):
        if f_x_out is None:
            f_x_out = f_x
        super(NodeModel, self).__init__()
        self.phi_x = model_fn(f_e + f_x + f_u, f_x_out)

    def forward(self, x, edge_index, e, u, batch):
        if not len(e):
            return 
        src, dest = edge_index
        # add nodes with the same dest
        e_agg_node = scatter_add(e, dest, dim=0)
        out = torch.cat([x, e_agg_node, u[batch]], 1)
        return self.phi_x(out)

class NodeModel_A(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_x_out=None):
        if f_x_out is None:
            f_x_out = f_x
        super(NodeModel_A, self).__init__()
        self.phi_x = model_fn(f_e + f_x + f_u, f_x_out)

    def forward(self, x, edge_index, e, u, batch):
        if not len(e):
            return 
        src, dest = edge_index
        # TODO : attention on edges
        a = ...
        # add nodes with the same dest
        e_agg_node = scatter_add(a * e, dest, dim=0)
        out = torch.cat([x, e_agg_node, u[batch]], 1)
        return self.phi_x(out)

class GlobalModel(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_u_out=None):
        super(GlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_e + f_x + f_u, f_u_out)

    def forward(self, x, edge_index, e, u, batch):
        src, dest = edge_index
        # compute the batch index for all edges
        e_batch = batch[src]
        # aggregate all edges in the graph
        e_agg = scatter_add(e, e_batch, dim=0)
        # aggregate all nodes in the graph
        x_agg = scatter_add(x, batch, dim=0)
        out = torch.cat([x_agg, e_agg, u], 1)
        return self.phi_u(out)

class GlobalModel_A(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_u_out=None):
        super(GlobalModel_A, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_e + f_x + f_u, f_u_out)

    def forward(self, x, edge_index, e, u, batch):
        src, dest = edge_index
        # compute the batch index for all edges
        e_batch = batch[src]
        # TODO : edge attention
        e_a = ...
        # aggregate all edges in the graph
        e_agg = scatter_add(a_e * e, e_batch, dim=0)
        # TODO : node attention
        x_a = ...
        # aggregate all nodes in the graph
        x_agg = scatter_add(x_a * x, batch, dim=0)
        out = torch.cat([x_agg, e_agg, u], 1)
        return self.phi_u(out)

class GlobalModel_NodeOnly(torch.nn.Module):
    def __init__(self,
                 f_x,
                 f_u,
                 model_fn,
                 f_u_out=None):
        super(GlobalModel_NodeOnly, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_x + f_u, f_u_out)

    def forward(self, x, edge_index, e, u, batch)
        src, dest = edge_index
        # aggregate all nodes in the graph
        x_agg = scatter_add(x, batch, dim=0)
        out = torch.cat([x_agg, u], 1)
        return self.phi_u(out)

class GlobalModel_NodeOnly_A(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_u_out=None):
        super(GlobalModel_NodeOnly_A, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_x + f_u, f_u_out)

    def forward(self, x, edge_index, e, u, batch):
        src, dest = edge_index
        # TODO : node attention
        a = ...
        # aggregate all nodes in the graph
        x_agg = scatter_add(a * x, batch, dim=0)
        out = torch.cat([x_agg, u], 1)
        return self.phi_u(out)

# GNN Layers

class DeepSet(torch.nn.Module):
    """
    Deep Set.
    """
    def __init__(self,
                 node_model,
                 global_model):
        super(DeepSet, self).__init__()
        self.node_model = node_model
        self.global_model = global_model

    def forward(self, x, u, batch):
        x = self.node_model(x, u, batch)
        u = self.global_model(x, u, batch)
        return x, u

class N_GNN(torch.nn.Module):
    """
    GNN layer, with no edge features.

    Messages are computed for the edges according to an edge model, but they
    are not kept into memory.
    """
    def __init__(self, edge_model, node_model, global_model):
        super(N_GNN, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, u, batch):
        row, col = edge_index
        e = self.edge_model(x[row], x[col], u, batch)
        x = self.node_model(x, edge_index, e, u, batch)
        u = self.global_model(x, edge_index, e, u, batch)
        return x, u


class GNN(torch.nn.Module):
    """
    GN block.

    Based on rusty1s' Metalayer : 

    https://github.com/rusty1s/pytorch_geometric
    """
    def __init__(self, edge_model, node_model, global_model):
        super(GNN, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, e, u, batch):
        row, col = edge_index
        e = self.edge_model(x[row], x[col], e, u, batch)
        x = self.node_model(x, edge_index, e, u, batch)
        u = self.global_model(x, edge_index, e, u, batch)
        return x, e, u

    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)
