"""
Cleaner version of the graph_nets module.
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU
from torch.nn import Sigmoid, LayerNorm, Dropout
from torch.nn import BatchNorm1d

try:
    from torch_geometric.data import Data
    from torch_scatter import scatter_mean
    from torch_scatter import scatter_add
except:
    from scatter import scatter_mean
    from scatter import scatter_add
    from utils import Data

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

# base aggregation (sum)

class SumAggreg():
    def __init__(self):
        pass

    def __call__(self, x, batch):
        return scatter_add(x, batch, dim=0)

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
        return self.phi_x(torch.cat([x, u[batch]], 1))

class DS_GlobalModel(torch.nn.Module):
    def  __init__(self,
                  f_x,
                  f_u,
                  model_fn,
                  f_u_out=None):
        super(DS_GlobalModel, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_x + f_u, f_u_out)

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
                  h,
                  model_fn,
                  f_u_out=None):
        super(DS_GlobalModel_A, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.h = h
        self.phi_u = model_fn(f_x + f_u, f_u_out)
        self.phi_k = Linear(f_x, h)
        self.phi_q = Linear(f_u, h)

    def forward(self, x, u, batch):
        k = self.phi_k(x)
        q = self.phi_q(u)[batch]
        a = torch.sigmoid(
            torch.bmm(k.view(-1, 1, self.h),
                      q.view(-1, self.h, 1)).squeeze(1))
        x_agg = scatter_add(a * x, batch, dim=0)
        return self.phi_u(torch.cat([x_agg, u], 1))

# Edge, Node and Global models for GNNs

class EdgeModel(torch.nn.Module):
    """
    Concat. Try also Diff ?
    """
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_e_out=None):
        super(EdgeModel, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
        self.phi_e = model_fn(f_e + 2*f_x + f_u, f_e_out)

    def forward(self, src, dest, e, u, batch):
        out = torch.cat([src, dest, e, u[batch]], 1)
        return self.phi_e(out)

class EdgeModel_NoMem(torch.nn.Module):
    """
    Concat. Try also Diff ?
    """
    def __init__(self,
                 f_x,
                 f_u,
                 model_fn,
                 f_e_out=None):
        super(EdgeModel_NoMem, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
        self.phi_e = model_fn(2*f_x + f_u, f_e_out)

    def forward(self, src, dest, u, batch):
        out = torch.cat([src, dest, u[batch]], 1)
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
                 h,
                 model_fn,
                 f_x_out=None):
        if f_x_out is None:
            f_x_out = f_x
        self.h = h
        super(NodeModel_A, self).__init__()
        self.phi_k = Linear(f_x, h)
        self.phi_q = Linear(f_x, h)
        self.phi_x = model_fn(f_e + f_x + f_u, f_x_out)

    def forward(self, x, edge_index, e, u, batch):
        if not len(e):
            return 
        src, dest = edge_index
        # attention on edges
        k = self.phi_k(x)[src]
        q = self.phi_q(x)[dest]
        # batch-wise dot product
        a = torch.sigmoid(
            torch.bmm(k.view(-1, 1, self.h),
                      q.view(-1, self.h, 1)).squeeze(1))
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
                 h,
                 model_fn,
                 f_u_out=None):
        super(GlobalModel_A, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.h = h
        self.phi_u = model_fn(f_e + f_x + f_u, f_u_out)
        self.phi_k_e = Linear(f_e, h)
        self.phi_q_e = Linear(f_u, h)
        self.phi_k_x = Linear(f_x, h)
        self.phi_q_x = Linear(f_u, h)

    def forward(self, x, edge_index, e, u, batch):
        src, dest = edge_index
        # compute the batch index for all edges
        e_batch = batch[src]
        # TODO : edge attention
        k = self.phi_k_e(e)
        q = self.phi_q_e(u)[e_batch]
        a_e = torch.sigmoid(
            torch.bmm(k.view(-1, 1, self.h),
                      q.view(-1, self.h, 1)).squeeze(1))
        # aggregate all edges in the graph
        e_agg = scatter_add(a_e * e, e_batch, dim=0)
        # TODO : node attention
        k = self.phi_k_x(x)
        q = self.phi_q_x(u)[batch]
        x_a = torch.sigmoid(
            torch.bmm(k.view(-1, 1, self.h),
                      q.view(-1, self.h, 1)).squeeze(1))
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

    def forward(self, x, edge_index, e, u, batch):
        src, dest = edge_index
        # aggregate all nodes in the graph
        x_agg = scatter_add(x, batch, dim=0)
        out = torch.cat([x_agg, u], 1)
        return self.phi_u(out)

class GlobalModel_NodeOnly_A(torch.nn.Module):
    def __init__(self,
                 f_x,
                 f_u,
                 h,
                 model_fn,
                 f_u_out=None):
        super(GlobalModel_NodeOnly_A, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.h = h
        self.phi_u = model_fn(f_x + f_u, f_u_out)
        self.phi_k = Linear(f_x, h) 
        self.phi_q = Linear(f_u, h) 

    def forward(self, x, edge_index, e, u, batch):
        src, dest = edge_index
        # node attention
        k = self.phi_k(x)
        q = self.phi_q(u)[batch]
        a = torch.sigmoid(
            torch.bmm(k.view(-1, 1, self.h),
                      q.view(-1, self.h, 1)).squeeze(1))
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
        src, dest = edge_index
        e = self.edge_model(x[src], x[dest], u, batch[src])
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
        src, dest = edge_index
        e = self.edge_model(x[src], x[dest], e, u, batch[src])
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

class SelfAttention(torch.nn.Module):
    """
    """
    def __init__(self, f_in, h_dim):
        super(SelfAttention, self).__init__()
        self.h = h_dim
        self.phi_q = Linear(f_in, h_dim)
        self.phi_k = Linear(f_in, h_dim)
        self.phi_v = Linear(f_in, h_dim)
        # self.phi_x = Linear(h_dim, f_in) # to map back on x's n_dims
        # no normalization ?

    def forward(self, x, edge_index, batch):
        # e_i makes sure all that what happens in a batch stays in a batch
        # we need self loops for this model
        src, dest = edge_index
        q = self.phi_q(x)[dest]
        k = self.phi_k(x)[src]
        v = self.phi_v(x)[src]
        prod = torch.bmm(k.view(-1, 1, self.h), q.view(-1, self.h, 1))
        prod = prod.squeeze(1)
        prod /= np.sqrt(self.h)
        # softmax
        exp = torch.exp(prod)
        a = exp / (scatter_add(exp, batch[src], dim=0)[batch[src]] + 10e-7)
        # a = torch.sigmoid(prod) # softmax here !!!
        v = v * a # attention weighting
        x = scatter_add(v, dest, dim=0)
        return x

class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention layer.
    Not an optimal implem, the heads don't run in parallel, fix this one day.
    """
    def __init__(self, f_in, n_heads, h_dim):
        super(MultiHeadAttention, self).__init__()
        self.modlist = torch.nn.ModuleList() # to hold the heads
        for _ in range(n_heads):
            self.modlist.append(SelfAttention(f_in, h_dim))
        self.phi_x = Linear(n_heads * h_dim, f_in)
        # maybe add a normalization layer, and a residual connexion

    def forward(self, x, edge_index, batch):
        tlist = []
        for mod in self.modlist:
            tlist.append(mod(x, edge_index, batch))
        for t in tlist:
            print(t.shape)
        x = torch.cat(tlist, 1) # residual connexion
        return self.phi_x(x)