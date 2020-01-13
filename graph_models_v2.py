"""
New module for GNN models.
"""
import time
import numpy as np
import torch
import torch_geometric

import graph_nets_v2 as gn

from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data

from graph_utils import data_from_graph_maker
from graph_utils import cross_graph_ei_maker

class GraphModel(torch.nn.Module):
    def __init__(self, f_dict):
        super(GraphModel, self).__init__()
        # maybe define different attributes for a simple-input GM and a double-
        # input GM.
        f_e, f_x, f_u, h, f_out = self.get_features(f_dict)
        self.fe = f_e
        self.fx = f_x
        self.fu = f_u
        self.h = h
        self.fout = f_out
        self.GPU = False

        self.data_from_graph = data_from_graph_maker()

    def get_features(self, f_dict):
        """
        Gets the input and output features for graph processing.
        """
        f_e = f_dict['f_e']
        f_x = f_dict['f_x']
        f_u = f_dict['f_u']
        h = f_dict['h']
        f_out = f_dict['f_out']
        return f_e, f_x, f_u, h, f_out

    def cuda(self):
        super(GraphModelDouble, self).cuda()
        self.GPU = True
        self.data_from_graph = data_from_graph_maker(cuda=True)

    def cpu(self):
        super(GraphModelDouble, self).cpu()
        self.GPU = False
        self.data_from_graph = data_from_graph_maker(cuda=False)

# deep sets

class DeepSet(GraphModel):
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(DeepSet, self).__init__(f_dict)
        self.N = N # we allow multiple rounds
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.deepset = gn.DeepSet(
            gn.DS_NodeModel(self.fx, self.fu, mlp_fn, self.fx),
            gn.DS_GlobalModel(self.fx, self.fu, mlp_fn, self.fu))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, graph):
        x, _, _, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, u = self.deepset(x, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class DeepSet_A(GraphModel):
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(DeepSet_A, self).__init__(f_dict)
        self.N = N # we allow multiple rounds
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.deepset = gn.DeepSet(
            gn.DS_NodeModel(self.fx, self.fu, mlp_fn, self.fx),
            gn.DS_GlobalModel_A(self.fx, self.fu, self.h, mlp_fn, self.fu))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, graph):
        x, _, _, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, u = self.deepset(x, u, batch)
            out_list.append(self.mlp(u))
        return out_list

# GNNs

# Node only

class N_GNN(GraphModel):
    """
    Node-GNN. (No edge features)
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(N_GNN, self).__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.N_GNN(
            gn.EdgeModel_NoMem(self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, graph):
        x, edge_index, _, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, u = self.gnn(x, edge_index, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class N_GNN_A(GraphModel):
    """
    Node-GNN, with attention in node and global aggregation.
    """
    def __init__(self, 
                 mlp_layers,
                 N,
                 f_dict):
        super(N_GNN_A, self).__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.N_GNN(
            gn.EdgeModel_NoMem(self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel_A(self.fe, self.fx, self.fu, self.h, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly_A(self.fx, self.fu, self.h, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, graph):
        x, edge_index, _, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, u = self.gnn(x, edge_index, u, batch)
            out_list.append(self.mlp(u))
        return out_list

# With edge features

class GNN_NAgg(GraphModel):
    """
    Edge-feature GNN, with node aggregation in the global model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(GNN_NAgg, self).__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, graph):
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class GNN_NAgg_A(GraphModel):
    """
    Edge-feature GNN, with node aggregation in the global model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(GNN_NAgg_A, self).__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel_A(self.fe, self.fx, self.fu, self.h, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly_A(self.fx, self.fu, self.h, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, graph):
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class GNN_NEAgg(GraphModel):
    """
    Edge-feature GNN, with node aggregation in the global model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(GNN_NEAgg, self).__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu, mlp_fn, self.fx),
            gn.GlobalModel(self.fe, self.fx, self.fu, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, graph):
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class GNN_NEAgg_A(GraphModel):
    """
    Edge-feature GNN, with node aggregation in the global model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(GNN_NEAgg_A, self).__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel_A(self.fe, self.fx, self.fu, self.h, mlp_fn, self.fx),
            gn.GlobalModel_A(self.fe, self.fx, self.fu, self.h, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, graph):
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list

# edge aggregation ? does it make sense ?

# Graph model utilities

model_list = [
    DeepSet,
    DeepSet_A,
    N_GNN,
    N_GNN_A,
    GNN_NAgg,
    GNN_NAgg_A,
    GNN_NEAgg,
    GNN_NEAgg_A]