"""
New module for GNN models.
"""
import time
import numpy as np
import torch

import graph_nets_v2 as gn

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
        super(GraphModel, self).cuda()
        self.GPU = True
        self.data_from_graph = data_from_graph_maker(cuda=True)

    def cpu(self):
        super(GraphModel, self).cpu()
        self.GPU = False
        self.data_from_graph = data_from_graph_maker(cuda=False)

class GraphModelSimple(GraphModel):
    """Single-input graph model"""
    def __init__(self, f_dict):
        super(GraphModelSimple, self).__init__(f_dict)

class GraphModelDouble(GraphModel):
    """Double-input graph model"""
    def __init__(self, f_dict):
        super(GraphModelDouble, self).__init__(f_dict)

# deep sets

class DeepSet(GraphModelSimple):
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(DeepSet, self).__init__(f_dict)
        mlp_fn = gn.mlp_fn(mlp_layers)
        self.deepset = gn.DeepSet(mlp_fn, self.fx, self.h, self.fout)

    def forward(self, graph):
        x, _, _, _, batch = self.data_from_graph(graph)
        return self.deepset(x, batch)

class DeepSetPlus(GraphModelSimple):
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(DeepSetPlus, self).__init__(f_dict)
        self.N = N # we allow multiple rounds
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.deepset = gn.DeepSetPlus(
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

class DeepSetPlus_A(GraphModelSimple):
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(DeepSetPlus_A, self).__init__(f_dict)
        self.N = N # we allow multiple rounds
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.deepset = gn.DeepSetPlus(
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

class N_GNN(GraphModelSimple):
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

class N_GNN_A(GraphModelSimple):
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

class GNN_NAgg(GraphModelSimple):
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

class GNN_NAgg_A(GraphModelSimple):
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

class GNN_NEAgg(GraphModelSimple):
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

class GNN_NEAgg_A(GraphModelSimple):
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

# other models

class TGNN(GraphModelSimple):
    """
    Transformer-GNN, the nodes do a transformer-style aggregation on their
    neighbours.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(TGNN, self).__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)
        self.tgnn = gn.MultiHeadAttention(self.fx, 8, self.h)        
        self.agg = gn.SumAggreg()

    def forward(self, graph):
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        # out list ?
        for _ in range(self.N):
            x = self.tgnn(x, edge_index, batch)
        return self.agg(x, batch)

# Double-input graph models



# edge aggregation ? does it make sense ?

# Graph model utilities

model_list = [
    DeepSetPlus,
    DeepSetPlus_A,
    N_GNN,
    N_GNN_A,
    GNN_NAgg,
    GNN_NAgg_A,
    GNN_NEAgg,
    GNN_NEAgg_A,
    TGNN,
    DeepSet]

model_names = [
    'Deep Set++ (0)',
    'Deep Set++, attention (1)',
    'Node GNN (2)',
    'Node GNN, attention (3)',
    'GNN, node aggreg (4)',
    'GNN, node aggreg, attention (5)',
    'GNN, node-edge aggreg (6)',
    'GNN, node-edge aggreg, attention (7)',
    'TGNN (8)',
    'Deep Set (9)'
]