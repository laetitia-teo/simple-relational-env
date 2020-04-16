"""
New module for GNN models.
"""
import time
import numpy as np
import torch

import graph_nets as gn

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

class Parallel(GraphModelDouble):
    """
    Parallel processing of inputs.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):

        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'MPGNN'

        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu,model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu,model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu,model_fn, self.fx))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu,model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu,model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu,model_fn, self.fx))
        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, u1, batch1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, u2, batch2)
            out_list.append(self.mlp(torch.cat([u1, u2], 1)))
        return out_list

class RecurrentGraphEmbedding(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(RecurrentGraphEmbedding, self).__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'MPGNN'

        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu,model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu,model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu,model_fn, self.fx))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, 2 * self.fu,model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, 2 * self.fu,model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, 2 * self.fu,model_fn, self.fx))
        self.mlp =model_fn(self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class AlternatingSimple(GraphModelDouble):
    """
    Simple version of the Alternating model.
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
        super(AlternatingSimple, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

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
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDouble(GraphModelDouble):
    """
    Different gnns inside.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(AlternatingDouble, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn1(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingSimpleRDS(GraphModelDouble):
    """
    RDS layer inside.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(AlternatingSimple, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'RDS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, 2 * self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, u1 = self.gnn(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn(x2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingSimplev2(GraphModelDouble):
    """
    Projects the input features into a higher-dimensional space.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(AlternatingSimplev2, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn = gn.GNN(
            gn.EdgeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, 2 * self.h, model_fn, self.h))
        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoublev2(GraphModelDouble):
    """
    Projects the input features into a higher-dimensional space.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(AlternatingDoublev2, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, 2 * self.h, model_fn, self.h))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, 2 * self.h, model_fn, self.h))
        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn1(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoubleRDS(GraphModelDouble):
    """
    Recurrent DeepSet version of the AlternatingDouble model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(AlternatingDoubleRDS, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'RDS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, 2 * self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, 2 * self.fu, model_fn, self.fu))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, 2 * self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, u1 = self.gnn1(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn2(x2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoubleRDSv2(GraphModelDouble):
    """
    Recurrent DeepSet version of the AlternatingDouble model with linear
    projection on h dimensions.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(AlternatingDoubleRDSv2, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'RDS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.h, 2 * self.h, model_fn, self.h),
            gn.DS_GlobalModel(self.h, 2 * self.h, model_fn, self.h))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.h, 2 * self.h, model_fn, self.h),
            gn.DS_GlobalModel(self.h, 2 * self.h, model_fn, self.h))

        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, u1 = self.gnn1(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn2(x2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class RecurrentGraphEmbeddingv2(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(RecurrentGraphEmbeddingv2, self).__init__(f_dict)
        self.N = N
        self.component = 'MPGNN'
        model_fn = gn.mlp_fn(mlp_layers)

        self.proj = torch.nn.Linear(self.fx, self.h)

        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.h, self.h, self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, self.h, model_fn, self.h))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, 2 * self.h, model_fn, self.h))
        self.mlp = model_fn(self.h, self.fout)

    def forward(self, graph1, graph2):
        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class RecurrentGraphEmbeddingRDS(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(RecurrentGraphEmbeddingRDS, self).__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'RDS'

        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, self.fu, model_fn, self.fu))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, 2 * self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, 2 * self.fu, model_fn, self.fu))
        self.mlp = model_fn(self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, u1 = self.gnn1(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn2(x2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class RecurrentGraphEmbeddingRDSv2(GraphModelDouble):
    """
    Cast to h dimensions, and use RDS layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(RecurrentGraphEmbeddingRDSv2, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'RDS'

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.h, self.h, model_fn, self.h),
            gn.DS_GlobalModel(self.h, self.h, model_fn, self.h))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.h, 2 * self.h, model_fn, self.h),
            gn.DS_GlobalModel(self.h, 2 * self.h, model_fn, self.h))
        self.mlp = model_fn(self.h, self.fout)

    def forward(self, graph1, graph2):
        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []
        for _ in range(self.N):
            x1, u1 = self.gnn1(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn2(x2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class ResAlternatingDouble(GraphModelDouble):
    """
    Different gnns inside.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(ResAlternatingDouble, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn1 = gn.GNN(
            gn.ResEdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.ResNodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.ResGlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))
        self.gnn2 = gn.GNN(
            gn.ResEdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.ResNodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.ResGlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn1(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            print('u {}'.format(u2.shape))
            print('e {}'.format(e2.shape))
            print('x {}'.format(x2.shape))
            x2, e2, u2 = self.gnn2(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class ResRecurrentGraphEmbedding(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(ResRecurrentGraphEmbedding, self).__init__(f_dict)
        self.N = N
        self.component = 'MPGNN'
        model_fn = gn.mlp_fn(mlp_layers)

        self.gnn1 = gn.GNN(
            gn.ResEdgeModel(self.fe, self.fx, self.fu, model_fn, self.fe),
            gn.ResNodeModel(self.fe, self.fx, self.fu, model_fn, self.fx),
            gn.ResGlobalModel_NodeOnly(self.fx, self.fu, model_fn, self.fx))
        self.gnn2 = gn.GNN(
            gn.ResEdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.ResNodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.ResGlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fx))
        self.mlp = model_fn(self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class AlternatingDoubleDS(GraphModelDouble):
    """
    Recurrent DeepSet version of the AlternatingDouble model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(AlternatingDoubleDS, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'DS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.ds1 = gn.DeepSet(model_fn, self.fu + self.fx, self.h, self.fx)
        self.ds2 = gn.DeepSet(model_fn, self.fu + self.fx, self.h, self.fx)

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            x1_ = torch.cat([x1, u2[batch2]], 1)
            u1 = self.ds1(x1_, batch1)
            x2_ = torch.cat([x2, u1[batch1]], 1)
            u2 = self.ds2(x2_, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoubleRSv2(GraphModelDouble):
    """
    Recurrent DeepSet version of the AlternatingDouble model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(AlternatingDoubleDSv2, self).__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'DS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.ds1 = gn.DeepSet(model_fn, 2 * self.h, self.h, self.h)
        self.ds2 = gn.DeepSet(model_fn, 2 * self.h, self.h, self.h)

        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, graph1, graph2):
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            x1_ = torch.cat([x1, u2[batch2]], 1)
            u1 = self.ds1(x1_, batch1)
            x2_ = torch.cat([x2, u1[batch1]], 1)
            u2 = self.ds2(x2_, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class RecurrentGraphEmbeddingDS(GraphModelDouble):
    """
    baseline.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(RecurrentGraphEmbeddingDS, self).__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'DS'

        self.ds1 = gn.DeepSet(model_fn, self.fx, self.h, self.fu)
        self.ds2 = gn.DeepSet(model_fn, self.fu + self.fx, self.h, self.fu)
        self.mlp = model_fn(self.fu, self.fout)

    def forward(self, graph1, graph2):
        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        u1 = self.ds1(x1, batch1)
        x2 = torch.cat([x2, u1[batch1]], 1)
        u2 = self.ds2(x2, batch2)
        out_list.append(self.mlp(u2))
        return out_list

class RecurrentGraphEmbeddingDSv2(GraphModelDouble):
    """
    baseline.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        super(RecurrentGraphEmbeddingDS, self).__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'DS'

        self.ds1 = gn.DeepSet(model_fn, self.h, self.h, self.h)
        self.ds2 = gn.DeepSet(model_fn, 2 * self.h, self.h, self.h)
        self.mlp = model_fn(self.h, self.fout)

    def forward(self, graph1, graph2):
        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)

        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []
        u1 = self.ds1(x1, batch1)
        x2 = torch.cat([x2, u1[batch1]], 1)
        u2 = self.ds2(x2, batch2)
        out_list.append(self.mlp(u2))
        return out_list

# Graph model utilities

model_list = [
    DeepSetPlus,
    GNN_NAgg,
    DeepSet]

model_list_double = [
    AlternatingDouble,
    AlternatingDoublev2,
    AlternatingDoubleRDS,
    RecurrentGraphEmbedding,
    RecurrentGraphEmbeddingv2,
    RecurrentGraphEmbeddingRDS,
    AlternatingDoubleRDSv2,
    RecurrentGraphEmbeddingRDSv2,
    AlternatingDoubleDS,
    RecurrentGraphEmbeddingDS]

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