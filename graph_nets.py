"""
Small library for creating arbitrary Graph Networks.
"""

import torch
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU
from torch.nn import Sigmoid, LayerNorm, Dropout

from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_scatter import scatter_add
from torch_geometric.nn import MetaLayer

from utils import cosine_similarity
from utils import cos_sim
from utils import sim

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
        # layers.append(LayerNorm(f_out))
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

class CosineAttention():
    """
    Class for computing the cosine similarity between nodes of two different
    graphs. Not a torch module, this has no learnable components.

    Used to Implement Graph Matching Networks.
    """
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 f_e_out=None):
        """
        This object serves for computing the cosine similarities between nodes
        of two graphs. Between node i of graph 1 and node j of graph 2, the
        attention is computed as the softmax of the cosine similarity between
        nodes i and j against the cosine similarities of nodes i' of graph 1
        and node j. Algorithmically; this is quite similar to computing edge
        attributes, except there are no learnable parameters, the function is
        fixed.
        """
        pass

    def __call__(self, x, x_src, cg_edge_index, batch):
        """
        No edge_attr and no u on this one.

        Arguments :

            - x (node feature tensor, size [X, f_x]) : node features of the
                current graph
            - x_src (node feature tensor, size [X, f_x]) : node features of the
                other graph
            - cg_edge_index (edge index tensor, size [2, E]) : cross-graph edge
                index tensor, mapping nodes of the other graph (source) to the 
                current graph (dest)
            - batch (batch tensor, size [X]) : batch tensor mapping the nodes
                of the other graph to their respective graph in the batch.

        Returns :
            - attentions (size [X], node size of the other graph.)
        """
        src, dest = cg_edge_index
        # exp-cosine-similarity vector
        ecs = torch.exp(sim(x_src[src], x[dest]))
        a = ecs / (scatter_add(ecs, batch[src])[batch[src]])
        a = a * x_src[src].T
        return scatter_add(a, batch[src]).T # see if we don't need mean instead here

class CosineSimNodeModel(torch.nn.Module):
    """
    Node model with cosine similarity attentions between nodes of 2 different
    graphs. Used to implement Graph Matching Networks.
    """
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_x_out=None):
        """
        Cosine Similarity Node model : this model performs the node updates
        in a similar fashion to the vanilla NodeModel, except that it takes
        as additional input the cross-graph attentions, coming from the other
        graph. These cross-graph attentions are computed by the
        CosineAttention function above.

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
        self.phi_x = model_fn(f_e + 2 * f_x + f_u, f_x_out)

    def forward(self, x, a, edge_index, edge_attr, u, batch):
        """
        """
        src, dest = edge_index
        # aggregate all edges which have the same destination
        e_agg_node = scatter_mean(edge_attr, dest, dim=0)
        out = torch.cat([x, a, e_agg_node, u[batch]], 1)
        return self.phi_x(out)

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
    def __init__(self,
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
    def __init__(self,
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

class NodeGlobalModelAttention(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_u_out=None):
        """
        This global model aggregates all node features by doing their 
        weighted mean, were the weights are computed by a gating (or attention)
        model.

        Arguments :

            - f_e (int): number of edge features
            - f_x (int): number of vertex features
            - f_u (int): number of global features
            - model_fn : function that takes input and output features and
                returns a model.
        """
        super(NodeGlobalModelAttention, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_x + f_u, f_u_out)
        self.gating = model_fn(f_x + f_u, f_x) # use sth simpler maybe

    def forward(self,  x, edge_index, edge_attr, u, batch):
        # attentions
        a = self.gating(torch.cat([x, u], 1))
        # aggregate attention-weighted nodes
        x_agg = scatter_mean(x * a, batch, dim=0)
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

class MetaLayer(torch.nn.Module):
    """
    GN block.

    Taken from rusty1s' PyTorch Geometric library, check it out here : 

    https://github.com/rusty1s/pytorch_geometric
    """
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()


    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
        """"""
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
                                        batch if batch is None else batch[row])

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u


    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)



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

class CosineAttentionLayer(torch.nn.Module):
    """
    Implementation of the Graph mAtching Network with cosine similarity used to
    compute attentions between node features.
    """
    def __init__(self,
                 edge_model,
                 attention_function,
                 node_model,
                 global_model):
        """
        Initializes the Cosine Attention Layer. 

        This layer is similar to the usual MetaLayer, with an additional 
        twist : the node feature update expects additional input from another
        graph, in the form of attentions between nodes of the current graph 
        and the nodes of the other graph. This additional input vector has the
        same size as number of nodes in the current graph.

        This vector contains the sum of source node features weighted by their
        attentions over the destination nodes. The attentions are computed
        as the ratio of the exp of cosine similarity of node i and j over the
        sum of exps of all cosine similarities between node i and j', node i 
        being the destination node of the current graph and nodes j and j' the
        source nodes.
        """
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model # this needs a specific node model
        self.global_model = global_model

        self.attention_function = attention_funtion

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()


    def forward(self,
                x,
                x_src,
                edge_index,
                cg_edge_index,
                edge_attr,
                u,
                batch):
        """
        Similar to MetaLayer, but has two additional terms, x_src and cg_edge_index.
        x_src is the tensor of node features of the second graph.
        cg_edge_index is the cross-graph connectivity (complete by default)
        """
        row, col = edge_index

        edge_attr = self.edge_model(x[row],
                                    x[col],
                                    edge_attr,
                                    u,
                                    batch if batch is None else batch[row])

        # maybe change the inputs for this, because it does not have the same 
        # format as the other functions
        a = self.attention_function(x, x_src, cg_edge_index, batch)
        x = self.node_model(x, a, edge_index, edge_attr, u, batch)

        u = self.global_model(x, edge_index, edge_attr, u, batch)

        return x, edge_attr, u


    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)