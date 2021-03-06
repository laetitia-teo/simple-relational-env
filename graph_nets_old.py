"""
Small library for creating arbitrary Graph Networks.
"""

import torch
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU
from torch.nn import Sigmoid, LayerNorm, Dropout
from torch.nn import BatchNorm1d

try:
    from torch_geometric.data import Data
    from torch_scatter import scatter_mean
    from torch_scatter import scatter_add
    # from torch_geometric.nn import MetaLayer
except:
    from utils import Data
    from scatter import scatter_add, scatter_mean

from utils import cosine_similarity
from utils import cos_sim
from utils import sim

###############################################################################
#                                                                             #
#                               MLP function                                  #
#                                                                             #
###############################################################################

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

class EdgeModelConcatNoMem(torch.nn.Module):
    def __init__(self,
                 f_e,
                 f_x,
                 f_u,
                 model_fn,
                 f_e_out=None):
        super(EdgeModelConcatNoMem, self).__init__()
        if f_e_out is None:
            f_e_out = f_e
        self.phi_e = model_fn(2 * f_x + f_u, f_e_out)

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([dest, src, u[batch]], 1)
        return self.phi_e(out)

class TGNNEdge(torch.nn.Module):
    """
    Edge model of the TGNN model.
    It computes the query and key vectors for each node, and then uses their
    dot product across node pairs as edge attributes.

    No learnable parameters.
    """
    def __init__(self,
                 f_x,
                 f_u,
                 f_int):
        """
        Arguments: 
            - f_x : number of node features;
            - f_u : number of global features;
            - f_int : number of features of the intermediate vectors. 
        """
        super(TGNNEdge, self).__init__()
        if f_e_out is None:
            f_e_out = f_e

    def forward(self, src, dest, edge_attr, u, batch):
        K, Q = src, dest
        a = torch.bmm(K.view(-1, 1, f_int), Q.view(-1, 1, f_int)).squeeze()
        return a # new edge_attr

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

    def __call__(self, x, x_src, cg_edge_index, batch_src):
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
        # attentions
        a = ecs / (scatter_add(ecs, batch_src[src])[batch_src[src]])
        vec = x[dest] - x_src[src] # to be multiplied by attentions
        mu = (a * vec.T)
        return scatter_add(mu, dest).T # see if we don't need mean instead here

class LearnedCrossGraphAttention(torch.nn.Module):
    """
    Class for computing scalar attentions between the nodes of two different
    graphs.
    """
    def __init__(self,
                 f_x,
                 model_fn):
        """
        This cross-graph attention function, to be used with a model similar
        to the GraphMatching model, uses the node features of the source
        (other) graph and the destination (current) graph to compute scalar
        attentions that will be multiplied with the destination node features.
        The computation of the attentions is done with a mlp.

        The output of the model is a vector of size [X_dest], the number of 
        destination graph nodes (on all batches).
        """
        super(LearnedCrossGraphAttention, self).__init__()
        self.mlp = model_fn(2 * f_x , 1)

    def forward(self, x, x_src, cg_edge_index, batch, batch_src):
        src, dest = cg_edge_index
        # attentions
        a = self.mlp(torch.cat([x[dest], x_src[src]], 1))
        a = scatter_add(a, dest, dim=0)
        # get the attentions to the [0, 1] range after summing them over
        # all destination nodes
        a = torch.sigmoid(a)
        return a

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
        super(CosineSimNodeModel, self).__init__()
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
        if not len(edge_attr):
            return 
        src, dest = edge_index
        # aggregate all edges which have the same destination
        e_agg_node = scatter_mean(edge_attr, dest, dim=0)
        # print(src.shape)
        # print()
        out = torch.cat([x, e_agg_node, u[batch]], 1)
        return self.phi_x(out)

class NodeModelAdd(torch.nn.Module):
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
        super(NodeModelAdd, self).__init__()
        self.phi_x = model_fn(f_e + f_x + f_u, f_x_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        """
        if not len(edge_attr):
            return 
        src, dest = edge_index
        # aggregate all edges which have the same destination
        e_agg_node = scatter_add(edge_attr, dest, dim=0)
        # print(src.shape)
        # print()
        out = torch.cat([x, e_agg_node, u[batch]], 1)
        return self.phi_x(out)

class TGNNNode(torch.nn.Module):
    """
    Node model for the TGNN.
    """
    def __init__(self,
                 f_x,
                 f_u,
                 f_out):
        if f_x_out is None:
            f_x_out = f_x
        super(TGNNNode, self).__init__()
        self.phi_K = Linear(f_x + f_u, f_out)
        self.phi_Q = Linear(f_x + f_u, f_out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        a = edge_attr
        src, dest = edge_index
        # how to keep the information ?
        x = scatter_add(a * x[src], dest, dim=0)
        K = self.phi_K(torch.cat([x, u[batch]], 1))
        Q = self.phi_Q(torch.cat([x, u[batch]], 1))

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

class GlobalModelAdd(torch.nn.Module):
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
        super(GlobalModelAdd, self).__init__()
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
        e_agg = scatter_add(edge_attr, e_batch, dim=0)
        # aggregate all nodes in the graph
        x_agg = scatter_add(x, batch, dim=0)
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

class GlobalModelNodeAttention(torch.nn.Module):
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
        super(GlobalModelNodeAttention, self).__init__()
        if f_u_out is None:
            f_u_out = f_u
        self.phi_u = model_fn(f_x + f_u, f_u_out)
        self.gating = model_fn(f_x + f_u, f_x) # use sth simpler maybe

    def forward(self,  x, edge_index, edge_attr, u, batch):
        # attentions
        a = self.gating(torch.cat([x, u[batch]], 1))
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
        super(CosineAttentionLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model # this needs a specific node model
        self.global_model = global_model

        self.attention_function = attention_function

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
                batch,
                batch_src):
        """
        Similar to MetaLayer, but has two additional terms, x_src and 
        cg_edge_index.
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
        a = self.attention_function(x, x_src, cg_edge_index, batch_src)
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

class CrossGraphAttentionLayer(torch.nn.Module):
    """
    This GNN layer computes scalar attentions between every node pair from
    graph1 and graph2 respectively, and multiplies it by the node features 
    before the edge and node models.

    Note that the cross-graph attentions are applied to the node features
    before any edge message-passing is done.
    """
    def __init__(self,
                 attention_function,
                 edge_model,
                 node_model,
                 global_model):
        """
        Initializes the layer. 
        """
        super(CrossGraphAttentionLayer, self).__init__()

        self.attention_function = attention_function

        self.edge_model = edge_model
        self.node_model = node_model # this needs a specific node model
        self.global_model = global_model

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
                batch,
                batch_src):
        """
        Forward pass.
        Similar to the usual MetaLayer, but has additional inputs for the
        source graph node features (x_src), the source graph batch (batch_src)
        and the cross-graph edge index tensor (cg_edge_index).
        """
        row, col = edge_index
        a = self.attention_function(x, x_src, cg_edge_index, batch, batch_src)
        x = x * a
        edge_attr = self.edge_model(x[row],
                                    x[col],
                                    edge_attr,
                                    u,
                                    batch[row])

        # this cross-graph function modulates the node features of the current
        # graph
        x = self.node_model(x, edge_index, edge_attr, u, batch)
        u = self.global_model(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u

class TGNNLayer(torch.nn.Module):
    """
    TGNN Layer.
    """
    def __init__(self):
        pass