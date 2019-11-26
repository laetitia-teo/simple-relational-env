###############################################################################
#                                                                             #
#                            Utility functions                                #
#                                                                             #
###############################################################################

import numpy as np
import torch

from torch_geometric.data import Data

from torch_scatter import scatter_mean

DTYPE = torch.float

def complete_edge_index(n, self_edges=True):
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
    if not self_edges:
        ei = ei[:, ei[0] != ei[1]]
    return ei

def complete_ei(n, m=None):
    """
    This function creates a set of complete edges (via its edge_index tensor)
    between nodes of two graphs of respective number of nodes n and m.
    If m is unspecified, the second graph is considered to have the same number
    of nodes as the first
    """
    en = torch.ones((n, m)).long() * torch.arange(n).unsqueeze(1)
    if m is None:
        em = en
    else:
        em = torch.ones((m, n)).long() * torch.arange(m).unsqueeze(1)
    return torch.stack((
        torch.reshape(en, (-1,)), torch.reshape(em.T, (-1,))))

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

    Use this for data from the SimpleTask Dataloader.
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

def data_to_graph_parts(data):
    """
    Converts the data yielded by the PartsDataset DataLoader into graph form
    for input of the graph models.

    The PartsDataset yields a list of targets (nodes of the first graph),
    t_batch (batch of the first graph), refs (second graph), r_batch (batch
    of the second graph) and labels (not used here).

    The function creates edges (complete) and global vectors for each 
    scene graph in the target and reference batches.
    """
    x1, x2, labels, batch1, batch2 = data
    f_x = x1.shape[-1]
    # create edges for graph1
    ei1 = torch.zeros((2, 0), dtype=torch.long)
    n = batch1[-1] + 1 # number of graphs, same as batch size
    count = 0 # for counting node index offset
    for i in range(n):
        idx = (batch1 == i).nonzero(as_tuple=True)[0]
        n_x = len(idx)
        # create edge index
        ei = complete_edge_index(n_x, self_edges=False)
        ei += count
        ei1 = torch.cat((ei1, ei), 1)
        count += n_x
    e1 = x1[ei1[1]] - x1[ei1[0]]
    # create edges for graph2
    ei2 = torch.zeros((2, 0), dtype=torch.long)
    n = batch2[-1] + 1 # number of graphs, same as batch size
    count = 0 # for counting node index offset
    for i in range(n):
        idx = (batch2 == i).nonzero(as_tuple=True)[0]
        n_x = len(idx)
        # create edge index
        ei = complete_edge_index(n_x, self_edges=False)
        ei += count
        ei2 = torch.cat((ei2, ei), 1)
        count += n_x
    e2 = x2[ei2[1]] - x2[ei2[0]]
    # create globals by averaging nodes in the same graph
    u1 = scatter_mean(x1, batch1, dim=0)
    u2 = scatter_mean(x2, batch2, dim=0)
    # build graphs
    graph1 = Data(x=x1, edge_index=ei1, edge_attr=e1, y=u1, batch=batch1)
    graph2 = Data(x=x2, edge_index=ei2, edge_attr=e2, y=u2, batch=batch2)
    return graph1, graph2

def state_list_to_graph(state_list):
    """
    Transforms a single state list into a fully connected graphs.
    """
    x1 = torch.tensor(state_list, dtype=DTYPE)
    batch1 = torch.zeros(len(state_list), dtype=int)
    f_x = x1.shape[-1]
    # create edges for graph1
    ei1 = torch.zeros((2, 0), dtype=torch.long)
    n = batch1[-1] + 1 # number of graphs, same as batch size
    count = 0 # for counting node index offset
    for i in range(n):
        idx = (batch1 == i).nonzero(as_tuple=True)[0]
        n_x = len(idx)
        # create edge index
        ei = complete_edge_index(n_x, self_edges=False)
        ei += count
        ei1 = torch.cat((ei1, ei), 1)
        count += n_x
    e1 = x1[ei1[1]] - x1[ei1[0]]
    u1 = scatter_mean(x1, batch1, dim=0)
    graph = Data(x=x1, edge_index=ei1, edge_attr=e1, y=u1, batch=batch1)
    return graph

def merge_graphs(g_list):
    """
    Merge graphs.
    """
    pass

def data_from_graph(graph):
    x = graph.x
    edge_index = graph.edge_index
    e = graph.edge_attr
    u = graph.y
    batch = graph.batch
    return x, edge_index, e, u, batch

def cross_graph_ei(batch1, batch2):
    """
    Creates the cross-graph edge index for connecting both graphs.
    """
    bsize = batch1[-1] + 1

    cg_ei = torch.zeros((2, 0), dtype=torch.long)
    count1 = 0 # for keeping track of node offset
    count2 = 0
    for i in range(bsize):
        idx1 = (batch1 == i).nonzero(as_tuple=True)[0]
        idx2 = (batch2 == i).nonzero(as_tuple=True)[0]
        n_x1 = len(idx1)
        n_x2 = len(idx2)
        # create edge index
        ei = complete_ei(n_x1, n_x2)
        # offset the node indices
        ei[0] += count1
        ei[1] += count2
        # concatenate to complete edge_index
        cg_ei = torch.cat((cg_ei, ei), 1)
        count1 += n_x1
        count2 += n_x2

    return cg_ei