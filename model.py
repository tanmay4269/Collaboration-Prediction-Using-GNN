import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, **kwargs):
        super(EdgeSAGEConv, self).__init__(aggr='mean', **kwargs)
        self.lin_self = Linear(in_channels, out_channels)
        self.lin_neigh = Linear(in_channels + edge_dim, out_channels)
        self.edge_dim = edge_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_self.reset_parameters()
        self.lin_neigh.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        num_original_edges = edge_index.size(1)
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        num_self_loops_added = edge_index_with_self_loops.size(1) - num_original_edges

        if num_self_loops_added > 0:
            self_loop_edge_attr = torch.zeros((num_self_loops_added, self.edge_dim),
                                               dtype=edge_attr.dtype, device=edge_attr.device)
            edge_attr_expanded = torch.cat([edge_attr, self_loop_edge_attr], dim=0)
        else:
            edge_attr_expanded = edge_attr

        out = self.propagate(edge_index_with_self_loops, x=x, edge_attr=edge_attr_expanded)
        return out

    def message(self, x_j, edge_attr):
        combined_features = torch.cat([x_j, edge_attr], dim=-1)
        return self.lin_neigh(combined_features)

    def update(self, aggr_out, x):
        x_self = self.lin_self(x)
        updated_features = x_self + aggr_out
        return F.relu(updated_features)


class LinkPredictionModel(torch.nn.Module):
    def __init__(self, in_channels_node, in_channels_edge, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = EdgeSAGEConv(in_channels_node, hidden_channels, in_channels_edge)
        self.conv2 = EdgeSAGEConv(hidden_channels, out_channels, in_channels_edge)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return x

    def decode(self, z, edge_indices):
        source_node_embeddings = z[edge_indices[0]] 
        target_node_embeddings = z[edge_indices[1]] 

        scores = F.cosine_similarity(source_node_embeddings, target_node_embeddings, dim=-1)
        return scores
