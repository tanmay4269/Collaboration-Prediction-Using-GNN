import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels_node, out_channels, in_channels_edge):
        super().__init__(aggr='mean')
        self.lin_neigh = nn.Linear(in_channels_node + in_channels_edge, out_channels)
        self.lin_self = nn.Linear(in_channels_node, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

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
        self.dropout = nn.Dropout(0.2)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr):
        # First layer 
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer 
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x

    def decode(self, z, edge_indices):
        source_node_embeddings = z[edge_indices[0]] 
        target_node_embeddings = z[edge_indices[1]] 
        
        scores = (source_node_embeddings * target_node_embeddings).sum(dim=-1)
        return scores