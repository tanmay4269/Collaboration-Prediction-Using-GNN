import torch
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)  # Good 'ol dot product

class LinkPredictor(torch.nn.Module):
    def __init__(self, node_embedding_dim, edge_attr_dim):
        super().__init__()
        self.lin = torch.nn.Linear(2 * node_embedding_dim + edge_attr_dim, 1)

    def forward(self, z, edge_index, edge_attr=None):
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]

        if edge_attr is None:
            edge_attr = torch.zeros(z_i.size(0), self.lin.in_features - 2 * z.size(-1), device=z.device)
        
        x = torch.cat([z_i, z_j, edge_attr], dim=-1)
        return self.lin(x).view(-1)