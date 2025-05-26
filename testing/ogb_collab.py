import torch
from torch_geometric.nn import SAGEConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling

# 1. Load dataset and extract splits
dataset = PygLinkPropPredDataset(name='ogbl-collab')
split_edge = dataset.get_edge_split()
data = dataset[0]

# 2. GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# 3. Train loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = data.x.to(device)
edge_index = data.edge_index.to(device)
model = GraphSAGE(x.size(-1), 128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
evaluator = Evaluator(name='ogbl-collab')

train_pos = split_edge['train']['edge'].t().to(device)

print("Starting training...")
for epoch in range(1, 101):
    print(f"Epoch #{epoch:03d}")
    model.train()
    optimizer.zero_grad()
    z = model(x, edge_index)

    # Positive and negative edges
    pos_score = decode(z, train_pos)
    neg_edge_index = negative_sampling(edge_index=edge_index,
                                       num_nodes=x.size(0),
                                       num_neg_samples=pos_score.size(0),
                                       method='sparse')
    neg_score = decode(z, neg_edge_index)

    # Loss
    score = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels)
    loss.backward()
    optimizer.step()

    # Evaluate on validation
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index)
            pos_valid = split_edge['valid']['edge'].t().to(device)
            neg_valid = split_edge['valid']['edge_neg'].t().to(device)
            pos_pred = decode(z, pos_valid).sigmoid()
            neg_pred = decode(z, neg_valid).sigmoid()
            y_pred = torch.cat([pos_pred, neg_pred])
            y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).to(device)

            input_dict = {'y_pred_pos': pos_pred.view(-1), 'y_pred_neg': neg_pred.view(-1)}
            hits = evaluator.eval(input_dict)['hits@50']
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Hits@50: {hits:.4f}')

# 4. Final test evaluation
print("\nTesting...")
model.eval()
with torch.no_grad():
    z = model(x, edge_index)
    pos_test = split_edge['test']['edge'].t().to(device)
    neg_test = split_edge['test']['edge_neg'].t().to(device)
    pos_pred = decode(z, pos_test).sigmoid()
    neg_pred = decode(z, neg_test).sigmoid()

    input_dict = {'y_pred_pos': pos_pred.view(-1), 'y_pred_neg': neg_pred.view(-1)}
    test_result = evaluator.eval(input_dict)
    print(f"Test Hits@50: {test_result['hits@50']:.4f}")
