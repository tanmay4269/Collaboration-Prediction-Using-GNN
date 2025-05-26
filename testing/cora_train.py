import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.metrics import roc_auc_score

# 1. Load dataset and split edges
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)
print("1. Dataset loaded")

# 2. GCN encoder
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# 3. Link predictor: dot product
def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

# 4. Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_features, 64).to(device)
x, train_pos = data.x.to(device), data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("2. Train helpers loaded")

print("3. Gonna start training now:")
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos)
    neg_edge_index = negative_sampling(
        edge_index=train_pos, num_nodes=data.num_nodes,
        num_neg_samples=train_pos.size(1)
    )

    pos_score = decode(z, train_pos)
    neg_score = decode(z, neg_edge_index)
    score = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(device)
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels)
    loss.backward()
    optimizer.step()

    # AUC logging
    if epoch % 10 == 0:
        with torch.no_grad():
            pred = torch.sigmoid(score)
            auc = roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy())
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train AUC: {auc:.4f}")

print("4. Testing the model on held out data")
model.eval()
with torch.no_grad():
    z = model(x, train_pos)

    pos_test = decode(z, data.test_pos_edge_index.to(device))
    neg_test = decode(z, data.test_neg_edge_index.to(device))
    
    scores = torch.cat([pos_test, neg_test])
    labels = torch.cat([
        torch.ones(pos_test.size(0)),
        torch.zeros(neg_test.size(0))
    ]).to(device)
    
    pred = torch.sigmoid(scores)
    test_auc = roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy())
    print(f"\nTest AUC: {test_auc:.4f}")
