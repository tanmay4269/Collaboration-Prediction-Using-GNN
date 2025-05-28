import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from model import LinkPredictionModel
from dataset import OpenAlexGraphDataset

# Config
BASE_LR = 0.005
END_LR_FACTOR = 1.0
NUM_EPOCHS = 50
LOG_EVERY = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset_builder = OpenAlexGraphDataset(json_path="data/openalex_cs_papers.json", num_authors=-1, use_cache=False)
train_graph_data = dataset_builder.get_train_data()
val_graph_data = dataset_builder.get_val_data()
test_graph_data = dataset_builder.get_test_data()

# Move data to device
train_graph_data = train_graph_data.to(device)
val_graph_data = val_graph_data.to(device)
test_graph_data = test_graph_data.to(device)

# Pre-generate negative samples for validation and test (fixed across epochs)
val_neg_edge_index = negative_sampling(
    edge_index=train_graph_data.edge_index,  # Only use train edges
    num_nodes=train_graph_data.num_nodes,
    num_neg_samples=val_graph_data.edge_index.size(1),
    method='sparse'
).to(device)

test_neg_edge_index = negative_sampling(
    edge_index=torch.cat([train_graph_data.edge_index, val_graph_data.edge_index], dim=1),
    num_nodes=train_graph_data.num_nodes,
    num_neg_samples=test_graph_data.edge_index.size(1),
    method='sparse'
).to(device)

def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    
    # Get embeddings
    z = model(data.x, data.edge_index, data.edge_attr)
    
    # Positive edges
    pos_score = model.decode(z, data.edge_index)
    
    # Generate negative samples (only using train edges)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_score.size(0),
        method='sparse'
    )
    neg_score = model.decode(z, neg_edge_index)
    
    # Combine scores and labels
    score = torch.cat([pos_score, neg_score])
    labels = torch.cat([
        torch.ones(pos_score.size(0), device=device),
        torch.zeros(neg_score.size(0), device=device)
    ])
    
    # Compute loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(data, neg_edge_index, model, known_edges):
    model.eval()
    
    # Get embeddings using only known edges
    z = model(data.x, known_edges, data.edge_attr if hasattr(data, 'edge_attr') else None)
    
    # Predict on positive edges
    pos_pred = model.decode(z, data.edge_index)
    
    # Predict on negative edges
    neg_pred = model.decode(z, neg_edge_index)
    
    # Combine predictions
    y_pred = torch.cat([pos_pred, neg_pred]).cpu()
    y_true = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ]).cpu()
    
    return roc_auc_score(y_true, y_pred)

# Initialize model
NODE_FEAT_DIM = train_graph_data.x.size(-1)
EDGE_FEAT_DIM = train_graph_data.edge_attr.size(-1) if hasattr(train_graph_data, 'edge_attr') else 0
HIDDEN_CHANNELS = 64  # Increased
OUT_CHANNELS = 16     # Increased

model = LinkPredictionModel(NODE_FEAT_DIM, EDGE_FEAT_DIM, HIDDEN_CHANNELS, OUT_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=END_LR_FACTOR, total_iters=NUM_EPOCHS)

# Training loop
print("Starting training...")
best_val_auc = 0
for epoch in range(NUM_EPOCHS):
    loss = train(train_graph_data, model, optimizer)
    
    if epoch % LOG_EVERY == 0:
        # Evaluate on validation set
        val_auc = evaluate(val_graph_data, val_neg_edge_index, model, train_graph_data.edge_index)
        
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
    
    scheduler.step()

# Load best model and test
model.load_state_dict(torch.load('best_model.pth'))
test_auc = evaluate(test_graph_data, test_neg_edge_index, model, 
                   torch.cat([train_graph_data.edge_index, val_graph_data.edge_index], dim=1))

print(f"\nFinal Test AUC: {test_auc:.4f}")

# Additional diagnostics
print(f"Train edges: {train_graph_data.edge_index.size(1)}")
print(f"Val edges: {val_graph_data.edge_index.size(1)}")
print(f"Test edges: {test_graph_data.edge_index.size(1)}")
print(f"Number of nodes: {train_graph_data.num_nodes}")
print(f"Node feature dim: {NODE_FEAT_DIM}")
if EDGE_FEAT_DIM > 0:
    print(f"Edge feature dim: {EDGE_FEAT_DIM}")