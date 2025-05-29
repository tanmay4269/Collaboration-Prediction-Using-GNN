import numpy as np

import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from model import LinkPredictionModel
from dataset import OpenAlexGraphDataset

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # all GPUs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to {seed}")


# Config - Reduced learning rate and added weight decay
seed_everything()
BASE_LR = 0.001  # Reduced from 0.005
WEIGHT_DECAY = 1e-4  # Added regularization
NUM_EPOCHS = 100  # Increased epochs
LOG_EVERY = 5  # Less frequent logging
PATIENCE = 15  # Early stopping patience

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset_builder = OpenAlexGraphDataset(json_path="data/openalex_cs_papers.json", num_authors=-1, use_cache=True)
# dataset_builder = OpenAlexGraphDataset(json_path="data/openalex_cs_papers.json", num_authors=-1, use_cache=False)
train_graph_data = dataset_builder.get_train_data()
val_graph_data = dataset_builder.get_val_data()
test_graph_data = dataset_builder.get_test_data()

# Pre-compute negative edges for validation and test
val_neg_edge_index = negative_sampling(
    edge_index=train_graph_data.edge_index,
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
    
    z = model(data.x, data.edge_index, data.edge_attr)
    
    pos_score = model.decode(z, data.edge_index)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_score.size(0),
        method='sparse'
    )
    neg_score = model.decode(z, neg_edge_index)
    
    # Combine scores and labels
    score = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(device)
    
    # Use BCE with logits loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels)
    
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model,             # The GNN model
             node_features_all, # All node features (e.g., train_data.x)
             train_edge_idx,    # Training edge indices (e.g., train_data.edge_index)
             train_edge_attr,   # Training edge attributes (e.g., train_data.edge_attr)
             eval_pos_edge_idx, # Positive edges for evaluation (e.g., val_data.edge_index)
             eval_neg_edge_idx  # Negative edges for evaluation (e.g., precomputed val_neg_edge_index)
            ):
    model.eval()
    # Compute node embeddings (z) using the training graph structure
    z = model(node_features_all, train_edge_idx, train_edge_attr)

    # Decode scores for positive evaluation edges
    pos_scores = model.decode(z, eval_pos_edge_idx)
    # Decode scores for negative evaluation edges
    neg_scores = model.decode(z, eval_neg_edge_idx)

    # Concatenate scores and create true labels
    scores = torch.cat([pos_scores, neg_scores]).cpu()
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ]).cpu()

    return roc_auc_score(labels, scores)

# Model initialization
NODE_FEAT_DIM = train_graph_data.x.size(-1)
EDGE_FEAT_DIM = train_graph_data.edge_attr.size(-1)
HIDDEN_CHANNELS = 64  # Increased from 32
OUT_CHANNELS = 32     # Increased from 8

model = LinkPredictionModel(NODE_FEAT_DIM, EDGE_FEAT_DIM, HIDDEN_CHANNELS, OUT_CHANNELS).to(device)

# Improved optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

# Better learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, verbose=True
)

# Early stopping
best_val_auc = 0
patience_counter = 0

print("Starting training...")
print(f"Train edges: {train_graph_data.edge_index.size(1)}")
print(f"Val edges: {val_graph_data.edge_index.size(1)}")
print(f"Test edges: {test_graph_data.edge_index.size(1)}")
print(f"Nodes: {train_graph_data.num_nodes}")


# Untrained Val AUC
val_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index, train_graph_data.edge_attr,
                   val_graph_data.edge_index, val_neg_edge_index)
print(f"Untrained Val AUC: {val_auc:.4f}")

# Untrained Test AUC
test_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index, train_graph_data.edge_attr,
                    test_graph_data.edge_index, test_neg_edge_index)
print(f"Untrained Test AUC: {test_auc:.4f}")

for epoch in range(NUM_EPOCHS):
    loss = train(train_graph_data, model, optimizer)
    
    if epoch % LOG_EVERY == 0:
        # Evaluate on validation set using training graph structure for embeddings
        val_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index, train_graph_data.edge_attr,
                           val_graph_data.edge_index, val_neg_edge_index)
        
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    

# Load best model for testing
model.load_state_dict(torch.load('best_model.pt'))
print(f"\nBest validation AUC: {best_val_auc:.4f}")

print("Testing...")
# Evaluate on test set using training graph structure for embeddings
test_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index, train_graph_data.edge_attr,
                    test_graph_data.edge_index, test_neg_edge_index)
print(f"Test AUC: {test_auc:.4f}")