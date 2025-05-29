import numpy as np
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from model import LinkPredictionModel
from dataset import OpenAlexGraphDataset

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def train(data, model, optimizer) -> float:
    """Train the model for one epoch.
    
    Args:
        data: Graph data containing node features and edge information
        model: The GNN model
        optimizer: The optimizer
    
    Returns:
        float: Training loss for this epoch
    """
    model.train()
    optimizer.zero_grad()
    
    z = model(data.x, data.edge_index, data.edge_attr)
    
    # Calculate positive and negative scores
    pos_score = model.decode(z, data.edge_index)
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_score.size(0),
        method='sparse'
    )
    neg_score = model.decode(z, neg_edge_index)
    
    # Prepare labels and compute loss
    score = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.size(0)), 
                       torch.zeros(neg_score.size(0))]).cuda()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(model, node_features_all, train_edge_idx, train_edge_attr,
            eval_pos_edge_idx, eval_neg_edge_idx) -> float:
    """Evaluate the model using ROC AUC score.
    
    Args:
        model: The GNN model
        node_features_all: Node features matrix
        train_edge_idx: Training edge indices
        train_edge_attr: Training edge attributes
        eval_pos_edge_idx: Positive edges for evaluation
        eval_neg_edge_idx: Negative edges for evaluation
    
    Returns:
        float: ROC AUC score
    """
    model.eval()
    z = model(node_features_all, train_edge_idx, train_edge_attr)
    
    pos_scores = model.decode(z, eval_pos_edge_idx)
    neg_scores = model.decode(z, eval_neg_edge_idx)
    
    scores = torch.cat([pos_scores, neg_scores]).cpu()
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ]).cpu()
    
    return roc_auc_score(labels, scores)

def main():
    # Configuration
    CONFIG = {
        'base_lr': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'log_every': 5,
        'patience': 15,
        'hidden_channels': 32,
        'out_channels': 32
    }
    
    # Set random seed
    seed_everything()
    
    # Load dataset
    dataset_builder = OpenAlexGraphDataset(
        json_path="data/openalex_cs_papers.json",
        num_authors=-1,
        use_cache=True
    )
    train_graph_data = dataset_builder.get_train_data()
    val_graph_data = dataset_builder.get_val_data()
    test_graph_data = dataset_builder.get_test_data()
    
    # Pre-compute negative edges
    val_neg_edge_index = negative_sampling(
        edge_index=train_graph_data.edge_index,
        num_nodes=train_graph_data.num_nodes,
        num_neg_samples=val_graph_data.edge_index.size(1),
        method='sparse'
    ).cuda()
    
    test_neg_edge_index = negative_sampling(
        edge_index=torch.cat([train_graph_data.edge_index, val_graph_data.edge_index], dim=1),
        num_nodes=train_graph_data.num_nodes,
        num_neg_samples=test_graph_data.edge_index.size(1),
        method='sparse'
    ).cuda()
    
    # Initialize model
    node_feat_dim = train_graph_data.x.size(-1)
    edge_feat_dim = train_graph_data.edge_attr.size(-1)
    
    model = LinkPredictionModel(
        node_feat_dim, 
        edge_feat_dim,
        CONFIG['hidden_channels'],
        CONFIG['out_channels']
    ).cuda()
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Print initial info
    print("Starting training...")
    print(f"Train edges: {train_graph_data.edge_index.size(1)}")
    print(f"Val edges: {val_graph_data.edge_index.size(1)}")
    print(f"Test edges: {test_graph_data.edge_index.size(1)}")
    print(f"Nodes: {train_graph_data.num_nodes}")
    
    # Evaluate untrained model
    val_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                      train_graph_data.edge_attr, val_graph_data.edge_index,
                      val_neg_edge_index)
    test_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                       train_graph_data.edge_attr, test_graph_data.edge_index,
                       test_neg_edge_index)
    print(f"Untrained Val AUC: {val_auc:.4f}")
    print(f"Untrained Test AUC: {test_auc:.4f}")
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        loss = train(train_graph_data, model, optimizer)
        
        if epoch % CONFIG['log_every'] == 0:
            val_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                             train_graph_data.edge_attr, val_graph_data.edge_index,
                             val_neg_edge_index)
            
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= CONFIG['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    
    print("Testing...")
    test_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                       train_graph_data.edge_attr, test_graph_data.edge_index,
                       test_neg_edge_index)
    print(f"Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()