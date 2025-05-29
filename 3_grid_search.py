import os
import itertools
from datetime import datetime
timestamp = datetime.now().strftime('%d-%H-%M-%S')

import torch
from torch_geometric.utils import negative_sampling

import numpy as np
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

def trainer(config: dict, dataset_builder, save_dir: str) -> float:
    """Train a model with given configuration and return best validation AUC.
    
    Args:
        config: Dictionary containing model and training parameters
        dataset_builder: Dataset object containing train/val/test splits
        save_dir: Directory to save model weights
    
    Returns:
        float: Best validation AUC achieved
    """
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
        config['hidden_channels'],
        config['out_channels']
    ).cuda()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['base_lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    print(f"\nTraining with config: {config}")
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
    
    for epoch in range(config['num_epochs']):
        loss = train(train_graph_data, model, optimizer)
        
        if epoch % config['log_every'] == 0:
            val_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                             train_graph_data.edge_attr, val_graph_data.edge_index,
                             val_neg_edge_index)
            
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                
                # Save model with timestamp and AUC
                save_path = os.path.join(save_dir, f"{timestamp}.pth")
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Testing...")
    test_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                       train_graph_data.edge_attr, test_graph_data.edge_index,
                       test_neg_edge_index)
    print(f"Test AUC: {test_auc:.4f}")
    
    return best_val_auc

def main():
    # Grid search parameters
    grid_params = {
        'base_lr': [0.001],
        'hidden_channels': [32],
        'out_channels': [32]
    }
    # grid_params = {
    #     'base_lr': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    #     'hidden_channels': [16, 32, 64],
    #     'out_channels': [16, 32, 64]
    # }
    
    # Fixed parameters
    fixed_params = {
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'log_every': 5,
        'patience': 15
    }
    
    # Create directory for saved weights
    save_dir = "saved_weights"
    os.makedirs(save_dir, exist_ok=True)
    
    # Set random seed
    seed_everything()
    
    # Load dataset
    dataset_builder = OpenAlexGraphDataset(
        json_path="data/openalex_cs_papers.json",
        num_authors=-1,
        use_cache=True
    )
    
    # Generate all possible combinations of parameters
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    
    best_config = None
    best_val_auc = 0
    
    # Perform grid search
    for params in itertools.product(*param_values):
        # Create config for this run
        current_config = dict(zip(param_names, params))
        current_config.update(fixed_params)  # Add fixed parameters
        
        # Train model with current config
        val_aucs = []
        for train_idx in range(10):
            print("*** Training run", train_idx + 1, "***")
            val_aucs.append(trainer(current_config, dataset_builder, save_dir))
        
        print(f"*** Val AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f} ***")
        # Update best config if necessary
        if np.mean(val_aucs) > best_val_auc:
            best_val_auc = np.mean(val_aucs)
            best_config = current_config.copy()
    
    print("\nGrid Search Results:")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print("Best configuration:")
    for key, value in best_config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()