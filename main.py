import torch
from torch_geometric.utils import negative_sampling

import numpy as np
from sklearn.metrics import roc_auc_score

from model import LinkPredictionModel
from dataset import OpenAlexGraphDataset

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    labels = torch.cat([
        torch.ones(pos_score.size(0)), 
        torch.zeros(neg_score.size(0))
    ]).cuda()
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

def runner(dataset_builder, base_lr=0.001, hidden_channels=32, num_layers=2, 
          dropout=0.5, out_channels=32, print_info=False, **kwargs):
    """Runner function with hyperparameters as arguments"""
    seed_everything(42)
    
    CONFIG = {
        'base_lr': base_lr,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'dropout': dropout,
        'out_channels': out_channels,

        'weight_decay': 1e-4,
        'num_epochs': 250,
        'log_every': 5,  # Epochs between logging
        'patience': 5,  # val_auc checked every time logging
    }
    
    # Load dataset
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
        # edge_index=train_graph_data.edge_index,
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
        CONFIG['num_layers'],
        CONFIG['dropout'],
        CONFIG['out_channels']
    ).cuda()
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['base_lr'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Print initial info
    if print_info:
        print("Starting training...")
        print(f"Train edges: {train_graph_data.edge_index.size(1)}")
        print(f"Val edges: {val_graph_data.edge_index.size(1)}")
        print(f"Test edges: {test_graph_data.edge_index.size(1)}")
        print(f"Nodes: {train_graph_data.num_nodes}")
    
    # Evaluate untrained model
    untrained_val_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                      train_graph_data.edge_attr, val_graph_data.edge_index,
                      val_neg_edge_index)
    untrained_test_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                       train_graph_data.edge_attr, test_graph_data.edge_index,
                       test_neg_edge_index)
    if print_info:
        print(f"Untrained Val AUC: {untrained_val_auc:.4f}")
        print(f"Untrained Test AUC: {untrained_test_auc:.4f}")
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None  # Store best model state in memory
    
    for epoch in range(CONFIG['num_epochs']):
        loss = train(train_graph_data, model, optimizer)
        
        if epoch % CONFIG['log_every'] == 0:
            val_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                             train_graph_data.edge_attr, val_graph_data.edge_index,
                             val_neg_edge_index)
            
            if print_info:
                print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | '
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Store model state in memory instead of disk
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1
            
            if patience_counter >= CONFIG['patience']:
                if print_info:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model state from memory
    model.load_state_dict(best_model_state)
    if print_info:
        print(f"\nBest validation AUC: {best_val_auc:.4f}")
    
    best_test_auc = evaluate(model, train_graph_data.x, train_graph_data.edge_index,
                       train_graph_data.edge_attr, test_graph_data.edge_index,
                       test_neg_edge_index)
    if print_info:
        print(f"Test AUC: {best_test_auc:.4f}")
    
    # Return the metrics we want to track
    return {
        'untrained_val': untrained_val_auc,
        'untrained_test': untrained_test_auc,
        'final_val': best_val_auc,
        'final_test': best_test_auc
    }

def main():
    N_RUNS = 2
    results = {
        'untrained_val': [],
        'untrained_test': [],
        'final_val': [],
        'final_test': []
    }
    
    dataset_builder = OpenAlexGraphDataset()
    
    for run in range(N_RUNS):
        print(f"Run {run + 1}/{N_RUNS}")
        run_results = runner(
            dataset_builder=dataset_builder,
            base_lr=0.0036, 
            hidden_channels=256, 
            num_layers=1, 
            dropout=0.66, 
            out_channels=128,
            print_info=False
        )
        for metric in results:
            results[metric].append(run_results[metric])
    
    print(f"\nFinal Statistics over {N_RUNS} runs with the best config:")
    print("-" * 20)
    for metric, values_list in results.items():
        values_np = np.array(values_list)
        mean_val = np.mean(values_np)
        std_val = np.std(values_np)
        print(f"{metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
    print("-" * 20)

if __name__ == "__main__":
    main()