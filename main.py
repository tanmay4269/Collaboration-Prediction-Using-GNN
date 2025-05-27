import torch
from torch_geometric.utils import negative_sampling

from sklearn.metrics import roc_auc_score

from model import GraphSAGE, LinkPredictor
from dataset import OpenAlexGraphDataset

# Config
BASE_LR = 0.01

NUM_EPOCHS = 70
LOG_EVERY = 10  # Epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_builder = OpenAlexGraphDataset(json_path="data/openalex_cs_papers.json", num_authors=-1)

train_graph_data = dataset_builder.get_train_data()
val_graph_data = dataset_builder.get_val_data()
test_graph_data = dataset_builder.get_test_data()

def custom_negative_sampling(mode, num_neg_samples):
    edge_index = None
    num_nodes = train_graph_data.num_nodes

    if mode == 'train':
        edge_index = torch.cat([
            train_graph_data.edge_index,
            val_graph_data.edge_index,
            test_graph_data.edge_index
        ], dim=1)
    elif mode == 'val':
        edge_index = torch.cat([
            val_graph_data.edge_index,
            test_graph_data.edge_index
        ], dim=1) 
    else:
        edge_index = test_graph_data.edge_index

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method='sparse'
    )

    return neg_edge_index
    

def train(data, model, link_predictor, optimizer):
    model.train()
    link_predictor.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)
    
    pos_score = link_predictor(z, data.edge_index, data.edge_attr)

    neg_edge_index = custom_negative_sampling('train', pos_score.size(0))
    neg_score = link_predictor(z, neg_edge_index, None)

    score = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))]).to(device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(data, model, link_predictor, mode):
    model.eval()
    link_predictor.eval()
    z = model(data.x, data.edge_index)

    pos_pred = link_predictor(z, data.edge_index, data.edge_attr).sigmoid()
    
    neg_edge_index = custom_negative_sampling(mode, pos_pred.size(0))
    neg_pred = link_predictor(z, neg_edge_index, None).sigmoid()

    y_pred = torch.cat([pos_pred, neg_pred]).cpu()
    y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).cpu()

    return roc_auc_score(y_true, y_pred)


model = GraphSAGE(train_graph_data.x.size(-1), 128).to(device)
link_predictor = LinkPredictor(model.conv2.out_channels, train_graph_data.edge_attr.size(-1)).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=BASE_LR)

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    loss = train(train_graph_data, model, link_predictor, optimizer)

    if epoch % LOG_EVERY == LOG_EVERY-1:
        val_auc = evaluate(val_graph_data, model, link_predictor, mode='val')
        print(f'Epoch {epoch+1:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}')

print("\nTesting...")
test_auc = evaluate(test_graph_data, model, link_predictor, mode='test')
print(f"Test AUC: {test_auc:.4f}")