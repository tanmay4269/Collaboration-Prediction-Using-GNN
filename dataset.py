from tqdm import tqdm
import json
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

print("Loading sentence model")
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Done!")

G = nx.Graph()
max_citations = -1

# This json file has all the filtered data provided by OpenAlex API
with open("data/openalex_cs_papers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Collecting author and co-authorship features
for work in data["results"]:
    authors = []
    cited_by = work.get("cited_by_count", 0)  

    # Adding/updating attributes for each author
    for author_data in work["authorships"]:
        author_id = author_data["author"]["id"]
        affiliation = (
            author_data["institutions"][0]["display_name"]
            if author_data.get("institutions")
            else "Unknown"
        )
        
        authors.append({"id": author_id, "title": work['title']})

        # Custom attributes for author nodes
        if author_id not in G:
            G.add_node(
                author_id,
                affiliated_institution=affiliation,
                citation_count=cited_by
            )
        else:
            G.nodes[author_id]["citation_count"] += cited_by

        max_citations = max(max_citations, G.nodes[author_id]["citation_count"])

    # Adding co-authorship edges
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            id_1, id_2 = authors[i]["id"], authors[j]["id"]
            if G.has_edge(id_1, id_2):
                G[id_1][id_2]["title"].append(authors[i]["title"])
            else:
                G.add_edge(id_1, id_2)
                G[id_1][id_2]["title"] = [
                    authors[i]["title"]
                ]

# Nodes
institution_names = [G.nodes[node_id]['affiliated_institution'] for node_id in G.nodes()]
institution_embeddings = sentence_model.encode(institution_names, convert_to_tensor=True).to('cuda')

node_features_list = []
for i, node_id in tqdm(enumerate(G.nodes())):
    node = G.nodes[node_id]
    scaled_citation_count = torch.tensor([node["citation_count"] / max_citations], dtype=torch.float).to('cuda')
    feat = torch.cat((scaled_citation_count, institution_embeddings[i]))
    node_features_list.append(feat)

node_features = torch.stack(node_features_list)


# Edges
edge_list = list(G.edges())
edge_indices = torch.empty((2, 0), dtype=torch.long)
edge_features = torch.empty((0, sentence_model.get_sentence_embedding_dimension()), dtype=torch.float).to('cuda')

assert edge_list is not None

node_to_idx = {node: i for i, node in enumerate(G.nodes())}
mapped_edges = [[node_to_idx[u], node_to_idx[v]] for u, v in edge_list]
edge_indices = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous().to('cuda')

all_individual_titles = []
title_slices = []
current_idx = 0
for u, v in tqdm(edge_list):
    titles = G[u][v]['title']
    all_individual_titles.extend(titles)
    title_slices.append((current_idx, current_idx + len(titles)))
    current_idx += len(titles)

batched_title_embeddings = sentence_model.encode(all_individual_titles, convert_to_tensor=True).to('cuda')

edge_features_list = []
for i, (u, v) in tqdm(enumerate(edge_list)):
    start_idx, end_idx = title_slices[i]
    individual_embeddings_for_edge = batched_title_embeddings[start_idx:end_idx]
    averaged_embedding = individual_embeddings_for_edge.mean(dim=0)
    edge_features_list.append(averaged_embedding)

edge_features = torch.stack(edge_features_list)


# Final stuff
graph_data = Data(
    x=node_features,
    edge_index=edge_indices,
    edge_attr=edge_features
)

dataset = [graph_data]
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Assuming 'single_batch_dataloader' is the DataLoader object you created

for batch in dataloader:
    print(f"Type of batch object: {type(batch)}")
    print(f"Batch content: {batch}")
    
    print(f"\nNode features (x):")
    print(f"  Shape: {batch.x.shape}")
    print(f"  Device: {batch.x.device}")
    
    print(f"\nEdge indices (edge_index):")
    print(f"  Shape: {batch.edge_index.shape}")
    print(f"  Device: {batch.edge_index.device}")
    
    print(f"\nEdge features (edge_attr):")
    print(f"  Shape: {batch.edge_attr.shape}")
    print(f"  Device: {batch.edge_attr.device}")

    # You can also check other attributes of the Data object
    print(f"\nNumber of nodes: {batch.num_nodes}")
    print(f"Number of edges: {batch.num_edges}")

    # Break after the first (and only) batch if you only want to inspect it
    break