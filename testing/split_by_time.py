import json
from datetime import datetime

import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sentence_transformers import SentenceTransformer

class OpenAlexGraphDataset:
    def __init__(self, json_path="data/openalex_cs_papers.json", num_authors=200):
        print("Loading sentence model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Done!")

        G = self._build_networkx_graph(json_path, max_num_nodes=num_authors)
        node_features, node_to_idx = self._process_node_features(G) 
        edge_indices, edge_features = self._process_edge_features(G, node_to_idx)

        self.graph_data = Data(
            x=node_features,
            edge_index=edge_indices,
            edge_attr=edge_features
        )

    def _build_networkx_graph(self, json_path, max_num_nodes):
        G = nx.Graph()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for work in data["results"]:
            authors = []
            cited_by = work.get("cited_by_count", 0)
            publication_date_str = work.get("publication_date")
            assert publication_date_str is not None
            
            publication_date = datetime.strptime(publication_date_str, '%Y-%m-%d')

            for author_data in work["authorships"]:
                author_id = author_data["author"]["id"]
                affiliation = (
                    author_data["institutions"][0]["display_name"]
                    if author_data.get("institutions")
                    else "Unknown"
                )
                authors.append({"id": author_id, "title": work['title']})

                if author_id not in G:
                    G.add_node(
                        author_id,
                        affiliated_institution=affiliation,
                        citation_count=cited_by
                    )
                else:
                    G.nodes[author_id]["citation_count"] += cited_by

            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    id_1, id_2 = authors[i]["id"], authors[j]["id"]
                    if G.has_edge(id_1, id_2):
                        G[id_1][id_2]["title"].append(authors[i]["title"])
                        G[id_1][id_2]["publication_dates"].append(publication_date)
                    else:
                        G.add_edge(id_1, id_2)
                        G[id_1][id_2]["title"] = [authors[i]["title"]]
                        G[id_1][id_2]["publication_dates"] = [publication_date]
                        
        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda item: item[1])
        nodes_to_remove_count = G.number_of_nodes() - max_num_nodes
        nodes_to_remove = [node for node, _ in sorted_nodes[:nodes_to_remove_count]]
        nodes_to_retain = [node for node in G.nodes() if node not in nodes_to_remove]

        return G.subgraph(nodes_to_retain).copy()
    
    def _process_node_features(self, G):
        max_citations = np.max([G.nodes[node_id]['citation_count'] for node_id in G.nodes()])
        institution_names = [G.nodes[node_id]['affiliated_institution'] for node_id in G.nodes()]
        institution_embeddings = self.sentence_model.encode(institution_names, convert_to_tensor=True).to('cuda')

        node_features_list = []
        for i, node_id in enumerate(G.nodes()):
            node = G.nodes[node_id]
            scaled_citation_count = torch.tensor([node["citation_count"] / max_citations], dtype=torch.float).to('cuda')
            feat = torch.cat((scaled_citation_count, institution_embeddings[i]))
            node_features_list.append(feat)

        node_features = torch.stack(node_features_list)
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        return node_features, node_to_idx

    def _process_edge_features(self, G, node_to_idx):
        edge_list = list(G.edges())
        mapped_edges = [[node_to_idx[u], node_to_idx[v]] for u, v in edge_list]
        edge_indices = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous().to('cuda')

        all_individual_titles = []
        title_slices = []
        current_idx = 0
        for u, v in edge_list:
            titles = G[u][v]['title']
            all_individual_titles.extend(titles)
            title_slices.append((current_idx, current_idx + len(titles)))
            current_idx += len(titles)

        batched_title_embeddings = self.sentence_model.encode(all_individual_titles, convert_to_tensor=True).to('cuda')

        edge_features_list = []
        for i, (u, v) in enumerate(edge_list):
            start_idx, end_idx = title_slices[i]
            individual_embeddings_for_edge = batched_title_embeddings[start_idx:end_idx]
            averaged_embedding = individual_embeddings_for_edge.mean(dim=0)
            edge_features_list.append(averaged_embedding)

        edge_features = torch.stack(edge_features_list)
        return edge_indices, edge_features

    def get_data(self):
        return self.graph_data

if __name__ == "__main__":
    dataset_builder = OpenAlexGraphDataset(json_path="data/openalex_cs_papers.json")
    graph_data = dataset_builder.get_data()

    dataset = [graph_data]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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

        print(f"\nNumber of nodes: {batch.num_nodes}")
        print(f"Number of edges: {batch.num_edges}")
        break
