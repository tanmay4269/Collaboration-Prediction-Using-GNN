import os
import json
from datetime import datetime

import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Data

from sentence_transformers import SentenceTransformer

class OpenAlexGraphDataset:
    def __init__(self, json_path="data/openalex_cs_papers.json", num_authors=200, cache_dir="cache", use_cache=True):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.train_cache_path = os.path.join(self.cache_dir, "train_data.pt")
        self.val_cache_path = os.path.join(self.cache_dir, "val_data.pt")
        self.test_cache_path = os.path.join(self.cache_dir, "test_data.pt")

        if use_cache and (
            os.path.exists(self.train_cache_path) and \
            os.path.exists(self.val_cache_path) and \
            os.path.exists(self.test_cache_path)
        ):
            print("Loading cached data...")
            self.train_data = torch.load(self.train_cache_path, weights_only=False)
            self.val_data = torch.load(self.val_cache_path, weights_only=False)
            self.test_data = torch.load(self.test_cache_path, weights_only=False)
            print("Done!")
        else:
            print("Loading sentence model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Done!")

            print("Building datasets...")
            G = self._build_networkx_graph(json_path, max_num_nodes=num_authors)
            self.train_data, self.val_data, self.test_data = self._create_splits(G)
            print("Done!")

            print(f"Saving processed data to {self.cache_dir}...")
            torch.save(self.train_data, self.train_cache_path)
            torch.save(self.val_data, self.val_cache_path)
            torch.save(self.test_data, self.test_cache_path)
            print("Done!")
    
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
                        work_data=[(work["title"], publication_date)],
                        affiliated_institution=affiliation,
                        citation_count=cited_by
                    )
                else:
                    G.nodes[author_id]["citation_count"] += cited_by
                    G.nodes[author_id]["work_data"].append((work["title"], publication_date))

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
                        
        if max_num_nodes < 0:
            return G

        degrees = dict(G.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda item: item[1])
        nodes_to_remove_count = G.number_of_nodes() - max_num_nodes
        nodes_to_remove = [node for node, _ in sorted_nodes[:nodes_to_remove_count]]
        nodes_to_retain = [node for node in G.nodes() if node not in nodes_to_remove]

        return G.subgraph(nodes_to_retain).copy()
        
    def _process_node_features(self, G, min_date, max_date):
        max_citations = np.max([G.nodes[node_id]['citation_count'] for node_id in G.nodes()])
        # institution_names = [G.nodes[node_id]['affiliated_institution'] for node_id in G.nodes()]
        # institution_embeddings = self.sentence_model.encode(institution_names, convert_to_tensor=True).to('cuda')
        
        # all_work_data = [G.nodes[node_id]["work_data"] for node_id in G.nodes()]
        filtered_author_titles = []
        all_filtered_titles = []
        for node_id in G.nodes():
            data = G.nodes[node_id]["work_data"]
            filtered_titles = []
            for title, date in data:
                if date >= min_date and date <= max_date:  # ! check if it works
                    filtered_titles.append(title)

            filtered_author_titles.append(filtered_titles)
            all_filtered_titles.extend(filtered_titles)
        
        title_embeddings = self.sentence_model.encode(all_filtered_titles, convert_to_tensor=True).to('cuda')

        i = 0
        all_title_embeddings = []
        for work_data in filtered_author_titles:
            n = len(work_data)
            all_title_embeddings.append(title_embeddings[i:i+n].mean(dim=1))  # ! check dim
            i += n

        assert len(all_title_embeddings) == len(G.nodes()), "Title embeddings' messed up"
        
        node_features_list = []
        for i, node_id in enumerate(G.nodes()):
            node = G.nodes[node_id]
            scaled_citation_count = torch.tensor([node["citation_count"] / max_citations], dtype=torch.float).to('cuda')
            feat = torch.cat([scaled_citation_count, all_title_embeddings[i]])
            node_features_list.append(feat)
        
        node_features = torch.stack(node_features_list)
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        return node_features, node_to_idx

    def _process_edge_features(self, G, node_to_idx, edges_to_include, return_feats=False):
        edge_list = [(u, v) for u, v in edges_to_include]
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

        if not return_feats:
            return edge_indices
        
        batched_title_embeddings = self.sentence_model.encode(all_individual_titles, convert_to_tensor=True).to('cuda')

        edge_features_list = []
        for i, (u, v) in enumerate(edge_list):
            start_idx, end_idx = title_slices[i]
            individual_embeddings_for_edge = batched_title_embeddings[start_idx:end_idx]
            averaged_embedding = individual_embeddings_for_edge.mean(dim=0)
            edge_features_list.append(averaged_embedding)

        edge_features = torch.stack(edge_features_list)
        return edge_indices, edge_features

    def _create_splits(self, G, add_edge_attr=False):
        all_edges_with_dates = []
        for u, v, data in G.edges(data=True):
            for date in data['publication_dates']:
                all_edges_with_dates.append(((u, v), date))

        all_edges_with_dates.sort(key=lambda x: x[1])

        num_edges = len(all_edges_with_dates)
        train_end_idx = int(num_edges * 0.6)
        val_end_idx = int(num_edges * 0.8)

        train_edges_with_dates = all_edges_with_dates[:train_end_idx]
        val_edges_with_dates = all_edges_with_dates[train_end_idx:val_end_idx]
        test_edges_with_dates = all_edges_with_dates[val_end_idx:]

        train_edges = list(set([edge for edge, _ in train_edges_with_dates]))
        val_edges = list(set([edge for edge, _ in val_edges_with_dates]))
        test_edges = list(set([edge for edge, _ in test_edges_with_dates]))
        
        last_train_date = train_edges_with_dates[-1][1]
        last_val_date = val_edges_with_dates[-1][1]

        if add_edge_attr:
            node_features, node_to_idx = self._process_node_features(G)

            train_edge_indices, train_edge_features = self._process_edge_features(G, node_to_idx, train_edges)
            val_edge_indices, val_edge_features = self._process_edge_features(G, node_to_idx, val_edges)
            test_edge_indices, test_edge_features = self._process_edge_features(G, node_to_idx, test_edges)

            train_data = Data(x=node_features, edge_index=train_edge_indices, edge_attr=train_edge_features)
            val_data = Data(x=node_features, edge_index=val_edge_indices, edge_attr=val_edge_features)
            test_data = Data(x=node_features, edge_index=test_edge_indices, edge_attr=test_edge_features)
        else:
            ...

        return train_data, val_data, test_data
    
    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data
