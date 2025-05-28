import os
import json
from datetime import datetime
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from collections import defaultdict

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
            self.train_data, self.val_data, self.test_data = self._create_proper_temporal_splits(G)
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

        # Store all collaborations with their dates
        collaborations = []
        author_info = {}

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
                authors.append(author_id)

                # Store author info
                if author_id not in author_info:
                    author_info[author_id] = {
                        'affiliated_institution': affiliation,
                        'citation_count': cited_by,
                        'work_count': 1
                    }
                else:
                    author_info[author_id]['citation_count'] += cited_by
                    author_info[author_id]['work_count'] += 1

            # Store all collaborations with dates
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    collaborations.append({
                        'authors': (authors[i], authors[j]),
                        'date': publication_date,
                        'title': work['title']
                    })

        # Filter authors if needed
        if max_num_nodes > 0:
            work_counts = {author: info['work_count'] for author, info in author_info.items()}
            sorted_authors = sorted(work_counts.items(), key=lambda x: x[1], reverse=True)
            top_authors = set([author for author, _ in sorted_authors[:max_num_nodes]])
            
            # Filter collaborations to only include top authors
            collaborations = [
                collab for collab in collaborations 
                if collab['authors'][0] in top_authors and collab['authors'][1] in top_authors
            ]
            author_info = {k: v for k, v in author_info.items() if k in top_authors}

        # Build graph with all collaborations
        G.add_nodes_from(author_info.keys())
        for author, info in author_info.items():
            G.nodes[author].update(info)

        # Add edges with collaboration info
        edge_info = defaultdict(lambda: {'titles': [], 'dates': []})
        for collab in collaborations:
            u, v = collab['authors']
            edge_info[(u, v)]['titles'].append(collab['title'])
            edge_info[(u, v)]['dates'].append(collab['date'])

        for (u, v), info in edge_info.items():
            G.add_edge(u, v, title=info['titles'], publication_dates=info['dates'])

        # Store all collaborations for temporal splitting
        G.graph['all_collaborations'] = collaborations
        
        return G
        
    def _create_proper_temporal_splits(self, G):
        """Create proper temporal splits with no data leakage"""
        
        # Get all collaborations sorted by date
        all_collabs = sorted(G.graph['all_collaborations'], key=lambda x: x['date'])
        
        print(f"Total collaborations: {len(all_collabs)}")
        print(f"Date range: {all_collabs[0]['date']} to {all_collabs[-1]['date']}")
        
        # Define temporal cutoffs
        total_collabs = len(all_collabs)
        train_end_idx = int(total_collabs * 0.6)
        val_end_idx = int(total_collabs * 0.8)
        
        train_collabs = all_collabs[:train_end_idx]
        val_collabs = all_collabs[train_end_idx:val_end_idx]
        test_collabs = all_collabs[val_end_idx:]
        
        train_cutoff_date = train_collabs[-1]['date']
        val_cutoff_date = val_collabs[-1]['date']
        
        print(f"Train cutoff: {train_cutoff_date}")
        print(f"Val cutoff: {val_cutoff_date}")
        print(f"Train: {len(train_collabs)}, Val: {len(val_collabs)}, Test: {len(test_collabs)}")
        
        # Create edge sets (no duplicates)
        train_edges = set()
        val_edges = set()
        test_edges = set()
        
        for collab in train_collabs:
            u, v = collab['authors']
            train_edges.add((min(u, v), max(u, v)))
            
        for collab in val_collabs:
            u, v = collab['authors']
            edge = (min(u, v), max(u, v))
            if edge not in train_edges:  # Only new collaborations
                val_edges.add(edge)
            
        for collab in test_collabs:
            u, v = collab['authors']
            edge = (min(u, v), max(u, v))
            if edge not in train_edges and edge not in val_edges:  # Only new collaborations
                test_edges.add(edge)
        
        print(f"Unique edges - Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")
        
        # Verify no overlap
        assert len(train_edges.intersection(val_edges)) == 0, "Train-Val overlap detected!"
        assert len(train_edges.intersection(test_edges)) == 0, "Train-Test overlap detected!"
        assert len(val_edges.intersection(test_edges)) == 0, "Val-Test overlap detected!"
        
        # Process features
        node_features, node_to_idx = self._process_node_features(G)
        
        # Convert edges to torch format
        def edges_to_torch(edge_set, G, node_to_idx):
            if not edge_set:
                # Return empty tensors if no edges
                return torch.zeros((2, 0), dtype=torch.long).to('cuda'), torch.zeros((0, 384), dtype=torch.float).to('cuda')
            
            edge_list = list(edge_set)
            mapped_edges = [[node_to_idx[u], node_to_idx[v]] for u, v in edge_list]
            edge_indices = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous().to('cuda')
            
            # Process edge features
            edge_features_list = []
            for u, v in edge_list:
                if G.has_edge(u, v):
                    titles = G[u][v]['title']
                    title_embeddings = self.sentence_model.encode(titles, convert_to_tensor=True).to('cuda')
                    avg_embedding = title_embeddings.mean(dim=0)
                    edge_features_list.append(avg_embedding)
                else:
                    # Default embedding for missing edges
                    edge_features_list.append(torch.zeros(384).to('cuda'))
            
            edge_features = torch.stack(edge_features_list)
            return edge_indices, edge_features
        
        train_edge_indices, train_edge_features = edges_to_torch(train_edges, G, node_to_idx)
        val_edge_indices, val_edge_features = edges_to_torch(val_edges, G, node_to_idx)
        test_edge_indices, test_edge_features = edges_to_torch(test_edges, G, node_to_idx)
        
        # Create data objects
        train_data = Data(x=node_features, edge_index=train_edge_indices, edge_attr=train_edge_features)
        val_data = Data(x=node_features, edge_index=val_edge_indices, edge_attr=val_edge_features)
        test_data = Data(x=node_features, edge_index=test_edge_indices, edge_attr=test_edge_features)
        
        return train_data, val_data, test_data

    def _process_node_features(self, G):
        max_citations = max([G.nodes[node_id]['citation_count'] for node_id in G.nodes()]) if G.nodes() else 1
        institution_names = [G.nodes[node_id]['affiliated_institution'] for node_id in G.nodes()]
        
        if institution_names:
            institution_embeddings = self.sentence_model.encode(institution_names, convert_to_tensor=True).to('cuda')
        else:
            institution_embeddings = torch.zeros((len(G.nodes()), 384)).to('cuda')

        node_features_list = []
        for i, node_id in enumerate(G.nodes()):
            node = G.nodes[node_id]
            scaled_citation_count = torch.tensor([node["citation_count"] / max_citations], dtype=torch.float).to('cuda')
            feat = torch.cat((scaled_citation_count, institution_embeddings[i]))
            node_features_list.append(feat)

        node_features = torch.stack(node_features_list) if node_features_list else torch.zeros((0, 385)).to('cuda')
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        return node_features, node_to_idx

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data