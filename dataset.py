import os
import json
from datetime import datetime

import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Data

from sentence_transformers import SentenceTransformer

class OpenAlexGraphDataset:
    def __init__(
        self, 
        json_path="data/openalex_cs_papers.json", 
        num_authors=-1,
        cache_dir="cache",
        use_cache=True,

        use_citation_count=True,
        use_work_count=True,
        use_institution_embedding=True,

        use_subfield_embedding=False,
        use_field_embedding=False,
        use_title_embedding=True,
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.use_citation_count = use_citation_count
        self.use_work_count = use_work_count
        self.use_institution_embedding = use_institution_embedding

        self.use_subfield_embedding = use_subfield_embedding
        self.use_field_embedding = use_field_embedding
        self.use_title_embedding = use_title_embedding

        # Store unique subfields and fields
        self.unique_subfields = set()
        self.unique_fields = set()

        self.train_cache_path = os.path.join(self.cache_dir, "train_data.pt")
        self.val_cache_path = os.path.join(self.cache_dir, "val_data.pt")
        self.dev_test_cache_path = os.path.join(self.cache_dir, "dev_test_data.pt")
        self.test_cache_path = os.path.join(self.cache_dir, "test_data.pt")

        if use_cache and (
            os.path.exists(self.train_cache_path) and \
            os.path.exists(self.val_cache_path) and \
            os.path.exists(self.dev_test_cache_path) and \
            os.path.exists(self.test_cache_path)
        ):
            # print("Loading cached data...")
            self.train_data = torch.load(self.train_cache_path, weights_only=False)
            self.val_data = torch.load(self.val_cache_path, weights_only=False)
            self.dev_test_data = torch.load(self.dev_test_cache_path, weights_only=False)
            self.test_data = torch.load(self.test_cache_path, weights_only=False)
            # print("Done!")
        else:
            print("Loading sentence model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

            print("Done!")

            print("Building datasets...")
            G = self._build_networkx_graph(json_path, max_num_nodes=num_authors)
            self.train_data, self.val_data, self.dev_test_data, self.test_data = self._create_splits(G)
            print("Done!")

            print(f"Saving processed data to {self.cache_dir}...")
            torch.save(self.train_data, self.train_cache_path)
            torch.save(self.val_data, self.val_cache_path)
            torch.save(self.dev_test_data, self.dev_test_cache_path)
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

            # Extract topics information
            subfields = []
            fields = []
            for topic_dict in work['topics']:
                subfields.append(topic_dict['subfield']['id'])
                fields.append(topic_dict['field']['id'])
            
            # Update unique subfields and fields
            self.unique_subfields.update(subfields)
            self.unique_fields.update(fields)

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
                        citation_count=cited_by,
                        work_count=1
                    )
                else:
                    G.nodes[author_id]["citation_count"] += cited_by
                    G.nodes[author_id]["work_count"] += 1

            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    id_1, id_2 = authors[i]["id"], authors[j]["id"]
                    if G.has_edge(id_1, id_2):
                        G[id_1][id_2]["title"].append(authors[i]["title"])
                        G[id_1][id_2]["publication_dates"].append(publication_date)
                        G[id_1][id_2]["subfields"].append(subfields)
                        G[id_1][id_2]["fields"].append(fields)
                    else:
                        G.add_edge(id_1, id_2)
                        G[id_1][id_2]["title"] = [authors[i]["title"]]
                        G[id_1][id_2]["publication_dates"] = [publication_date]
                        G[id_1][id_2]["subfields"] = [subfields]
                        G[id_1][id_2]["fields"] = [fields]
                        
        if max_num_nodes < 0:
            return G

        # Filter based on work_count
        work_counts = {node: G.nodes[node]["work_count"] for node in G.nodes()}
        sorted_nodes = sorted(work_counts.items(), key=lambda item: item[1])
        nodes_to_remove_count = G.number_of_nodes() - max_num_nodes
        nodes_to_remove = [node for node, _ in sorted_nodes[:nodes_to_remove_count]]
        nodes_to_retain = [node for node in G.nodes() if node not in nodes_to_remove]

        return G.subgraph(nodes_to_retain).copy()
        
    def _process_node_features(self, G):
        max_citations = np.max([G.nodes[node_id]['citation_count'] for node_id in G.nodes()])
        max_work_count = np.max([G.nodes[node_id]['work_count'] for node_id in G.nodes()])
        institution_names = [G.nodes[node_id]['affiliated_institution'] for node_id in G.nodes()]
        institution_embeddings = self.sentence_model.encode(institution_names, convert_to_tensor=True).to('cuda')

        node_features_list = []
        for i, node_id in enumerate(G.nodes()):
            node = G.nodes[node_id]
            feat = []
            if self.use_citation_count:
                feat.append(torch.tensor([node["citation_count"] / max_citations], dtype=torch.float).cuda())
            if self.use_work_count:
                feat.append(torch.tensor([node["work_count"] / max_work_count], dtype=torch.float).cuda()) 
            if self.use_institution_embedding:
                feat.append(institution_embeddings[i])

            feat = torch.cat(feat) if feat else torch.zeros(1).cuda()
            node_features_list.append(feat)

        node_features = torch.stack(node_features_list)
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        return node_features, node_to_idx

    def _process_edge_features(self, G, node_to_idx, edges_to_include):
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

        batched_title_embeddings = self.sentence_model.encode(all_individual_titles, convert_to_tensor=True).to('cuda')

        # Convert unique subfields and fields to ordered lists for consistent indexing
        subfield_list = sorted(list(self.unique_subfields))
        field_list = sorted(list(self.unique_fields))
        subfield_to_idx = {sf: i for i, sf in enumerate(subfield_list)}
        field_to_idx = {f: i for i, f in enumerate(field_list)}

        edge_features_list = []
        for i, (u, v) in enumerate(edge_list):
            feats = []

            if self.use_subfield_embedding:
                subfield_vec = torch.zeros(len(subfield_list), dtype=torch.float).to('cuda') 
                edge_subfields = [sf for subfields_list in G[u][v]['subfields'] for sf in subfields_list]
                for sf in edge_subfields:
                    subfield_vec[subfield_to_idx[sf]] = 1.0
                feats.append(subfield_vec)

            if self.use_field_embedding:
                field_vec = torch.zeros(len(field_list), dtype=torch.float).to('cuda')
                edge_fields = [f for fields_list in G[u][v]['fields'] for f in fields_list]
                for f in edge_fields:
                    field_vec[field_to_idx[f]] = 1.0
                feats.append(field_vec)
            
            if self.use_title_embedding:
                start_idx, end_idx = title_slices[i]
                individual_embeddings_for_edge = batched_title_embeddings[start_idx:end_idx]
                averaged_embedding = individual_embeddings_for_edge.mean(dim=0)
                feats.append(averaged_embedding)

            edge_feature = torch.cat(feats) if feats else torch.zeros(1).cuda()
            edge_features_list.append(edge_feature)

        edge_features = torch.stack(edge_features_list)
        return edge_indices, edge_features

    def _create_splits(self, G):
        all_edges_with_dates = []
        for u, v, data in G.edges(data=True):
            for date in data['publication_dates']:
                all_edges_with_dates.append(((u, v), date))

        all_edges_with_dates.sort(key=lambda x: x[1])

        # 70% train, 10% val, 10% dev-test, 10% test
        num_edges = len(all_edges_with_dates)
        train_end_idx = int(num_edges * 0.7)
        val_end_idx = int(num_edges * 0.8)
        dev_test_end_idx = int(num_edges * 0.9)

        train_edges_with_dates = all_edges_with_dates[:train_end_idx]
        val_edges_with_dates = all_edges_with_dates[train_end_idx:val_end_idx]
        dev_test_edges_with_dates = all_edges_with_dates[val_end_idx:dev_test_end_idx]
        test_edges_with_dates = all_edges_with_dates[dev_test_end_idx:]

        train_edges = list(set([edge for edge, _ in train_edges_with_dates]))
        val_edges = list(set([edge for edge, _ in val_edges_with_dates]))
        dev_test_edges = list(set([edge for edge, _ in dev_test_edges_with_dates]))
        test_edges = list(set([edge for edge, _ in test_edges_with_dates]))

        node_features, node_to_idx = self._process_node_features(G)

        train_edge_indices, train_edge_features = self._process_edge_features(G, node_to_idx, train_edges)
        val_edge_indices, val_edge_features = self._process_edge_features(G, node_to_idx, val_edges)
        dev_test_edge_indices, dev_test_edge_features = self._process_edge_features(G, node_to_idx, dev_test_edges)
        test_edge_indices, test_edge_features = self._process_edge_features(G, node_to_idx, test_edges)

        train_data = Data(x=node_features, edge_index=train_edge_indices, edge_attr=train_edge_features)
        val_data = Data(x=node_features, edge_index=val_edge_indices, edge_attr=val_edge_features)
        dev_test_data = Data(x=node_features, edge_index=dev_test_edge_indices, edge_attr=dev_test_edge_features)
        test_data = Data(x=node_features, edge_index=test_edge_indices, edge_attr=test_edge_features)

        return train_data, val_data, dev_test_data, test_data

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_dev_test_data(self):
        return self.dev_test_data

    def get_test_data(self):
        return self.test_data
    
if __name__ == "__main__":
    # dataset = OpenAlexGraphDataset(json_path="data/openalex_cs_papers.json", num_authors=-1, use_cache=True)
    dataset = OpenAlexGraphDataset(json_path="data/openalex_cs_papers.json", num_authors=-1, use_cache=False)