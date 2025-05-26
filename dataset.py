import json
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

G = nx.Graph()
sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# This json file has all the filtered data provided by OpenAlex API
with open("data/openalex_cs_papers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for work in data["results"]:
    authors = []
    cited_by = work.get("cited_by_count", 0)  
    embeddings = sentence_embedding_model.encode(work['title'])

    # Adding/updating attributes for each author
    for author_data in work["authorships"]:
        author_id = author_data["author"]["id"]
        display_name = author_data["author"]["display_name"]
        affiliation = (
            author_data["institutions"][0]["display_name"]
            if author_data.get("institutions")
            else "Unknown"
        )
        
        authors.append({"id": author_id, "title-embeddings": embeddings})

        # Custom attributes for author nodes
        if author_id not in G:
            G.add_node(
                author_id,
                display_name=display_name,
                affiliated_institution=affiliation,
                work_count=1,
                citation_count=cited_by
            )
        else:
            G.nodes[author_id]["work_count"] += 1
            G.nodes[author_id]["citation_count"] += cited_by

    # Adding co-authorship edges
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            id_1, id_2 = id_1["id"], id_2["id"]
            if G.has_edge(id_1, id_2):
                G[id_1][id_2]["weight"] += 1
                # TODO: check if this works
                G[id_1][id_2]["title-embeddings"] = np.vstack([
                    G[id_1][id_2]["title-embeddings"],
                    authors[i]["title-embeddings"]
                ])
            else:
                G.add_edge(id_1, id_2, weight=1)
                G[id_1][id_2]["title-embeddings"] = authors[i]["title-embeddings"]
