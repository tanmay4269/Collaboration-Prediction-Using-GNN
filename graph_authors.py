import json
import networkx as nx
import csv

# This json file has all the filtered data provided by OpenAlex API
with open("openalex_cs_papers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

G = nx.Graph()

for work in data["results"]:
    cited_by = work.get("cited_by_count", 0)  
    authors = []

    # Adding/updating attributes for each author
    for author_data in work["authorships"]:
        author_id = author_data["author"]["id"]
        display_name = author_data["author"]["display_name"]
        affiliation = (
            author_data["institutions"][0]["display_name"]
            if author_data.get("institutions")
            else "Unknown"
        )

        authors.append(author_id)

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
            if G.has_edge(authors[i], authors[j]):
                G[authors[i]][authors[j]]["weight"] += 1
            else:
                G.add_edge(authors[i], authors[j], weight=1)

# Select top 200 authors by work count
top_authors = sorted(G.nodes(data=True), key=lambda x: x[1]["work_count"], reverse=True)[:200]
top_author_ids = {author_id for author_id, _ in top_authors}
G_sub = G.subgraph(top_author_ids).copy()

# Export nodes to CSV
with open("openalex_authors_nodes.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "display_name", "affiliated_institution", "work_count", "citation_count"])
    for node_id, attr in G_sub.nodes(data=True):
        writer.writerow([
            node_id,
            attr.get("display_name", ""),
            attr.get("affiliated_institution", ""),
            attr.get("work_count", 0),
            attr.get("citation_count", 0)
        ])

# Export edges to CSV
with open("openalex_coauthorship_edges.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target", "weight"])
    for u, v, attr in G_sub.edges(data=True):
        writer.writerow([u, v, attr.get("weight", 1)])

