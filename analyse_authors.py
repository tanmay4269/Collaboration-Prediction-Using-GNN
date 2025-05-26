import json

authors = set()

with open("openalex_cs_papers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for idx, work in enumerate(data["results"]):
    # print(work.keys()); break
    
    print(f"{idx}. Title: {work['title']}")
    for i, author in enumerate(work["authorships"]):
        print(f"  {i + 1}. {author['author']['display_name']} ({author['author']['id']})")
        authors.add(author['author']['id'])

print(len(authors))
