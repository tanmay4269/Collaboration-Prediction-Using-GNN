import requests
import json

url = "https://api.openalex.org/works"
params = {
    "filter": "concepts.id:C41008148,language:en,from_publication_date:2020-01-01,to_publication_date:2024-12-31",
    "per-page": 200
}

response = requests.get(url, params=params)
data = response.json()

# Save to file
with open("openalex_cs_papers.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Saved to openalex_cs_papers.json")

