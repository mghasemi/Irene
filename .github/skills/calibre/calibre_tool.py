import requests
import sys
import json

BASE_URL = "http://192.168.1.84:6060"
ALT_BASE_URL = "http://mghasemi.ddns.net:6060"

def search_calibre(query):
    errors = []
    data = None
    for base_url in (BASE_URL, ALT_BASE_URL):
        try:
            # Fetch the initial metadata payload, then fall back to alternate server if needed.
            response = requests.get(f"{base_url}/interface-data/books-init", timeout=10)
            response.raise_for_status()
            data = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{base_url}: {exc}")

    if data is None:
        return {"error": "Unable to reach Calibre server. Tried URLs: " + " | ".join(errors)}

    metadata = data.get("metadata", {})
    results = []
    query = query.lower()

    for bid, meta in metadata.items():
        title = meta.get("title", "").lower()
        authors = " ".join(meta.get("authors", [])).lower()

        if query in title or query in authors:
            results.append({
                "id": bid,
                "title": meta.get("title"),
                "authors": ", ".join(meta.get("authors", [])),
                "formats": meta.get("formats", [])
            })
    return results

if __name__ == "__main__":
    query_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    print(json.dumps(search_calibre(query_str), indent=2))
