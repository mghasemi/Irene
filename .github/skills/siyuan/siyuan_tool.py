import requests
import sys

TOKEN = "dmwaqp7yt2kv39w9"
URL = "http://192.168.1.70:6806"
ALT_URL = "http://mghasemi.ddns.net:6806"

def query_siyuan(path, data):
    headers = {"Authorization": f"Token {TOKEN}"}
    errors = []
    for base_url in (URL, ALT_URL):
        try:
            response = requests.post(f"{base_url}{path}", json=data, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{base_url}: {exc}")

    raise RuntimeError("Unable to reach SiYuan server. Tried URLs: " + " | ".join(errors))

def search_notes(query):
    # Using SQL to search for document names or content
    stmt = f"SELECT * FROM blocks WHERE (content LIKE '%{query}%' OR attribute LIKE '%{query}%') AND type='d' LIMIT 10"
    return query_siyuan("/api/query/sql", {"stmt": stmt})

def get_content(block_id):
    return query_siyuan("/api/export/exportMdContent", {"id": block_id})

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        print(search_notes(sys.argv[2]))
    elif cmd == "get":
        print(get_content(sys.argv[2]))
