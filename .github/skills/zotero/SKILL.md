---
name: zotero
description: "Use when adding papers to the Zotero library from DOI or arXiv ID, searching bibliography collections, exporting BibTeX, or syncing a local .bib file for LaTeX manuscripts."
metadata: {"clawdbot":{"emoji":"📚","requires":{"bins":["python3"],"pypackages":["pyzotero"]},"config":{"env":{"ZOTERO_LIBRARY_TYPE":{"description":"'user' or 'group'","default":"user","required":false},"ZOTERO_LIBRARY_ID":{"description":"Numeric Zotero library/user ID","default":"","required":true},"ZOTERO_API_KEY":{"description":"Zotero Web API key (read+write)","default":"","required":true},"ZOTERO_BIB_PATH":{"description":"Absolute path where sync-bib writes the .bib file","default":"","required":false}}}}}
---
# Zotero Bibliography Management

Use this skill to manage the research bibliography: add papers discovered during literature search, search existing entries, export BibTeX, and sync the local `.bib` file for LaTeX compilation.

## When to Use

- After academic-research-hub retrieves a paper, push its metadata to Zotero.
- Before manuscript writing, export a `.bib` file for the LaTeX environment.
- Search existing Zotero collections to avoid duplicate entries.
- Sync the local `.bib` file so `\cite{}` commands resolve correctly.

## When Not to Use

- Do not use to retrieve full paper text — use academic-research-hub or lightrag-ingest for that.
- Do not use for note-taking — use siyuan or research-memory.

## Commands

### Add a paper by DOI or arXiv ID

```bash
python3 {baseDir}/zotero_tool.py add-paper --doi "10.1145/3580305.3599533"
python3 {baseDir}/zotero_tool.py add-paper --arxiv "2310.17567"
python3 {baseDir}/zotero_tool.py add-paper --arxiv "2310.17567" --collection "MathAgent"
```

### Search existing entries

```bash
python3 {baseDir}/zotero_tool.py search "convex optimization"
python3 {baseDir}/zotero_tool.py search "LightRAG" --format json
```

### Export BibTeX

```bash
python3 {baseDir}/zotero_tool.py export-bibtex
python3 {baseDir}/zotero_tool.py export-bibtex --collection "MathAgent" --output /tmp/refs.bib
```

### Sync local .bib file

```bash
python3 {baseDir}/zotero_tool.py sync-bib
python3 {baseDir}/zotero_tool.py sync-bib --collection "MathAgent" --output /path/to/manuscript/refs.bib
```

## Output

`add-paper` returns:

```json
{
  "command": "add-paper",
  "key": "ABCD1234",
  "title": "LightRAG: Simple and Fast...",
  "authors": ["Guo et al."],
  "year": 2024,
  "bibtex_key": "guo2024lightrag",
  "success": true
}
```

## Configuration

- `ZOTERO_LIBRARY_TYPE`: `user` (default) or `group`.
- `ZOTERO_LIBRARY_ID`: numeric ID from your Zotero profile URL.
- `ZOTERO_API_KEY`: create at https://www.zotero.org/settings/keys.
- `ZOTERO_BIB_PATH`: default path for `sync-bib` output. Recommended: commit this path to `.gitignore`.

## Installation

```bash
pip install pyzotero
```
