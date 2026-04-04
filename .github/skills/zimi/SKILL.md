---
name: zimi
description: "Use when searching ZIM files, querying an offline encyclopedia, looking up article titles in a local wiki mirror, reading article text from a ZIMI server, retrieving offline reference material, or searching local mathematics and science archives served from ZIM files."
metadata: {"clawdbot":{"emoji":"📚","requires":{"bins":["python3"]},"config":{"env":{"ZIMI_URL":{"description":"ZIMI server URL","default":"http://192.168.1.70:8899","required":true},"ZIMI_TIMEOUT":{"description":"HTTP timeout in seconds","default":"20","required":false}}}}}
---
# ZIMI Offline Reference

Use this skill to search and read content from your local ZIMI server.

## When to Use

- Search an offline encyclopedia or reference set stored in ZIM files.
- Check whether an article title exists before doing a broader search.
- Read clean plain text from a specific ZIM article without HTML clutter.
- Retrieve bounded article text for math, science, or general knowledge queries.

## Commands

Run the tool directly from this skill folder.

### Suggest titles

```bash
python3 {baseDir}/zimi_tool.py suggest "topolog"
python3 {baseDir}/zimi_tool.py --url http://192.168.1.70:8899 --alt-url http://mghasemi.ddns.net:8899 suggest "topolog"
```

### Full-text search

```bash
python3 {baseDir}/zimi_tool.py search "Riemann hypothesis" --limit 3
python3 {baseDir}/zimi_tool.py search "sum of squares" --limit 5 --format json
python3 {baseDir}/zimi_tool.py search "sdp hierarchy" --prefer-source wikipedia_en_mathematics_nopic
```

### Read an article

```bash
python3 {baseDir}/zimi_tool.py read --zim wikipedia_en_mathematics_nopic --path Riemann_hypothesis --max-length 4000
```

### Deliberate retrieval flow

```bash
python3 {baseDir}/zimi_tool.py retrieve "Riemann hypothesis"
python3 {baseDir}/zimi_tool.py retrieve "Topology" --max-length 5000 --search-limit 5
python3 {baseDir}/zimi_tool.py retrieve "sdp hierarchy" --prefer-source planetmath.org
```

## Retrieval Guidance

The `retrieve` command is the default high-level workflow.

1. Keep the original query intact.
2. Treat short title-like requests as article-title lookups first.
3. Use full-text search for descriptive or question-style queries.
4. Only fall back to simplified rewrites if the first pass is weak.
5. Read at most one or two bounded articles so output stays usable.

## Configuration

Optional environment overrides:

- `ZIMI_URL`: defaults to `http://192.168.1.70:8899`
- `ZIMI_ALT_URL`: defaults to `http://mghasemi.ddns.net:8899` and is used automatically if the default primary URL fails
- `ZIMI_TIMEOUT`: defaults to `20`
- `ZIMI_PREFERRED_SOURCES`: comma-separated source names used as ranking preferences

If you override the primary URL on the CLI with `--url`, fallback is disabled unless you also provide `--alt-url`.

Examples:

```bash
export ZIMI_URL=http://192.168.1.70:8899
export ZIMI_ALT_URL=http://mghasemi.ddns.net:8899
export ZIMI_TIMEOUT=20
export ZIMI_PREFERRED_SOURCES=wikipedia_en_mathematics_nopic,planetmath.org
```

## Output

- `suggest` returns matching titles.
- `search` returns title, path, zim, and snippet.
- `read` returns plain text or JSON.
- `retrieve` returns the selected article plus short provenance so downstream agent steps know what was read.
- `search` and `retrieve` accept `--prefer-source` to bias ranking toward preferred ZIM sources without overriding relevance.