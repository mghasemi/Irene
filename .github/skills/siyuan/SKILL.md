---
name: siyuan
description: "Use when working with SiYuan notes or blocks, especially when asked to search notes, search my notes, find notes, find my notes, get note content, ingest note content, open a note, retrieve a SiYuan block by id, or look up what my notes say about a topic."
---

# SiYuan Access Skill
Skill to search and ingest notes from a running SiYuan instance.

## Usage
Execute the python script `siyuan_tool.py` located in the skill folder.

### Search for notes
\`\`\`bash
python3 {baseDir}/siyuan_tool.py search "your query"
\`\`\`

### Ingest note content
\`\`\`bash
python3 {baseDir}/siyuan_tool.py get "block_id"
\`\`\`
