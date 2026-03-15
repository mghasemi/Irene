---
name: calibre
description: "Use when working with a Calibre library or content server, especially when asked to search books, search my library, search articles, find a book, find books by keyword, find books by author, look up items in Calibre, or search my Calibre library for a title or author."
---

# Calibre Access Skill
Skill to search for books and articles in a Calibre Content Server instance.

## Usage
Execute `calibre_tool.py` with search terms.

### Search by keywords
\`\`\`bash
python3 {baseDir}/calibre_tool.py "Foundation"
\`\`\`

### Search by author
\`\`\`bash
python3 {baseDir}/calibre_tool.py "author:Asimov"
\`\`\`
