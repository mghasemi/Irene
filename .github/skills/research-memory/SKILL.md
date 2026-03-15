---
name: research-memory
description: "Use when you need to save or recall small research ideas, observations, conjectures, proof sketches, coding insights, or valuable web search findings. Triggers include: save this idea, remember this, log this observation, store this web result, recall my previous note, find that insight I wrote earlier, and retrieve prior research memory."
---

# Research Memory Skill
Tiny local memory skill for scientific coding and mathematical research notes.

## Usage
Execute the tool script from this skill folder.

### Initialize storage
```bash
python3 {baseDir}/research_memory_tool.py init
```

### Save a research idea
```bash
python3 {baseDir}/research_memory_tool.py add-idea \
  --title "Potential Lyapunov candidate" \
  --observation "Tried weighted quadratic around equilibrium; stable in numerics." \
  --why "May simplify proof for section 3" \
  --tags "stability,lyapunov,section3" \
  --context "Irene project"
```

### Save a web finding
```bash
python3 {baseDir}/research_memory_tool.py add-web \
  --query "sum of squares sparse polynomial relaxations" \
  --url "https://example.org/paper" \
  --title "Sparse SOS survey" \
  --summary "Good section on chordal decomposition assumptions." \
  --why "Useful for theorem assumptions comparison" \
  --tags "sos,sparsity,reference"
```

### Search memory
```bash
python3 {baseDir}/research_memory_tool.py search --query "chordal decomposition" --limit 10
```

### Search only ideas or only web findings
```bash
python3 {baseDir}/research_memory_tool.py search --query "lyapunov" --kind idea
python3 {baseDir}/research_memory_tool.py search --query "sos survey" --kind web
```

### Show recent memories
```bash
python3 {baseDir}/research_memory_tool.py recent --limit 20
```

### Read one memory
```bash
python3 {baseDir}/research_memory_tool.py get 12
```

### Update one memory
```bash
python3 {baseDir}/research_memory_tool.py update 12 --title "Updated title" --tags "proof,lemma"
```

### Delete one memory
```bash
python3 {baseDir}/research_memory_tool.py delete 12
```

## Configuration
Optional environment variable:
- RESEARCH_MEMORY_DB: custom path for the sqlite database file
