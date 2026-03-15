# Research Memory In Practice (Short Note)

Use these commands from the repository root (`/home/mehdi/Code/Irene`).

## 1) Save a tiny idea immediately
```bash
python3 .github/skills/research-memory/quick_capture.py idea "I can bound this polynomial by splitting the domain" --tags "bound,idea"
```

## 2) Save a valuable web result with reproducible context
```bash
python3 .github/skills/research-memory/quick_capture.py web "sparse sonc relaxations" "https://example.org/paper" "Useful assumptions and theorem statement" --tags "sonc,reference"
```

## 3) Save end-of-session summary in one line
```bash
python3 .github/skills/research-memory/quick_capture.py session "Tested 3 approaches; method 2 is stable; revisit proof details tomorrow"
```

## 4) Recall what you saved
```bash
python3 .github/skills/research-memory/quick_capture.py find "stable method 2"
```

## 5) See latest memories before starting work
```bash
python3 .github/skills/research-memory/quick_capture.py recent --limit 10
```

## Suggested habit
- Capture immediately when you discover something non-obvious.
- Add 1-3 tags each time.
- Run `recent` at the start of each coding session.
- Run `session` once before you stop for the day.
