---
name: vikunja
description: "Use when working with Vikunja projects or tasks, especially when asked to show my projects, list projects, read project details, show my tasks, list tasks, read task details, create a project, update a project, create a task, add a task, create a todo, update a task, complete a task, or log a task."
---

# Vikunja Access Skill
Skill to read and write projects and tasks in a Vikunja server.

## Usage
Execute `vikunja_tool.py` from this skill folder.

### List all projects
```bash
python3 {baseDir}/vikunja_tool.py projects list
```

### Search projects
```bash
python3 {baseDir}/vikunja_tool.py projects search --query "research"
```

### Get one project by title
```bash
python3 {baseDir}/vikunja_tool.py projects get-by-title --title "Research Backlog"
```

### Create a project
```bash
python3 {baseDir}/vikunja_tool.py projects create --title "Research Backlog"
```

### Update a project
```bash
python3 {baseDir}/vikunja_tool.py projects update 12 --title "Research Roadmap"
```

### List all tasks
```bash
python3 {baseDir}/vikunja_tool.py tasks list
```

### Search tasks
```bash
python3 {baseDir}/vikunja_tool.py tasks search --query "summary"
```

### List tasks in a project
```bash
python3 {baseDir}/vikunja_tool.py tasks list --project-id 12
```

### Create a task
```bash
python3 {baseDir}/vikunja_tool.py tasks create --project-id 12 --title "Write summary"
```

Date-only due dates are accepted and normalized automatically:

```bash
python3 {baseDir}/vikunja_tool.py tasks create --project-id 12 --title "Write summary" --due-date 2026-03-22
```

Priority and labels are supported directly:

```bash
python3 {baseDir}/vikunja_tool.py tasks create --project-id 12 --title "Write summary" --priority 4 --labels "writing,review"
```

Subtasks can be created under an existing parent task:

```bash
python3 {baseDir}/vikunja_tool.py tasks create --project-id 12 --title "Extract theorem list" --parent-task-id 45
```

### Log a task (alias for create)
```bash
python3 {baseDir}/vikunja_tool.py tasks log --project-id 12 --title "Capture meeting notes"
```

### Update a task
```bash
python3 {baseDir}/vikunja_tool.py tasks update 45 --done true
```

Task updates preserve existing title, description, due date, project, and priority unless you explicitly override them:

```bash
python3 {baseDir}/vikunja_tool.py tasks update 45 --priority 5
python3 {baseDir}/vikunja_tool.py tasks update 45 --labels "theory,review"
```

### Complete a task
```bash
python3 {baseDir}/vikunja_tool.py tasks complete 45
```

### Dry run a write request
```bash
python3 {baseDir}/vikunja_tool.py --dry-run tasks create --project-id 12 --title "Draft notes"
```

## Configuration
Defaults are built into the script:
- URL: `http://192.168.1.84:3456`
- ALT_URL: `http://mghasemi.ddns.net:3456`
- TOKEN: provided API token

Optional environment overrides:
- `VIKUNJA_URL`
- `VIKUNJA_ALT_URL`
- `VIKUNJA_TOKEN`
