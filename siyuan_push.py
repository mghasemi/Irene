#!/usr/bin/env python3
"""Push the full IreneRewrite master plan into Siyuan as a structured document."""

import sys
sys.path.insert(0, "/home/mehdi/.hermes/profiles/math/skills/productivity/siyuan")
from siyuan_tool import _request, _resolve_notebook_id, create_doc_with_md

# Load env vars the same way the tool does
import os
from pathlib import Path

current = Path("/home/mehdi/.hermes/profiles/math/skills/productivity/siyuan")
while True:
    emv_path = current / ".emv"
    if emv_path.is_file():
        for raw_line in emv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value
        break
    if current.parent == current:
        break
    current = current.parent

SIYUAN_URL = os.environ.get("SIYUAN_URL", "http://192.168.1.70:6806")
SIYUAN_TOKEN = os.environ.get("SIYUAN_TOKEN", "")
urls = [SIYUAN_URL]

# Read the full plan
with open("/home/mehdi/Code/Python/IreneRewrite/plan_master.md") as f:
    plan_content = f.read()

# Resolve notebook ID (auto-detect first open notebook)
try:
    notebook_id = _resolve_notebook_id("", SIYUAN_TOKEN, urls)
    print(f"Using notebook: {notebook_id}")
except Exception as e:
    print(f"Notebook resolution failed: {e}")
    sys.exit(1)

# Create the document
try:
    result = create_doc_with_md(notebook_id, "/IreneRewrite/plan_master", plan_content, SIYUAN_TOKEN, urls)
    print(f"Document created successfully!")
    print(f"Response: {result}")
except Exception as e:
    print(f"Create-doc failed: {e}")
    # Try creating in root path instead
    try:
        result = _request("/api/filetree/createDocWithMd", {
            "notebook": notebook_id,
            "path": "/IreneRewrite/plan_master_v2",
            "markdown": plan_content
        }, SIYUAN_TOKEN, urls)
        print(f"Alternative path succeeded: {result}")
    except Exception as e2:
        print(f"All Siyuan writes failed: {e2}")
