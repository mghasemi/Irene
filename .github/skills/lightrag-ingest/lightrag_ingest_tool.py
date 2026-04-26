#!/usr/bin/env python3
"""CLI for ingesting content into a running LightRAG knowledge graph."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen


DEFAULT_URL = os.environ.get("LIGHTRAG_URL", "http://192.168.1.70:9621")
DEFAULT_ALT_URL = os.environ.get("LIGHTRAG_ALT_URL", "http://mghasemi.ddns.net:9621")
DEFAULT_TIMEOUT = float(os.environ.get("LIGHTRAG_TIMEOUT", "60"))
DEFAULT_API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")

ARXIV_ABSTRACT_URL = "https://export.arxiv.org/abs/{}"
ARXIV_HTML_URL = "https://ar5iv.org/abs/{}"


def _fail(message: str, fmt: str = "text") -> None:
    if fmt == "json":
        print(json.dumps({"success": False, "error": message}))
    else:
        print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _build_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if DEFAULT_API_KEY:
        headers["Authorization"] = f"Bearer {DEFAULT_API_KEY}"
    return headers


def _post(endpoint: str, payload: dict[str, Any], fmt: str) -> Any:
    body = json.dumps(payload).encode("utf-8")
    errors: list[str] = []
    for base in [DEFAULT_URL, DEFAULT_ALT_URL]:
        if not base:
            continue
        url = f"{base.rstrip('/')}{endpoint}"
        req = Request(url=url, data=body, headers=_build_headers(), method="POST")
        try:
            with urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return json.loads(resp.read().decode(charset, errors="replace"))
        except HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            errors.append(f"{base}: HTTP {exc.code}: {msg}")
        except URLError as exc:
            errors.append(f"{base}: {exc.reason}")
        except json.JSONDecodeError as exc:
            errors.append(f"{base}: Invalid JSON: {exc}")
    _fail("Unable to reach LightRAG. Tried: " + " | ".join(errors), fmt)


def _get(endpoint: str, fmt: str) -> Any:
    errors: list[str] = []
    for base in [DEFAULT_URL, DEFAULT_ALT_URL]:
        if not base:
            continue
        url = f"{base.rstrip('/')}{endpoint}"
        req = Request(url=url, headers=_build_headers(), method="GET")
        try:
            with urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return json.loads(resp.read().decode(charset, errors="replace"))
        except (HTTPError, URLError) as exc:
            errors.append(f"{base}: {exc}")
    _fail("Unable to reach LightRAG. Tried: " + " | ".join(errors), fmt)


def _ingest_text(text: str, fmt: str) -> dict[str, Any]:
    result = _post("/documents/text", {"text": text}, fmt)
    return result or {"success": True}


def _fetch_url_text(url: str, fmt: str) -> str:
    try:
        with urlopen(url, timeout=30) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            raw = resp.read().decode(charset, errors="replace")
        # Strip HTML tags crudely
        import re
        return re.sub(r"<[^>]+>", " ", raw)
    except Exception as exc:
        _fail(f"Failed to fetch {url}: {exc}", fmt)
        return ""


def cmd_ingest_text(args: argparse.Namespace) -> dict[str, Any]:
    result = _ingest_text(args.text, args.format)
    return {
        "command": "ingest-text",
        "chars_ingested": len(args.text),
        "server_response": result,
        "success": True,
    }


def cmd_ingest_pdf(args: argparse.Namespace) -> dict[str, Any]:
    if not os.path.isfile(args.file):
        _fail(f"PDF not found: {args.file}", args.format)

    text = ""
    if shutil.which("pdftotext"):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(["pdftotext", args.file, tmp_path], check=True, capture_output=True)
            with open(tmp_path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # Pure-Python fallback using no deps (read raw bytes, extract printable ASCII)
        with open(args.file, "rb") as f:
            raw = f.read()
        import re
        chunks = re.findall(rb"\(([^\)]{4,})\)", raw)
        text = " ".join(c.decode("latin-1", errors="ignore") for c in chunks)

    if not text.strip():
        _fail("Could not extract text from PDF.", args.format)

    result = _ingest_text(text, args.format)
    return {
        "command": "ingest-pdf",
        "file": args.file,
        "chars_ingested": len(text),
        "server_response": result,
        "success": True,
    }


def cmd_ingest_arxiv(args: argparse.Namespace) -> dict[str, Any]:
    arxiv_id = args.arxiv_id.strip().lstrip("arXiv:").lstrip("arxiv:")
    # Fetch abstract page text as primary content
    url = ARXIV_ABSTRACT_URL.format(arxiv_id)
    text = _fetch_url_text(url, args.format)
    if len(text.strip()) < 100:
        # Fallback to ar5iv HTML
        url = ARXIV_HTML_URL.format(arxiv_id)
        text = _fetch_url_text(url, args.format)

    title_match = __import__("re").search(r"Title:\s*(.+)", text)
    title = title_match.group(1).strip() if title_match else f"arXiv:{arxiv_id}"

    result = _ingest_text(text, args.format)
    return {
        "command": "ingest-arxiv",
        "arxiv_id": arxiv_id,
        "title": title,
        "chars_ingested": len(text),
        "server_response": result,
        "success": True,
    }


def cmd_ingest_url(args: argparse.Namespace) -> dict[str, Any]:
    text = _fetch_url_text(args.url, args.format)
    result = _ingest_text(text, args.format)
    return {
        "command": "ingest-url",
        "url": args.url,
        "chars_ingested": len(text),
        "server_response": result,
        "success": True,
    }


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    # Try common LightRAG status/health endpoints
    for ep in ["/graph/stats", "/status", "/health"]:
        try:
            data = _get(ep, args.format)
            if data:
                return {"command": "status", "endpoint": ep, **data, "success": True}
        except SystemExit:
            continue
    return {"command": "status", "success": False, "error": "No status endpoint responded"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightRAG ingestion CLI")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    sub = parser.add_subparsers(dest="command", required=True)

    p_text = sub.add_parser("ingest-text", help="Ingest a text string")
    p_text.add_argument("text")
    p_text.add_argument("--format", choices=["text", "json"], default="text")

    p_pdf = sub.add_parser("ingest-pdf", help="Ingest a PDF file")
    p_pdf.add_argument("file")
    p_pdf.add_argument("--format", choices=["text", "json"], default="text")

    p_arxiv = sub.add_parser("ingest-arxiv", help="Ingest an arXiv paper by ID")
    p_arxiv.add_argument("arxiv_id")
    p_arxiv.add_argument("--format", choices=["text", "json"], default="text")

    p_url = sub.add_parser("ingest-url", help="Ingest text from a URL")
    p_url.add_argument("url")
    p_url.add_argument("--format", choices=["text", "json"], default="text")

    p_status = sub.add_parser("status", help="Check LightRAG graph status")
    p_status.add_argument("--format", choices=["text", "json"], default="text")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fmt = getattr(args, "format", "text")

    dispatch = {
        "ingest-text": cmd_ingest_text,
        "ingest-pdf": cmd_ingest_pdf,
        "ingest-arxiv": cmd_ingest_arxiv,
        "ingest-url": cmd_ingest_url,
        "status": cmd_status,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        _fail(f"Unknown command: {args.command}", fmt)

    try:
        data = handler(args)
    except Exception as exc:
        _fail(str(exc), fmt)
        return

    if fmt == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        if not data.get("success"):
            print(f"ERROR: {data.get('error', 'Failed')}", file=sys.stderr)
            sys.exit(1)
        if args.command == "status":
            for k, v in data.items():
                if k not in ("command", "success", "endpoint"):
                    print(f"  {k}: {v}")
        else:
            print(f"OK: {data.get('chars_ingested', '?')} chars ingested")


if __name__ == "__main__":
    main()
