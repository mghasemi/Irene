#!/usr/bin/env python3
"""CLI for querying a running LightRAG HTTP service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen


DEFAULT_URL = "http://192.168.1.70:9621"
DEFAULT_ALT_URL = "http://mghasemi.ddns.net:9621"
DEFAULT_TIMEOUT = 20.0
VALID_MODES = ("local", "global", "hybrid", "naive", "mix", "bypass")


@dataclass
class LightRAGClient:
    base_url: str
    timeout: float
    fallback_url: str | None = None
    api_key: str | None = None

    def candidate_base_urls(self) -> tuple[str, ...]:
        candidates = [self.base_url]
        if self.fallback_url and self.fallback_url != self.base_url:
            candidates.append(self.fallback_url)
        return tuple(candidates)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    def post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        errors: list[str] = []

        for base_url in self.candidate_base_urls():
            url = f"{base_url.rstrip('/')}{endpoint}"
            request = Request(url=url, data=body, headers=self._build_headers(), method="POST")
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    raw = response.read().decode(charset, errors="replace")
                return json.loads(raw)
            except HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                errors.append(f"{base_url}: HTTP {exc.code} from {endpoint}: {message}")
            except URLError as exc:
                errors.append(f"{base_url}: {exc.reason}")
            except json.JSONDecodeError as exc:
                errors.append(f"{base_url}: Invalid JSON from {endpoint}: {exc}")

        if len(errors) == 1:
            raise RuntimeError(f"Unable to reach LightRAG server at {self.base_url}: {errors[0].split(': ', 1)[1]}")
        raise RuntimeError("Unable to reach LightRAG servers. Tried: " + " | ".join(errors))

    def post_stream(self, endpoint: str, payload: dict[str, Any]) -> list[str]:
        body = json.dumps(payload).encode("utf-8")
        errors: list[str] = []

        for base_url in self.candidate_base_urls():
            url = f"{base_url.rstrip('/')}{endpoint}"
            request = Request(url=url, data=body, headers=self._build_headers(), method="POST")
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    raw = response.read().decode(charset, errors="replace")
                return [line for line in raw.splitlines() if line.strip()]
            except HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                errors.append(f"{base_url}: HTTP {exc.code} from {endpoint}: {message}")
            except URLError as exc:
                errors.append(f"{base_url}: {exc.reason}")

        if len(errors) == 1:
            raise RuntimeError(f"Unable to reach LightRAG server at {self.base_url}: {errors[0].split(': ', 1)[1]}")
        raise RuntimeError("Unable to reach LightRAG servers. Tried: " + " | ".join(errors))


def parse_keywords(values: Iterable[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        for token in value.split(","):
            item = token.strip()
            if item:
                out.append(item)
    return out or None


def build_query_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": args.query,
        "mode": args.mode,
    }
    if args.include_references is not None:
        payload["include_references"] = args.include_references
    if args.include_chunk_content is not None:
        payload["include_chunk_content"] = args.include_chunk_content
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.chunk_top_k is not None:
        payload["chunk_top_k"] = args.chunk_top_k
    if args.response_type:
        payload["response_type"] = args.response_type

    hl_keywords = parse_keywords(args.hl_keywords)
    ll_keywords = parse_keywords(args.ll_keywords)
    if hl_keywords:
        payload["hl_keywords"] = hl_keywords
    if ll_keywords:
        payload["ll_keywords"] = ll_keywords
    return payload


def format_query_text(payload: dict[str, Any]) -> str:
    answer = str(payload.get("response", "")).strip()
    references = payload.get("references", [])
    chunks = []
    if answer:
        chunks.append(answer)
    if isinstance(references, list) and references:
        chunks.append("\nReferences:")
        for index, ref in enumerate(references, start=1):
            if not isinstance(ref, dict):
                continue
            ref_id = ref.get("reference_id", "")
            file_path = ref.get("file_path", "")
            chunks.append(f"{index}. {ref_id} - {file_path}".strip())
    return "\n".join(chunks) if chunks else json.dumps(payload, ensure_ascii=True)


def build_client(args: argparse.Namespace) -> LightRAGClient:
    base_url = args.url or os.getenv("LIGHTRAG_URL", DEFAULT_URL)
    fallback_url = args.alt_url or os.getenv("LIGHTRAG_ALT_URL", DEFAULT_ALT_URL)
    timeout_raw = str(args.timeout) if args.timeout is not None else os.getenv("LIGHTRAG_TIMEOUT", str(DEFAULT_TIMEOUT))
    api_key = args.api_key or os.getenv("LIGHTRAG_API_KEY", "").strip() or None

    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid LIGHTRAG_TIMEOUT value: {timeout_raw}") from exc

    if fallback_url == base_url:
        fallback_url = None
    elif args.url and not args.alt_url:
        fallback_url = None
    elif base_url != DEFAULT_URL and not args.url:
        fallback_url = None

    return LightRAGClient(base_url=base_url, timeout=timeout, fallback_url=fallback_url, api_key=api_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query a LightRAG HTTP service")
    parser.add_argument("--url", help="Override LightRAG server URL")
    parser.add_argument("--alt-url", help="Optional alternate LightRAG server URL used when primary fails")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds")
    parser.add_argument("--api-key", help="Optional API key for protected LightRAG deployments")

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_query_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("query", help="Query text")
        subparser.add_argument("--mode", choices=VALID_MODES, default="mix")
        subparser.add_argument("--include-references", action=argparse.BooleanOptionalAction, default=None)
        subparser.add_argument("--include-chunk-content", action=argparse.BooleanOptionalAction, default=None)
        subparser.add_argument("--top-k", type=int)
        subparser.add_argument("--chunk-top-k", type=int)
        subparser.add_argument("--response-type")
        subparser.add_argument("--hl-keywords", action="append", default=[])
        subparser.add_argument("--ll-keywords", action="append", default=[])

    query_parser = subparsers.add_parser("query", help="Query LightRAG and get generated response")
    add_common_query_args(query_parser)
    query_parser.add_argument("--format", choices=("text", "json"), default="text")

    query_data_parser = subparsers.add_parser("query-data", help="Retrieve structured query data only")
    add_common_query_args(query_data_parser)
    query_data_parser.add_argument("--format", choices=("text", "json"), default="json")

    query_stream_parser = subparsers.add_parser("query-stream", help="Query LightRAG stream endpoint")
    add_common_query_args(query_stream_parser)
    query_stream_parser.add_argument("--format", choices=("text", "json"), default="text")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        client = build_client(args)
        payload = build_query_payload(args)

        if args.command == "query":
            response = client.post_json("/query", payload)
            if args.format == "json":
                print(json.dumps(response, indent=2, ensure_ascii=True))
            else:
                print(format_query_text(response if isinstance(response, dict) else {"response": response}))
            return 0

        if args.command == "query-data":
            response = client.post_json("/query/data", payload)
            if args.format == "json":
                print(json.dumps(response, indent=2, ensure_ascii=True))
            else:
                print(format_query_text(response if isinstance(response, dict) else {"response": response}))
            return 0

        if args.command == "query-stream":
            payload["stream"] = True
            lines = client.post_stream("/query/stream", payload)
            if args.format == "json":
                parsed_lines: list[Any] = []
                for line in lines:
                    try:
                        parsed_lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        parsed_lines.append({"raw": line})
                print(json.dumps(parsed_lines, indent=2, ensure_ascii=True))
            else:
                for line in lines:
                    print(line)
            return 0

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
