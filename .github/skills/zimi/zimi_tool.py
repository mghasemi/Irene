#!/usr/bin/env python3
"""CLI for searching and reading content from a ZIMI server."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Sequence
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_URL = "http://192.168.1.70:8899"
DEFAULT_ALT_URL = "http://mghasemi.ddns.net:8899"
DEFAULT_TIMEOUT = 20.0
DEFAULT_SOURCE_PREFERENCES = (
    "wikipedia_en_mathematics_nopic",
    "planetmath.org",
)
SOURCE_WEIGHT_RULES = (
    ("wikipedia_en_mathematics_nopic", 2.5),
    ("planetmath.org", 1.5),
    ("wikipedia_", 1.0),
    ("mathoverflow.net", -0.75),
    ("stackexchange", -1.0),
)
TITLE_HINT_WORDS = {
    "article",
    "entry",
    "page",
    "topic",
    "definition",
    "theorem",
    "conjecture",
    "lemma",
    "space",
    "group",
    "manifold",
}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "who",
    "why",
    "with",
}


def normalize_suggest_response(data: Any) -> list[str]:
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        suggestions: list[str] = []
        for items in data.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    title = str(item.get("title", "")).strip()
                    if title:
                        suggestions.append(title)
                elif isinstance(item, str):
                    title = item.strip()
                    if title:
                        suggestions.append(title)
        return suggestions
    raise RuntimeError("Unexpected /suggest response format")


def normalize_search_response(data: Any) -> list[dict[str, Any]]:
    payload = data
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            payload = payload["results"]
        elif "items" in payload and isinstance(payload["items"], list):
            payload = payload["items"]
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected /search response format")
    return [item for item in payload if isinstance(item, dict)]


def extract_read_text(payload: str) -> str:
    stripped = payload.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return payload
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data["error"]))
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            return str(data["text"])
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            return str(data["content"])
    return payload


def parse_source_preferences(values: Sequence[str] | None) -> tuple[str, ...]:
    raw_values = list(values or [])
    env_value = os.getenv("ZIMI_PREFERRED_SOURCES", "").strip()
    if env_value:
        raw_values.append(env_value)
    if not raw_values:
        return DEFAULT_SOURCE_PREFERENCES
    preferences: list[str] = []
    for value in raw_values:
        for item in value.split(","):
            source = item.strip()
            if source:
                preferences.append(source)
    return tuple(preferences) if preferences else DEFAULT_SOURCE_PREFERENCES


def source_quality_score(zim_name: str) -> float:
    lowered = zim_name.lower()
    score = 0.0
    for fragment, weight in SOURCE_WEIGHT_RULES:
        if fragment in lowered:
            score += weight
    return score


def source_preference_score(zim_name: str, preferred_sources: Sequence[str]) -> float:
    lowered = zim_name.lower()
    for index, preferred in enumerate(preferred_sources):
        preferred_lowered = preferred.lower()
        if lowered == preferred_lowered:
            return 4.0 - (index * 0.5)
        if preferred_lowered in lowered:
            return 2.5 - (index * 0.25)
    return 0.0


@dataclass
class ZimiClient:
    base_url: str
    timeout: float
    fallback_url: str | None = None

    def candidate_base_urls(self) -> tuple[str, ...]:
        candidates = [self.base_url]
        if self.fallback_url and self.fallback_url != self.base_url:
            candidates.append(self.fallback_url)
        return tuple(candidates)

    def get_json(self, endpoint: str, **params: Any) -> Any:
        payload = self.get_text(endpoint, **params)
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from {endpoint}: {exc}") from exc

    def get_text(self, endpoint: str, **params: Any) -> str:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        errors: list[str] = []
        for base_url in self.candidate_base_urls():
            url = f"{base_url.rstrip('/')}{endpoint}"
            if query:
                url = f"{url}?{query}"

            try:
                with urlopen(url, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    body = response.read()
                return body.decode(charset, errors="replace")
            except HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
                errors.append(f"{base_url}: HTTP {exc.code} from {endpoint}: {message}")
            except URLError as exc:
                errors.append(f"{base_url}: {exc.reason}")

        if len(errors) == 1:
            raise RuntimeError(f"Unable to reach ZIMI server at {self.base_url}: {errors[0].split(': ', 1)[1]}")
        raise RuntimeError("Unable to reach ZIMI servers. Tried: " + " | ".join(errors))

    def suggest(self, query: str) -> list[str]:
        data = self.get_json("/suggest", q=query)
        return normalize_suggest_response(data)

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        data = self.get_json("/search", q=query, limit=limit)
        return normalize_search_response(data)

    def read(self, zim: str, path: str, max_length: int | None = None) -> str:
        payload = self.get_text("/read", zim=zim, path=path, max_length=max_length)
        return extract_read_text(payload)


def build_client(
    base_url_override: str | None = None,
    fallback_url_override: str | None = None,
) -> ZimiClient:
    base_url = base_url_override or os.getenv("ZIMI_URL", DEFAULT_URL)
    fallback_url = fallback_url_override or os.getenv("ZIMI_ALT_URL", DEFAULT_ALT_URL)
    timeout_raw = os.getenv("ZIMI_TIMEOUT", str(DEFAULT_TIMEOUT))
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid ZIMI_TIMEOUT value: {timeout_raw}") from exc
    if fallback_url == base_url:
        fallback_url = None
    elif base_url_override and fallback_url_override is None:
        fallback_url = None
    elif base_url != DEFAULT_URL and base_url_override is None:
        fallback_url = None
    return ZimiClient(base_url=base_url, timeout=timeout, fallback_url=fallback_url)


def is_title_like(query: str) -> bool:
    stripped = query.strip()
    words = stripped.split()
    if not stripped:
        return False
    if len(words) <= 4 and not stripped.endswith("?"):
        return True
    lowered = stripped.lower()
    if any(phrase in lowered for phrase in ("article on ", "page for ", "entry on ", "title ")):
        return True
    if stripped.startswith('"') and stripped.endswith('"'):
        return True
    capitalized = sum(1 for word in words if word[:1].isupper())
    if 0 < capitalized == len(words) <= 5:
        return True
    return False


def normalize_title_query(query: str) -> str:
    lowered = query.strip()
    lowered = re.sub(r"^(article on|page for|entry on|look up|find)\s+", "", lowered, flags=re.IGNORECASE)
    return lowered.strip().strip('"')


def make_rewrites(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9_+-]+", query)
    filtered = [term for term in terms if term.lower() not in STOP_WORDS]
    rewrites: list[str] = []
    compact = " ".join(filtered)
    if compact and compact != query:
        rewrites.append(compact)
    hint_terms = [term for term in filtered if term.lower() in TITLE_HINT_WORDS or len(term) > 3]
    compact_hints = " ".join(hint_terms)
    if compact_hints and compact_hints not in rewrites and compact_hints != query:
        rewrites.append(compact_hints)
    return rewrites[:2]


def title_similarity(query: str, title: str) -> float:
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    title_words = set(re.findall(r"[a-z0-9]+", title.lower()))
    if not query_words or not title_words:
        return 0.0
    overlap = len(query_words & title_words)
    union = len(query_words | title_words)
    return overlap / union if union else 0.0


def lexical_overlap_score(query: str, result: dict[str, Any]) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if term not in STOP_WORDS}
    if not query_terms:
        return 0.0
    searchable_text = " ".join(
        [
            str(result.get("title", "")),
            str(result.get("snippet", "")),
            str(result.get("path", "")).replace("/", " ").replace("_", " "),
        ]
    ).lower()
    matches = 0.0
    for term in query_terms:
        if term in searchable_text:
            matches += 1.0
    return matches / len(query_terms)


def score_result(query: str, result: dict[str, Any], preferred_sources: Sequence[str] = DEFAULT_SOURCE_PREFERENCES) -> float:
    title = str(result.get("title", ""))
    snippet = str(result.get("snippet", ""))
    zim_name = str(result.get("zim", ""))
    score = 0.0
    similarity = title_similarity(query, title)
    overlap = lexical_overlap_score(query, result)
    score += similarity * 3.0
    score += overlap * 4.0
    lowered_query = query.lower()
    if title.lower() == lowered_query:
        score += 4.0
    elif title.lower().startswith(lowered_query):
        score += 2.0
    if lowered_query in snippet.lower():
        score += 1.5
    if result.get("path") and result.get("zim"):
        score += 0.25
    if overlap > 0.0 or similarity > 0.0:
        score += source_quality_score(zim_name)
        score += source_preference_score(zim_name, preferred_sources)
    return score


def best_search_results(
    query: str,
    results: Sequence[dict[str, Any]],
    preferred_sources: Sequence[str] = DEFAULT_SOURCE_PREFERENCES,
) -> list[dict[str, Any]]:
    ranked = sorted(results, key=lambda item: score_result(query, item, preferred_sources), reverse=True)
    return ranked


def resolve_title_candidates(
    client: ZimiClient,
    query: str,
    search_limit: int,
    preferred_sources: Sequence[str],
) -> list[dict[str, Any]]:
    suggestions = client.suggest(normalize_title_query(query))
    resolved: list[dict[str, Any]] = []
    for suggestion in suggestions[:3]:
        search_hits = client.search(suggestion, limit=search_limit)
        ranked = best_search_results(suggestion, search_hits, preferred_sources)
        if ranked:
            resolved.append(ranked[0])
    return best_search_results(query, dedupe_results(resolved), preferred_sources)


def dedupe_results(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for result in results:
        key = (str(result.get("zim", "")), str(result.get("path", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def retrieve_article(
    client: ZimiClient,
    query: str,
    max_length: int,
    search_limit: int,
    read_count: int,
    preferred_sources: Sequence[str],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    if is_title_like(query):
        title_candidates = resolve_title_candidates(client, query, search_limit, preferred_sources)
        attempts.append({"mode": "suggest", "query": normalize_title_query(query), "count": len(title_candidates)})
        candidates.extend(title_candidates)

    search_results = client.search(query, limit=search_limit)
    attempts.append({"mode": "search", "query": query, "count": len(search_results)})
    candidates.extend(best_search_results(query, search_results, preferred_sources))

    ranked = best_search_results(query, dedupe_results(candidates), preferred_sources)
    if not ranked or score_result(query, ranked[0], preferred_sources) < 1.0:
        for rewrite in make_rewrites(query):
            rewrite_results = client.search(rewrite, limit=search_limit)
            attempts.append({"mode": "rewrite-search", "query": rewrite, "count": len(rewrite_results)})
            ranked.extend(best_search_results(rewrite, rewrite_results, preferred_sources))
            ranked = best_search_results(query, dedupe_results(ranked), preferred_sources)
            if ranked and score_result(query, ranked[0], preferred_sources) >= 1.0:
                break

    ranked = ranked[: max(1, read_count)]
    if not ranked:
        return {
            "query": query,
            "attempts": attempts,
            "selected": [],
            "message": "No ZIMI articles matched the query.",
        }

    selected: list[dict[str, Any]] = []
    for result in ranked:
        zim = str(result.get("zim", ""))
        path = str(result.get("path", ""))
        text = client.read(zim=zim, path=path, max_length=max_length)
        selected.append(
            {
                "title": result.get("title", ""),
                "zim": zim,
                "path": path,
                "snippet": result.get("snippet", ""),
                "score": round(score_result(query, result, preferred_sources), 3),
                "text": text,
            }
        )

    return {
        "query": query,
        "attempts": attempts,
        "selected": selected,
    }


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=True))


def print_search_results(query: str, results: Sequence[dict[str, Any]], preferred_sources: Sequence[str]) -> None:
    if not results:
        print("No results.")
        return
    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result.get('title', '')}")
        print(f"  zim: {result.get('zim', '')}")
        print(f"  path: {result.get('path', '')}")
        if result.get("score") is not None:
            print(f"  source-score: {result.get('score')}")
        print(f"  rank-score: {round(score_result(query, result, preferred_sources), 3)}")
        snippet = str(result.get("snippet", "")).strip()
        if snippet:
            print(f"  snippet: {snippet}")


def print_retrieve_output(payload: dict[str, Any]) -> None:
    selected = payload.get("selected", [])
    if not selected:
        print(payload.get("message", "No results."))
        return
    print(f"Query: {payload.get('query', '')}")
    attempts = payload.get("attempts", [])
    if attempts:
        rendered = ", ".join(
            f"{item.get('mode')}[{item.get('query')}]={item.get('count')}" for item in attempts
        )
        print(f"Attempts: {rendered}")
    for index, item in enumerate(selected, start=1):
        print()
        print(f"Result {index}: {item.get('title', '')}")
        print(f"ZIM: {item.get('zim', '')}")
        print(f"Path: {item.get('path', '')}")
        print(f"Score: {item.get('score', 0)}")
        snippet = str(item.get("snippet", "")).strip()
        if snippet:
            print(f"Snippet: {snippet}")
        print()
        print(item.get("text", ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and read content from a ZIMI server")
    parser.add_argument("--url", help="Override ZIMI server URL")
    parser.add_argument("--alt-url", help="Optional alternate ZIMI server URL used if the primary URL fails")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds")

    subparsers = parser.add_subparsers(dest="command", required=True)

    suggest_parser = subparsers.add_parser("suggest", help="Suggest article titles")
    suggest_parser.add_argument("query", help="Title prefix")
    suggest_parser.add_argument("--format", choices=("text", "json"), default="text")

    search_parser = subparsers.add_parser("search", help="Full-text search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument(
        "--prefer-source",
        action="append",
        default=[],
        help="Preferred source/ZIM name. Repeat or comma-separate values to bias ranking.",
    )
    search_parser.add_argument("--format", choices=("text", "json"), default="text")

    read_parser = subparsers.add_parser("read", help="Read an article")
    read_parser.add_argument("--zim", required=True)
    read_parser.add_argument("--path", required=True)
    read_parser.add_argument("--max-length", type=int, default=4000)
    read_parser.add_argument("--format", choices=("text", "json"), default="text")

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve the best matching article text")
    retrieve_parser.add_argument("query", help="User query")
    retrieve_parser.add_argument("--max-length", type=int, default=5000)
    retrieve_parser.add_argument("--search-limit", type=int, default=5)
    retrieve_parser.add_argument("--read-count", type=int, default=1)
    retrieve_parser.add_argument(
        "--prefer-source",
        action="append",
        default=[],
        help="Preferred source/ZIM name. Repeat or comma-separate values to bias ranking.",
    )
    retrieve_parser.add_argument("--format", choices=("text", "json"), default="text")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        client = build_client(base_url_override=args.url, fallback_url_override=args.alt_url)
        if args.timeout is not None:
            client.timeout = args.timeout

        if args.command == "suggest":
            suggestions = client.suggest(args.query)
            if args.format == "json":
                print_json(suggestions)
            else:
                for title in suggestions:
                    print(title)
            return 0

        if args.command == "search":
            preferred_sources = parse_source_preferences(args.prefer_source)
            results = best_search_results(
                args.query,
                client.search(args.query, limit=args.limit),
                preferred_sources,
            )
            if args.format == "json":
                print_json(results)
            else:
                print_search_results(args.query, results, preferred_sources)
            return 0

        if args.command == "read":
            text = client.read(args.zim, args.path, max_length=args.max_length)
            if args.format == "json":
                print_json(
                    {
                        "zim": args.zim,
                        "path": args.path,
                        "max_length": args.max_length,
                        "text": text,
                    }
                )
            else:
                print(text)
            return 0

        if args.command == "retrieve":
            preferred_sources = parse_source_preferences(args.prefer_source)
            payload = retrieve_article(
                client=client,
                query=args.query,
                max_length=args.max_length,
                search_limit=args.search_limit,
                read_count=args.read_count,
                preferred_sources=preferred_sources,
            )
            if args.format == "json":
                print_json(payload)
            else:
                print_retrieve_output(payload)
            return 0

        parser.error(f"Unsupported command: {args.command}")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())