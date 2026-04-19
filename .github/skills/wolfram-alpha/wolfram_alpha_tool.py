#!/usr/bin/env python3
"""CLI for Wolfram|Alpha short answers and structured verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_RESULT_URL = "https://api.wolframalpha.com/v1/result"
DEFAULT_QUERY_URL = "https://api.wolframalpha.com/v2/query"
DEFAULT_VALIDATE_URL = "https://api.wolframalpha.com/v2/validatequery"
DEFAULT_TIMEOUT = 20.0
DEFAULT_QUERY_FORMATS = "plaintext,minput,moutput"
VERIFY_PROFILES: dict[str, dict[str, Any]] = {
    "full": {
        "format_types": DEFAULT_QUERY_FORMATS,
        "includepodid": [],
        "podtitle": [],
    },
    "symbolic": {
        "format_types": DEFAULT_QUERY_FORMATS,
        "includepodid": ["Result", "ExactResult", "DecimalApproximation"],
        "podtitle": [
            "Input*",
            "Result",
            "Exact result",
            "Decimal form",
            "Decimal approximation",
            "Alternate form*",
            "Possible intermediate steps",
            "Root",
        ],
    },
    "theorem": {
        "format_types": DEFAULT_QUERY_FORMATS,
        "includepodid": ["Result", "Definition", "Statement"],
        "podtitle": [
            "Input*",
            "Statement",
            "Result",
            "Definition*",
            "Properties",
            "Alternate form*",
            "Proof",
        ],
    },
}


@dataclass
class WolframAlphaClient:
    appid: str
    timeout: float
    result_url: str = DEFAULT_RESULT_URL
    query_url: str = DEFAULT_QUERY_URL
    validate_url: str = DEFAULT_VALIDATE_URL

    def get_text(self, url: str, params: list[tuple[str, str]]) -> str:
        query_string = urlencode(params, doseq=True)
        request_url = f"{url}?{query_string}"
        try:
            with urlopen(request_url, timeout=self.timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"Wolfram|Alpha HTTP {exc.code}: {message}") from exc
        except URLError as exc:
            raise RuntimeError(f"Unable to reach Wolfram|Alpha: {exc.reason}") from exc

    def short_answer(self, query: str, reinterpret: bool | None = None) -> str:
        params = [("appid", self.appid), ("i", query)]
        if reinterpret is not None:
            params.append(("reinterpret", bool_to_text(reinterpret)))
        return self.get_text(self.result_url, params).strip()

    def query(self, params: list[tuple[str, str]]) -> str:
        query_params = [("appid", self.appid)] + params
        return self.get_text(self.query_url, query_params)

    def validate(self, query: str) -> str:
        params = [("appid", self.appid), ("input", query)]
        return self.get_text(self.validate_url, params)


def bool_to_text(value: bool) -> str:
    return "true" if value else "false"


def parse_bool_attr(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def repeated_values(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        item = value.strip()
        if item:
            out.append(item)
    return out


def collapse_whitespace(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def element_plain_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return collapse_whitespace("".join(node.itertext()))


def normalize_states(node: ET.Element) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    states_node = node.find("states")
    if states_node is None:
        return states
    for child in states_node:
        if child.tag == "state":
            states.append({
                "kind": "state",
                "name": child.attrib.get("name", ""),
                "input": child.attrib.get("input", ""),
            })
        elif child.tag == "statelist":
            states.append(
                {
                    "kind": "statelist",
                    "value": child.attrib.get("value", ""),
                    "delimiters": child.attrib.get("delimiters", ""),
                    "options": [
                        {
                            "name": option.attrib.get("name", ""),
                            "input": option.attrib.get("input", ""),
                        }
                        for option in child.findall("state")
                    ],
                }
            )
    return states


def normalize_subpod(node: ET.Element) -> dict[str, Any]:
    subpod = {
        "title": node.attrib.get("title", ""),
        "plaintext": element_plain_text(node.find("plaintext")),
        "minput": element_plain_text(node.find("minput")),
        "moutput": element_plain_text(node.find("moutput")),
        "states": normalize_states(node),
    }
    return subpod


def normalize_pod(node: ET.Element) -> dict[str, Any]:
    pod = {
        "title": node.attrib.get("title", ""),
        "scanner": node.attrib.get("scanner", ""),
        "id": node.attrib.get("id", ""),
        "position": node.attrib.get("position", ""),
        "error": parse_bool_attr(node.attrib.get("error")),
        "primary": parse_bool_attr(node.attrib.get("primary")) or False,
        "numsubpods": int(node.attrib.get("numsubpods", "0") or 0),
        "subpods": [normalize_subpod(subpod) for subpod in node.findall("subpod")],
        "states": normalize_states(node),
    }
    return pod


def normalize_assumptions(root: ET.Element) -> list[dict[str, Any]]:
    assumptions_node = root.find("assumptions")
    if assumptions_node is None:
        return []

    out: list[dict[str, Any]] = []
    for assumption in assumptions_node.findall("assumption"):
        out.append(
            {
                "type": assumption.attrib.get("type", ""),
                "word": assumption.attrib.get("word", ""),
                "template": assumption.attrib.get("template", ""),
                "desc": assumption.attrib.get("desc", ""),
                "current": assumption.attrib.get("current", ""),
                "values": [
                    {
                        "name": value.attrib.get("name", ""),
                        "desc": value.attrib.get("desc", ""),
                        "input": value.attrib.get("input", ""),
                        "valid": parse_bool_attr(value.attrib.get("valid")),
                    }
                    for value in assumption.findall("value")
                ],
            }
        )
    return out


def normalize_warnings(root: ET.Element) -> list[dict[str, Any]]:
    warnings_node = root.find("warnings")
    if warnings_node is None:
        return []

    warnings: list[dict[str, Any]] = []
    for warning in warnings_node:
        entry: dict[str, Any] = {"kind": warning.tag, **warning.attrib}
        if warning.tag == "reinterpret":
            entry["alternatives"] = [collapse_whitespace(item.text) for item in warning.findall("alternative") if collapse_whitespace(item.text)]
        warnings.append(entry)
    return warnings


def normalize_sources(root: ET.Element) -> list[dict[str, str]]:
    sources_node = root.find("sources")
    if sources_node is None:
        return []
    return [{"text": source.attrib.get("text", ""), "url": source.attrib.get("url", "")} for source in sources_node.findall("source")]


def normalize_didyoumeans(root: ET.Element) -> list[dict[str, str]]:
    didyoumeans_node = root.find("didyoumeans")
    if didyoumeans_node is None:
        return []
    out: list[dict[str, str]] = []
    for suggestion in didyoumeans_node.findall("didyoumean"):
        out.append(
            {
                "value": collapse_whitespace(suggestion.text),
                "score": suggestion.attrib.get("score", ""),
                "level": suggestion.attrib.get("level", ""),
            }
        )
    return out


def normalize_tips(root: ET.Element) -> list[str]:
    tips_node = root.find("tips")
    if tips_node is None:
        return []
    tips: list[str] = []
    for tip in tips_node.findall("tip"):
        text = tip.attrib.get("text", "")
        if text:
            tips.append(text)
    return tips


def normalize_query_result_xml(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    if root.tag != "queryresult":
        raise RuntimeError(f"Unexpected root element: {root.tag}")

    error = parse_bool_attr(root.attrib.get("error")) or False
    error_node = root.find("error")
    error_payload = None
    if error_node is not None:
        error_payload = {
            "code": element_plain_text(error_node.find("code")),
            "message": element_plain_text(error_node.find("msg")),
        }

    result = {
        "success": parse_bool_attr(root.attrib.get("success")) or False,
        "error": error,
        "numpods": int(root.attrib.get("numpods", "0") or 0),
        "datatypes": [item for item in root.attrib.get("datatypes", "").split(",") if item],
        "timedout": [item for item in root.attrib.get("timedout", "").split(",") if item],
        "timedoutpods": [item for item in root.attrib.get("timedoutpods", "").split(",") if item],
        "timing": root.attrib.get("timing", ""),
        "parsetiming": root.attrib.get("parsetiming", ""),
        "parsetimedout": parse_bool_attr(root.attrib.get("parsetimedout")),
        "recalculate": root.attrib.get("recalculate", ""),
        "pods": [normalize_pod(pod) for pod in root.findall("pod")],
        "assumptions": normalize_assumptions(root),
        "warnings": normalize_warnings(root),
        "sources": normalize_sources(root),
        "didyoumeans": normalize_didyoumeans(root),
        "tips": normalize_tips(root),
        "languagemsg": root.find("languagemsg").attrib if root.find("languagemsg") is not None else None,
        "futuretopic": root.find("futuretopic").attrib if root.find("futuretopic") is not None else None,
        "examplepage": root.find("examplepage").attrib if root.find("examplepage") is not None else None,
        "generalization": root.find("generalization").attrib if root.find("generalization") is not None else None,
        "error_info": error_payload,
    }
    return result


def normalize_validate_result_xml(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text)
    if root.tag != "validatequeryresult":
        raise RuntimeError(f"Unexpected root element: {root.tag}")
    error = parse_bool_attr(root.attrib.get("error")) or False
    error_node = root.find("error")
    error_payload = None
    if error_node is not None:
        error_payload = {
            "code": element_plain_text(error_node.find("code")),
            "message": element_plain_text(error_node.find("msg")),
        }
    return {
        "success": parse_bool_attr(root.attrib.get("success")) or False,
        "error": error,
        "timing": root.attrib.get("timing", ""),
        "parsetiming": root.attrib.get("parsetiming", ""),
        "warnings": normalize_warnings(root),
        "assumptions": normalize_assumptions(root),
        "error_info": error_payload,
    }


def build_answer_json(answer: str, query: str) -> dict[str, Any]:
    return {"query": query, "answer": answer}


def summarize_assumption(assumption: dict[str, Any]) -> str:
    assumption_type = str(assumption.get("type") or "assumption")
    word = str(assumption.get("word") or "").strip()
    values = assumption.get("values") or []
    current = values[0].get("desc") if values else ""
    alternatives = [str(value.get("desc") or value.get("name") or "").strip() for value in values[1:] if str(value.get("desc") or value.get("name") or "").strip()]

    prefix = f"{assumption_type}"
    if word:
        prefix += f' for "{word}"'
    if current and alternatives:
        return f"{prefix}: current {current}; alternatives {', '.join(alternatives)}"
    if current:
        return f"{prefix}: current {current}"

    template = str(assumption.get("template") or "").strip()
    if template and "${" not in template:
        return template
    return prefix


def extract_primary_plaintext(result: dict[str, Any]) -> str:
    def pod_text(pod: dict[str, Any]) -> str:
        for subpod in pod.get("subpods", []):
            for key in ("plaintext", "moutput", "minput"):
                text = str(subpod.get(key, "")).strip()
                if text:
                    return text
        return ""

    pods = result.get("pods", [])
    for pod in pods:
        if pod.get("primary"):
            text = pod_text(pod)
            if text:
                return text
    for pod in pods:
        if pod.get("id") == "Result":
            text = pod_text(pod)
            if text:
                return text
    for pod in pods:
        text = pod_text(pod)
        if text:
            return text
    return ""


def format_verify_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    if result.get("error") and result.get("error_info"):
        error_info = result["error_info"]
        lines.append(f"API error: {error_info.get('message', 'Unknown error')}")
        return "\n".join(lines)

    primary = extract_primary_plaintext(result)
    if primary:
        lines.append(f"Primary result: {primary}")
    else:
        lines.append("Primary result: none")

    datatypes = result.get("datatypes") or []
    if datatypes:
        lines.append("Datatypes: " + ", ".join(datatypes))

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            details = warning.get("text") or warning.get("new") or json.dumps(warning, ensure_ascii=True)
            if warning.get("kind") == "reinterpret" and warning.get("new"):
                details = f"{details} {warning.get('new')}".strip()
                alternatives = warning.get("alternatives") or []
                if alternatives:
                    details = details + " | alternatives: " + ", ".join(str(item) for item in alternatives)
            lines.append(f"- {warning.get('kind', 'warning')}: {details}")

    assumptions = result.get("assumptions") or []
    if assumptions:
        lines.append("Assumptions:")
        for assumption in assumptions:
            lines.append(f"- {summarize_assumption(assumption)}")
            for value in assumption.get("values", []):
                desc = value.get("desc") or value.get("name") or "value"
                token = value.get("input") or ""
                lines.append(f"  {desc}: {token}")

    pods = result.get("pods") or []
    if pods:
        lines.append("Pods:")
        for pod in pods:
            pod_header = pod.get("title") or pod.get("id") or "pod"
            lines.append(f"- {pod_header}")
            for subpod in pod.get("subpods", []):
                pieces = [subpod.get("plaintext"), subpod.get("moutput"), subpod.get("minput")]
                text = next((piece for piece in pieces if piece), "")
                if text:
                    lines.append(f"  {text}")

    tips = result.get("tips") or []
    if tips:
        lines.append("Tips:")
        for tip in tips:
            lines.append(f"- {tip}")

    didyoumeans = result.get("didyoumeans") or []
    if didyoumeans:
        lines.append("Did you mean:")
        for item in didyoumeans:
            lines.append(f"- {item.get('value', '')}")

    return "\n".join(lines)


def format_validate_text(result: dict[str, Any]) -> str:
    lines = [f"Understood: {bool_to_text(bool(result.get('success')))}"]
    warnings = result.get("warnings") or []
    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            details = warning.get("text") or warning.get("new") or json.dumps(warning, ensure_ascii=True)
            lines.append(f"- {warning.get('kind', 'warning')}: {details}")
    assumptions = result.get("assumptions") or []
    if assumptions:
        lines.append("Assumptions:")
        for assumption in assumptions:
            lines.append(f"- {summarize_assumption(assumption)}")
    if result.get("error") and result.get("error_info"):
        lines.append(f"API error: {result['error_info'].get('message', 'Unknown error')}")
    return "\n".join(lines)


def build_client(args: argparse.Namespace) -> WolframAlphaClient:
    appid = (args.appid or os.getenv("WOLFRAM_ALPHA_APPID", "")).strip()
    timeout_raw = str(args.timeout) if args.timeout is not None else os.getenv("WOLFRAM_ALPHA_TIMEOUT", str(DEFAULT_TIMEOUT))
    result_url = args.result_url or os.getenv("WOLFRAM_ALPHA_RESULT_URL", DEFAULT_RESULT_URL)
    query_url = args.query_url or os.getenv("WOLFRAM_ALPHA_QUERY_URL", DEFAULT_QUERY_URL)
    validate_url = args.validate_url or os.getenv("WOLFRAM_ALPHA_VALIDATE_URL", DEFAULT_VALIDATE_URL)

    if not appid:
        raise RuntimeError("Missing WOLFRAM_ALPHA_APPID. Set the environment variable or pass --appid.")

    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid WOLFRAM_ALPHA_TIMEOUT value: {timeout_raw}") from exc

    return WolframAlphaClient(
        appid=appid,
        timeout=timeout,
        result_url=result_url,
        query_url=query_url,
        validate_url=validate_url,
    )


def append_repeated(params: list[tuple[str, str]], key: str, values: Iterable[str]) -> None:
    for value in values:
        params.append((key, value))


def verify_profile_options(profile: str) -> dict[str, Any]:
    return VERIFY_PROFILES.get(profile, VERIFY_PROFILES["full"])


def build_verify_params(args: argparse.Namespace, profile_override: str | None = None) -> list[tuple[str, str]]:
    profile_name = profile_override or args.profile
    profile = verify_profile_options(profile_name)
    params: list[tuple[str, str]] = [
        ("input", args.query),
        ("format", args.format_types or str(profile.get("format_types") or DEFAULT_QUERY_FORMATS)),
    ]

    if args.output == "json":
        params.append(("output", "json"))

    if args.reinterpret:
        params.append(("reinterpret", "true"))
    if args.translation:
        params.append(("translation", "true"))
    if args.ignorecase:
        params.append(("ignorecase", "true"))
    if args.units:
        params.append(("units", args.units))
    if args.location:
        params.append(("location", args.location))
    if args.scantimeout is not None:
        params.append(("scantimeout", str(args.scantimeout)))
    if args.podtimeout is not None:
        params.append(("podtimeout", str(args.podtimeout)))
    if args.formattimeout is not None:
        params.append(("formattimeout", str(args.formattimeout)))
    if args.parsetimeout is not None:
        params.append(("parsetimeout", str(args.parsetimeout)))
    if args.totaltimeout is not None:
        params.append(("totaltimeout", str(args.totaltimeout)))

    append_repeated(params, "includepodid", profile.get("includepodid", []))
    append_repeated(params, "podtitle", profile.get("podtitle", []))
    append_repeated(params, "assumption", repeated_values(args.assumption))
    append_repeated(params, "podstate", repeated_values(args.podstate))
    append_repeated(params, "includepodid", repeated_values(args.includepodid))
    append_repeated(params, "podtitle", repeated_values(args.podtitle))
    append_repeated(params, "scanner", repeated_values(args.scanner))
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query Wolfram|Alpha for math knowledge and verification")
    parser.add_argument("--appid", help="Override Wolfram|Alpha AppID")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds")
    parser.add_argument("--result-url", help="Override the short answer endpoint")
    parser.add_argument("--query-url", help="Override the full results endpoint")
    parser.add_argument("--validate-url", help="Override the validatequery endpoint")

    subparsers = parser.add_subparsers(dest="command", required=True)

    answer_parser = subparsers.add_parser("answer", help="Get a concise Wolfram|Alpha answer")
    answer_parser.add_argument("query", help="Free-form Wolfram|Alpha query")
    answer_parser.add_argument("--reinterpret", action=argparse.BooleanOptionalAction, default=None)
    answer_parser.add_argument("--format", choices=("text", "json"), default="text")

    verify_parser = subparsers.add_parser("verify", help="Inspect structured Wolfram|Alpha output")
    verify_parser.add_argument("query", help="Free-form Wolfram|Alpha query")
    verify_parser.add_argument("--assumption", action="append", default=[])
    verify_parser.add_argument("--podstate", action="append", default=[])
    verify_parser.add_argument("--includepodid", action="append", default=[])
    verify_parser.add_argument("--podtitle", action="append", default=[])
    verify_parser.add_argument("--scanner", action="append", default=[])
    verify_parser.add_argument("--reinterpret", action="store_true")
    verify_parser.add_argument("--translation", action="store_true")
    verify_parser.add_argument("--ignorecase", action="store_true")
    verify_parser.add_argument("--units", choices=("metric", "nonmetric"))
    verify_parser.add_argument("--location")
    verify_parser.add_argument("--profile", choices=tuple(VERIFY_PROFILES.keys()), default="full")
    verify_parser.add_argument("--format", choices=("text", "json"), default="text")
    verify_parser.add_argument("--output", choices=("xml", "json"), default="xml")
    verify_parser.add_argument("--format-types")
    verify_parser.add_argument("--scantimeout", type=float)
    verify_parser.add_argument("--podtimeout", type=float)
    verify_parser.add_argument("--formattimeout", type=float)
    verify_parser.add_argument("--parsetimeout", type=float)
    verify_parser.add_argument("--totaltimeout", type=float)

    validate_parser = subparsers.add_parser("validate", help="Check whether a query is understood")
    validate_parser.add_argument("query", help="Free-form Wolfram|Alpha query")
    validate_parser.add_argument("--format", choices=("text", "json"), default="text")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        client = build_client(args)

        if args.command == "answer":
            answer = client.short_answer(args.query, reinterpret=args.reinterpret)
            if args.format == "json":
                print(json.dumps(build_answer_json(answer, args.query), indent=2, ensure_ascii=True))
            else:
                print(answer)
            return 0

        if args.command == "verify":
            params = build_verify_params(args)
            raw = client.query(params)
            if args.output == "json":
                payload = json.loads(raw)
                if args.format == "json":
                    print(json.dumps(payload, indent=2, ensure_ascii=True))
                else:
                    print(json.dumps(payload, ensure_ascii=True))
                return 0

            normalized = normalize_query_result_xml(raw)
            if args.profile != "full" and not normalized.get("pods"):
                raw = client.query(build_verify_params(args, profile_override="full"))
                normalized = normalize_query_result_xml(raw)
            if args.format == "json":
                print(json.dumps(normalized, indent=2, ensure_ascii=True))
            else:
                print(format_verify_text(normalized))
            return 0

        if args.command == "validate":
            raw = client.validate(args.query)
            normalized = normalize_validate_result_xml(raw)
            if args.format == "json":
                print(json.dumps(normalized, indent=2, ensure_ascii=True))
            else:
                print(format_validate_text(normalized))
            return 0

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())