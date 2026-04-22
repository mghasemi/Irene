#!/usr/bin/env python3
"""Summarize Phase 3 benchmark JSONL into compact tables for ledger/manuscript use."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary report from phase3 JSONL records")
    parser.add_argument(
        "--input",
        default="MeansResearch/results/phase3_runs.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output-md",
        default="MeansResearch/results/phase3_pilot_summary.md",
        help="Output markdown report path",
    )
    parser.add_argument("--degree", type=int, default=4, help="Filter degree d")
    parser.add_argument("--variables", default="3,4", help="Comma-separated n values")
    parser.add_argument("--tolerances", default="1e-6,1e-8", help="Comma-separated tolerances")
    return parser.parse_args()


def parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def status_table(rows: list[dict[str, Any]]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        out[row.get("method", "unknown")][row.get("status", "unknown")] += 1
    return out


def pair_sonc_gp(rows: list[dict[str, Any]]) -> list[tuple[str, float, float, float]]:
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        m = row.get("method")
        if m in {"SONCRelaxations", "GPRelaxations"}:
            paired[row.get("id", "")][m] = row

    comparisons: list[tuple[str, float, float, float]] = []
    for rid, pair in paired.items():
        sonc = pair.get("SONCRelaxations")
        gp = pair.get("GPRelaxations")
        if not sonc or not gp:
            continue
        if sonc.get("status") != "success" or gp.get("status") != "success":
            continue
        sv = (sonc.get("bounds") or {}).get("sonc_lower_bound")
        gv = (gp.get("bounds") or {}).get("gp_lower_bound")
        if isinstance(sv, (int, float)) and isinstance(gv, (int, float)):
            comparisons.append((rid, float(sv), float(gv), float(gv - sv)))
    return comparisons


def failure_breakdown(rows: list[dict[str, Any]]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row.get("status") in {"success", "inconclusive"}:
            continue
        method = row.get("method", "unknown")
        alpha = tuple((row.get("polynomial_family_descriptor") or {}).get("alpha", []))
        out[method][str(alpha)] += 1
    return out


def render_report(rows: list[dict[str, Any]], degree: int, n_values: list[int], tolerances: list[float]) -> str:
    total = len(rows)
    by_method = Counter(row.get("method", "unknown") for row in rows)
    status = status_table(rows)
    comparisons = pair_sonc_gp(rows)
    failures = failure_breakdown(rows)

    lines: list[str] = []
    lines.append("# Phase 3 Pilot Summary")
    lines.append("")
    lines.append(f"Filter: d={degree}, n in {n_values}, tolerances={tolerances}")
    lines.append(f"Total records: {total}")
    lines.append("")

    lines.append("## Method Counts")
    lines.append("")
    lines.append("| Method | Records |")
    lines.append("|---|---:|")
    for method, count in sorted(by_method.items()):
        lines.append(f"| {method} | {count} |")
    lines.append("")

    lines.append("## Status by Method")
    lines.append("")
    lines.append("| Method | success | inconclusive | fail | infeasible |")
    lines.append("|---|---:|---:|---:|---:|")
    for method in sorted(status.keys()):
        c = status[method]
        lines.append(
            f"| {method} | {c.get('success', 0)} | {c.get('inconclusive', 0)} | {c.get('fail', 0)} | {c.get('infeasible', 0)} |"
        )
    lines.append("")

    lines.append("## SONC vs GP (paired successful cases)")
    lines.append("")
    lines.append(f"Paired comparisons: {len(comparisons)}")
    if comparisons:
        diffs = [d for _, _, _, d in comparisons]
        lines.append(f"Gap range (gp - sonc): min={min(diffs):.3e}, max={max(diffs):.3e}")
        lines.append("")
        lines.append("| id | sonc | gp | gp-sonc |")
        lines.append("|---|---:|---:|---:|")
        for rid, sv, gv, diff in comparisons[:10]:
            lines.append(f"| {rid} | {sv:.12g} | {gv:.12g} | {diff:.3e} |")
    lines.append("")

    lines.append("## Failure Concentration by Alpha")
    lines.append("")
    for method in sorted(failures.keys()):
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| alpha | fail_count |")
        lines.append("|---|---:|")
        for alpha, count in failures[method].most_common(10):
            lines.append(f"| {alpha} | {count} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output_md)

    rows = load_rows(input_path)
    n_values = parse_csv_ints(args.variables)
    tolerances = parse_csv_floats(args.tolerances)

    filtered = [
        row
        for row in rows
        if (row.get("polynomial_family_descriptor") or {}).get("d") == args.degree
        and (row.get("polynomial_family_descriptor") or {}).get("n") in n_values
        and row.get("tolerance_setting") in tolerances
        and row.get("method") in {"SDPRelaxations", "SONCRelaxations", "GPRelaxations"}
    ]

    report = render_report(filtered, args.degree, n_values, tolerances)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
