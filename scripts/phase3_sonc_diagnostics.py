#!/usr/bin/env python3
"""Run targeted SONC diagnostics for failed clean pilot families."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Irene.sonc import SONCRelaxations
from scripts.phase3_benchmarks import BenchmarkCase, create_problem


DIAGNOSTIC_CONFIGS = [
    {"label": "local_tol", "error_bound": None, "use_local_solve": True},
    {"label": "global_tol", "error_bound": None, "use_local_solve": False},
    {"label": "local_1e-10", "error_bound": 1e-10, "use_local_solve": True},
    {"label": "global_1e-10", "error_bound": 1e-10, "use_local_solve": False},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SONC diagnostics on clean pilot failures")
    parser.add_argument(
        "--input",
        default="MeansResearch/results/phase3_runs_clean.jsonl",
        help="Clean JSONL pilot results",
    )
    parser.add_argument(
        "--output",
        default="MeansResearch/results/phase3_sonc_diagnostics.jsonl",
        help="Diagnostic JSONL output",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional markdown summary output",
    )
    parser.add_argument("--item-id", default=None, help="Optional item id filter, e.g. L-C2")
    parser.add_argument("--degree", type=int, default=None, help="Optional degree filter")
    parser.add_argument(
        "--variables",
        default=None,
        help="Optional comma-separated n filter, e.g. 3,4",
    )
    parser.add_argument(
        "--alpha-templates",
        default=None,
        help="Optional comma-separated alpha-template filter, e.g. uniform,boundary,mixed",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_csv_ints(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_csv_strings(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def build_cases(
    rows: list[dict[str, Any]],
    *,
    item_id: str | None,
    degree: int | None,
    variables: set[int] | None,
    alpha_templates: set[str] | None,
) -> list[tuple[BenchmarkCase, float]]:
    cases: list[tuple[BenchmarkCase, float]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("method") != "SONCRelaxations":
            continue
        if row.get("status") != "fail":
            continue
        desc = row.get("polynomial_family_descriptor") or {}
        if item_id and desc.get("item_id") != item_id:
            continue
        if degree is not None and desc.get("d") != degree:
            continue
        if variables is not None and desc.get("n") not in variables:
            continue
        if alpha_templates is not None and desc.get("alpha_template") not in alpha_templates:
            continue
        key = (
            desc.get("d"),
            desc.get("n"),
            desc.get("p"),
            tuple(desc.get("alpha", [])),
            row.get("tolerance_setting"),
        )
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            (
                BenchmarkCase(
                    item_id=desc.get("item_id", "L-C2"),
                    d=desc["d"],
                    n=desc["n"],
                    p=desc["p"],
                    alpha=tuple(desc["alpha"]),
                    alpha_template=desc.get("alpha_template", "unknown"),
                ),
                float(row.get("tolerance_setting")),
            )
        )
    return cases


def run_case(case: BenchmarkCase, base_tolerance: float, cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        problem, objective = create_problem(case)
        error_bound = cfg["error_bound"] if cfg["error_bound"] is not None else base_tolerance
        sonc = SONCRelaxations(
            problem,
            error_bound=error_bound,
            verbosity=0,
            use_local_solve=cfg["use_local_solve"],
        )
        value = sonc.solve(verbosity=0)
        return {
            "status": "success",
            "runtime_sec": float(time.perf_counter() - start),
            "objective": str(objective),
            "sonc_lower_bound": value,
            "solution_available": sonc.solution is not None,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "runtime_sec": float(time.perf_counter() - start),
            "objective": None,
            "sonc_lower_bound": None,
            "solution_available": False,
            "notes": str(exc),
        }


def render_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> str:
    config_status: dict[str, Counter] = defaultdict(Counter)
    template_status: dict[str, Counter] = defaultdict(Counter)
    template_config_status: dict[tuple[str, str], Counter] = defaultdict(Counter)
    case_counts: Counter = Counter()

    for record in records:
        label = record["config"]["label"]
        status = record["status"]
        template = record["case"]["alpha_template"]
        config_status[label][status] += 1
        template_status[template][status] += 1
        template_config_status[(template, label)][status] += 1
        case_counts[template] += 1

    unique_case_counts = {template: count // len(DIAGNOSTIC_CONFIGS) for template, count in case_counts.items()}

    lines: list[str] = []
    lines.append("# SONC Diagnostics Summary")
    lines.append("")
    lines.append(f"Input: {args.input}")
    lines.append(f"Output: {args.output}")
    filters: list[str] = []
    if args.item_id:
        filters.append(f"item={args.item_id}")
    if args.degree is not None:
        filters.append(f"d={args.degree}")
    if args.variables:
        filters.append(f"n in [{args.variables}]")
    if args.alpha_templates:
        filters.append(f"templates in [{args.alpha_templates}]")
    lines.append(f"Filters: {', '.join(filters) if filters else 'none'}")
    lines.append(f"Unique failed SONC cases rerun: {sum(unique_case_counts.values())}")
    lines.append("")

    lines.append("## Case Count by Template")
    lines.append("")
    lines.append("| alpha_template | cases |")
    lines.append("|---|---:|")
    for template, count in sorted(unique_case_counts.items()):
        lines.append(f"| {template} | {count} |")
    lines.append("")

    lines.append("## Status by Diagnostic Config")
    lines.append("")
    lines.append("| config | success | fail |")
    lines.append("|---|---:|---:|")
    for label, counts in sorted(config_status.items()):
        lines.append(f"| {label} | {counts.get('success', 0)} | {counts.get('fail', 0)} |")
    lines.append("")

    lines.append("## Status by Template")
    lines.append("")
    lines.append("| alpha_template | success | fail |")
    lines.append("|---|---:|---:|")
    for template, counts in sorted(template_status.items()):
        lines.append(f"| {template} | {counts.get('success', 0)} | {counts.get('fail', 0)} |")
    lines.append("")

    lines.append("## Template x Config")
    lines.append("")
    lines.append("| alpha_template | config | success | fail |")
    lines.append("|---|---|---:|---:|")
    for (template, label), counts in sorted(template_config_status.items()):
        lines.append(f"| {template} | {label} | {counts.get('success', 0)} | {counts.get('fail', 0)} |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = REPO_ROOT / args.input
    output_path = REPO_ROOT / args.output
    rows = load_rows(input_path)
    cases = build_cases(
        rows,
        item_id=args.item_id,
        degree=args.degree,
        variables=parse_csv_ints(args.variables),
        alpha_templates=parse_csv_strings(args.alpha_templates),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for case, base_tolerance in cases:
            for cfg in DIAGNOSTIC_CONFIGS:
                result = run_case(case, base_tolerance, cfg)
                record = {
                    "case": {
                        "d": case.d,
                        "n": case.n,
                        "p": case.p,
                        "alpha": list(case.alpha),
                        "alpha_template": case.alpha_template,
                        "base_tolerance": base_tolerance,
                    },
                    "config": {
                        "label": cfg["label"],
                        "error_bound": cfg["error_bound"] if cfg["error_bound"] is not None else base_tolerance,
                        "use_local_solve": cfg["use_local_solve"],
                    },
                }
                record.update(result)
                written_records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    if args.output_md:
        summary_path = REPO_ROOT / args.output_md
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_summary(written_records, args) + "\n", encoding="utf-8")

    print(f"Wrote diagnostics to {output_path}")


if __name__ == "__main__":
    main()
