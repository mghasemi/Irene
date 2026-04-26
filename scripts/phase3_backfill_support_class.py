#!/usr/bin/env python3
"""Backfill support_class in historical phase3 JSONL benchmark logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase3_benchmarks import BenchmarkCase, support_class_from_case


DEFAULT_INPUTS = (
    "MeansResearch/results/phase3_runs.jsonl,"
    "MeansResearch/results/phase3_runs_clean.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d5.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch1.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch2.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch3.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch4_boundary.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch4_boundary_probe.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch5_mixed.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch6_uniform_p6.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch7_boundary_p6.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_batch8_mixed_p6.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_smoke.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass1.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_uniform.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_boundary.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_mixed.jsonl,"
    "MeansResearch/results/phase3_runs_clean_d6_lc2.jsonl,"
    "MeansResearch/results/phase3_runs_clean_cx1_d3_n5_probe.jsonl,"
    "MeansResearch/results/phase3_runs_clean_cx1_d4_n5.jsonl,"
    "MeansResearch/results/phase3_runs_clean_cx1_d4_n6.jsonl,"
    "MeansResearch/results/phase3_runs_clean_cx1_batch1_d5_n5_boundary_mixed.jsonl,"
    "MeansResearch/results/phase3_runs_clean_cx2_transform_probe.jsonl,"
    "MeansResearch/results/phase3_runs_clean_cx2_transform_probe_batch2.jsonl,"
    "MeansResearch/results/phase3_runs_postfix_check.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill support_class in phase3 JSONL logs")
    parser.add_argument(
        "--inputs",
        default=DEFAULT_INPUTS,
        help="Comma-separated JSONL paths to rewrite in place",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files",
    )
    return parser.parse_args()


def recompute_support_class(desc: dict[str, Any]) -> str:
    case = BenchmarkCase(
        item_id=str(desc.get("item_id") or "unknown"),
        d=int(desc["d"]),
        n=int(desc["n"]),
        p=int(desc["p"]),
        alpha=tuple(int(x) for x in desc.get("alpha", [])),
        alpha_template=str(desc.get("alpha_template") or "unknown"),
    )
    return support_class_from_case(case)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def backfill_file(path: Path, dry_run: bool) -> tuple[int, int]:
    rows = load_jsonl(path)
    touched = 0
    skipped = 0

    for row in rows:
        desc = row.get("polynomial_family_descriptor")
        if not isinstance(desc, dict):
            skipped += 1
            continue
        try:
            new_support = recompute_support_class(desc)
        except Exception:
            skipped += 1
            continue
        if desc.get("support_class") != new_support:
            desc["support_class"] = new_support
            touched += 1

    if touched > 0 and not dry_run:
        dump_jsonl(path, rows)
    return touched, skipped


def main() -> None:
    args = parse_args()
    paths = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]

    total_touched = 0
    total_skipped = 0
    processed = 0

    for path in paths:
        if not path.exists():
            print(f"MISSING {path}")
            continue
        touched, skipped = backfill_file(path, dry_run=args.dry_run)
        processed += 1
        total_touched += touched
        total_skipped += skipped
        action = "WOULD_UPDATE" if args.dry_run else "UPDATED"
        print(f"{action} {path} touched={touched} skipped={skipped}")

    mode = "dry-run" if args.dry_run else "write"
    print(
        f"DONE mode={mode} processed={processed} total_touched={total_touched} total_skipped={total_skipped}"
    )


if __name__ == "__main__":
    main()