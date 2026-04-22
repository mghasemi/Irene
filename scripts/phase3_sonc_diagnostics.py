#!/usr/bin/env python3
"""Run targeted SONC diagnostics for failed clean pilot families."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_cases(rows: list[dict[str, Any]]) -> list[tuple[BenchmarkCase, float]]:
    cases: list[tuple[BenchmarkCase, float]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("method") != "SONCRelaxations":
            continue
        if row.get("status") != "fail":
            continue
        desc = row.get("polynomial_family_descriptor") or {}
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


def main() -> None:
    args = parse_args()
    input_path = REPO_ROOT / args.input
    output_path = REPO_ROOT / args.output
    rows = load_rows(input_path)
    cases = build_cases(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"Wrote diagnostics to {output_path}")


if __name__ == "__main__":
    main()
