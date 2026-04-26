#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Irene.geometric import GPRelaxations
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations


DIAGNOSTIC_CONFIGS = [
    {"label": "local_tol", "error_bound": None, "use_local_solve": True},
    {"label": "global_tol", "error_bound": None, "use_local_solve": False},
    {"label": "local_1e-10", "error_bound": 1e-10, "use_local_solve": True},
    {"label": "global_1e-10", "error_bound": 1e-10, "use_local_solve": False},
]

FIELDNAMES = [
    "experiment_id",
    "coefficient_c",
    "transform_t",
    "transform_condition_number",
    "sonc_status_base",
    "gp_status_base",
    "sonc_status_local_tol",
    "sonc_status_global_tol",
    "sonc_status_local_1e-10",
    "sonc_status_global_1e-10",
    "final_class",
    "runtime_seconds",
]


def coeff_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_robinson_hat(sa: SemigroupAlgebra, coeff: float, t: float) -> tuple[Any, float]:
    x, y, z, w = sa["x"], sa["y"], sa["z"], sa["w"]

    transform = np.array(
        [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, -1.0],
            [1.0, 1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    matrix = (1.0 - t) * np.eye(4) + t * transform
    cond = float(np.linalg.cond(matrix))

    xt = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2] * z + matrix[0, 3] * w
    yt = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2] * z + matrix[1, 3] * w
    zt = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2] * z + matrix[2, 3] * w
    wt = matrix[3, 0] * x + matrix[3, 1] * y + matrix[3, 2] * z + matrix[3, 3] * w

    poly = (
        xt**2 * (xt - wt) ** 2
        + yt**2 * (yt - wt) ** 2
        + zt**2 * (zt - wt) ** 2
        + coeff * xt * yt * zt * (xt + yt + zt - 2 * wt)
    )
    return poly, cond


def run_sonc(problem: OptimizationProblem, *, error_bound: float, use_local_solve: bool) -> dict[str, Any]:
    sonc = SONCRelaxations(problem, error_bound=error_bound, verbosity=0, use_local_solve=use_local_solve)
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            value = float(sonc.solve(verbosity=0))
        return {
            "status": "success",
            "lower_bound": value,
            "runtime_sec": float(time.perf_counter() - start),
            "error": None,
            "solution_available": sonc.solution is not None,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "lower_bound": None,
            "runtime_sec": float(time.perf_counter() - start),
            "error": str(exc),
            "solution_available": False,
        }


def run_gp(problem: OptimizationProblem) -> dict[str, Any]:
    gp = GPRelaxations(problem, auto_transform=True, verbosity=0)
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            value = float(gp.solve())
        return {
            "status": "success",
            "lower_bound": value,
            "runtime_sec": float(time.perf_counter() - start),
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "fail",
            "lower_bound": None,
            "runtime_sec": float(time.perf_counter() - start),
            "error": str(exc),
        }


def adjacency_crossing(rows: list[dict[str, Any]], c_values: list[float], t_values: list[float]) -> int:
    index = {(row["coefficient_c"], row["transform_t"]): row["final_class"] for row in rows}
    crossings = 0
    for i, c in enumerate(c_values):
        for j, t in enumerate(t_values):
            here = index[(c, t)]
            if here not in {"robust_success", "robust_fail"}:
                continue
            if i + 1 < len(c_values):
                other = index[(c_values[i + 1], t)]
                if {here, other} == {"robust_success", "robust_fail"}:
                    crossings += 1
            if j + 1 < len(t_values):
                other = index[(c, t_values[j + 1])]
                if {here, other} == {"robust_success", "robust_fail"}:
                    crossings += 1
    return crossings


def write_summary(path: Path, rows: list[dict[str, Any]], *, c_values: list[float], t_values: list[float], base_tolerance: float, cond_cap: float) -> None:
    total = len(rows)
    class_counts: dict[str, int] = {}
    for row in rows:
        class_counts[row["final_class"]] = class_counts.get(row["final_class"], 0) + 1

    center_rows = [row for row in rows if abs(row["coefficient_c"] - 2.0) < 1e-12 and abs(row["transform_t"] - 1.0) < 1e-12]
    non_center_successes = [row for row in rows if row["final_class"] == "robust_success" and not (abs(row["coefficient_c"] - 2.0) < 1e-12 and abs(row["transform_t"] - 1.0) < 1e-12)]
    crossings = adjacency_crossing(rows, c_values, t_values)
    classified = class_counts.get("robust_success", 0) + class_counts.get("robust_fail", 0)

    h1 = len(non_center_successes) > 0
    h2 = class_counts.get("robust_success", 0) > 0 and class_counts.get("robust_fail", 0) > 0 and crossings > 0
    h3 = (classified / total) >= 0.9 if total else False

    lines = [
        "# OP3 Local Neighborhood Summary",
        "",
        f"Base tolerance: {base_tolerance:.0e}",
        f"Condition cap: {cond_cap:.3g}",
        f"Total grid points: {total}",
        "",
        "## Class Counts",
        "",
        "| final_class | count |",
        "|---|---:|",
    ]
    for label in ["robust_success", "robust_fail", "unstable", "skipped_conditioned"]:
        lines.append(f"| {label} | {class_counts.get(label, 0)} |")

    lines.extend(
        [
            "",
            "## Gate Evaluation (H1-H3)",
            "",
            f"- H1 (local-neighborhood existence): {'PASS' if h1 else 'FAIL'}",
            f"  - Non-center robust_success points: {len(non_center_successes)}",
            f"- H2 (boundary detectability): {'PASS' if h2 else 'FAIL'}",
            f"  - Adjacency crossings between robust_success and robust_fail: {crossings}",
            f"- H3 (diagnostic stability): {'PASS' if h3 else 'FAIL'}",
            f"  - Stable classified share: {classified}/{total} = {(classified / total):.3f}",
            "",
            "## Center Point",
            "",
        ]
    )
    if center_rows:
        center = center_rows[0]
        lines.append(f"- final_class: {center['final_class']}")
        lines.append(f"- sonc_status_base: {center['sonc_status_base']}")
        lines.append(f"- gp_status_base: {center['gp_status_base']}")
    else:
        lines.append("- Center point missing from sweep output.")

    lines.extend(
        [
            "",
            "## Non-center robust_success Points",
            "",
        ]
    )
    if non_center_successes:
        lines.append("| coefficient_c | transform_t | gp_status_base | runtime_seconds |")
        lines.append("|---:|---:|---|---:|")
        for row in non_center_successes:
            lines.append(
                f"| {row['coefficient_c']:.4f} | {row['transform_t']:.2f} | {row['gp_status_base']} | {row['runtime_seconds']:.4f} |"
            )
    else:
        lines.append("- None")

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OP3 Robinson-hat local neighborhood sweep")
    parser.add_argument(
        "--output-jsonl",
        default="MeansResearch/results/op3_local_neighborhood_sweep.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--output-csv",
        default="MeansResearch/results/op3_local_neighborhood_table.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-md",
        default="MeansResearch/results/op3_local_neighborhood_summary.md",
        help="Output markdown summary",
    )
    parser.add_argument(
        "--coeff-grid",
        default="1.9925,1.9950,1.9975,2.0000,2.0025,2.0050,2.0075",
        help="Comma-separated Robinson-hat coefficient grid",
    )
    parser.add_argument(
        "--transform-grid",
        default="0.90,0.94,0.97,1.00,1.03,1.06,1.10",
        help="Comma-separated transform-path grid",
    )
    parser.add_argument(
        "--base-tolerance",
        type=float,
        default=1e-8,
        help="Base tolerance used in primary SONC run and local/global_tol diagnostics",
    )
    parser.add_argument(
        "--condition-cap",
        type=float,
        default=1e8,
        help="Skip points whose interpolated transform matrix condition number exceeds this cap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    c_values = parse_csv_floats(args.coeff_grid)
    t_values = parse_csv_floats(args.transform_grid)
    sa = SemigroupAlgebra(CommutativeSemigroup(["x", "y", "z", "w"]))

    rows: list[dict[str, Any]] = []
    for c in c_values:
        for t in t_values:
            experiment_id = f"Rhat-local-c{coeff_tag(c)}-t{coeff_tag(t)}"
            poly, cond = build_robinson_hat(sa, c, t)
            row: dict[str, Any] = {
                "experiment_id": experiment_id,
                "coefficient_c": c,
                "transform_t": t,
                "transform_condition_number": cond,
                "sonc_status_base": "skipped",
                "gp_status_base": "skipped",
                "sonc_status_local_tol": "skipped",
                "sonc_status_global_tol": "skipped",
                "sonc_status_local_1e-10": "skipped",
                "sonc_status_global_1e-10": "skipped",
                "final_class": "skipped_conditioned",
                "runtime_seconds": 0.0,
            }

            if cond > args.condition_cap:
                rows.append(row)
                continue

            problem = OptimizationProblem(sa)
            problem.set_objective(poly)
            started = time.perf_counter()
            sonc_base = run_sonc(problem, error_bound=args.base_tolerance, use_local_solve=True)
            gp_base = run_gp(problem)
            row["sonc_status_base"] = sonc_base["status"]
            row["gp_status_base"] = gp_base["status"]

            diag_statuses: dict[str, str] = {}
            if sonc_base["status"] == "success":
                for cfg in DIAGNOSTIC_CONFIGS:
                    diag_statuses[cfg["label"]] = "success"
                row["final_class"] = "robust_success"
            else:
                for cfg in DIAGNOSTIC_CONFIGS:
                    error_bound = cfg["error_bound"] if cfg["error_bound"] is not None else args.base_tolerance
                    diag = run_sonc(problem, error_bound=error_bound, use_local_solve=cfg["use_local_solve"])
                    diag_statuses[cfg["label"]] = diag["status"]

                if all(status == "fail" for status in diag_statuses.values()):
                    row["final_class"] = "robust_fail"
                elif all(status == "success" for status in diag_statuses.values()):
                    row["final_class"] = "robust_success"
                else:
                    row["final_class"] = "unstable"

            row["sonc_status_local_tol"] = diag_statuses.get("local_tol", row["sonc_status_base"])
            row["sonc_status_global_tol"] = diag_statuses.get("global_tol", row["sonc_status_base"])
            row["sonc_status_local_1e-10"] = diag_statuses.get("local_1e-10", row["sonc_status_base"])
            row["sonc_status_global_1e-10"] = diag_statuses.get("global_1e-10", row["sonc_status_base"])
            row["runtime_seconds"] = float(time.perf_counter() - started)
            rows.append(row)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(output_md, rows, c_values=c_values, t_values=t_values, base_tolerance=args.base_tolerance, cond_cap=args.condition_cap)
    print(f"Wrote {len(rows)} rows to {output_jsonl}")
    print(f"Wrote table to {output_csv}")
    print(f"Wrote summary to {output_md}")


if __name__ == "__main__":
    main()