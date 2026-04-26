#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.sonc import SONCRelaxations
from Irene.geometric import GPRelaxations


@dataclass(frozen=True)
class CX2Case:
    id: str
    label: str
    kind: str
    transform_tag: str
    polynomial: Any
    family_coeff: float


def coeff_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def build_cases(q_tail_coeffs: list[float], rhat_tail_coeffs: list[float]) -> list[CX2Case]:
    sg = CommutativeSemigroup(["x", "y", "z", "w"])
    sa = SemigroupAlgebra(sg)
    x, y, z, w = sa["x"], sa["y"], sa["z"], sa["w"]

    a, b, c, d = x - w, y - w, z - w, x + y + z - w
    cases: list[CX2Case] = []

    for q_coeff in q_tail_coeffs:
        q = x**4 + y**4 + z**4 + w**4 - q_coeff * x * y * z * w
        q_tag = coeff_tag(q_coeff)
        q_label = f"Choi-Lam family (tail={q_coeff:g})"
        cases.append(CX2Case(f"Q-c{q_tag}", q_label, "Q", "pre", q, q_coeff))

    for r_coeff in rhat_tail_coeffs:
        rhat = x**2 * (x - w) ** 2 + y**2 * (y - w) ** 2 + z**2 * (z - w) ** 2 + r_coeff * x * y * z * (x + y + z - 2 * w)
        rhat_transform = (
            a**2 * (a - d) ** 2
            + b**2 * (b - d) ** 2
            + c**2 * (c - d) ** 2
            + r_coeff * a * b * c * (a + b + c - 2 * d)
        )
        r_tag = coeff_tag(r_coeff)
        r_label = f"Robinson-hat family (tail={r_coeff:g})"
        cases.append(CX2Case(f"Rhat-c{r_tag}", r_label, "Rhat", "pre", rhat, r_coeff))
        cases.append(CX2Case(f"Rhat-c{r_tag}-transform", r_label + " transformed", "Rhat", "post", rhat_transform, r_coeff))

    return cases


def structure_diagnostics(sa: SemigroupAlgebra, poly: Any) -> dict[str, Any]:
    p = OptimizationProblem(sa)
    p.set_objective(poly)
    diag: dict[str, Any] = {}

    try:
        delta = p.delta(poly, p.program_degree())
        diag["delta_eq_d_size"] = len(delta.get("=d", []))
        diag["delta_lt_d_size"] = len(delta.get("<d", []))
    except Exception as exc:  # pragma: no cover - diagnostics only
        diag["delta_error"] = str(exc)

    try:
        p.newton()
        diag["newton_vertices"] = len(getattr(p, "vertices", []) or [])
    except Exception as exc:  # pragma: no cover - diagnostics only
        diag["newton_error"] = str(exc)

    return diag


def run_methods(sa: SemigroupAlgebra, poly: Any, tol: float) -> dict[str, Any]:
    out: dict[str, Any] = {}

    sp = OptimizationProblem(sa)
    sp.set_objective(poly)
    sonc = SONCRelaxations(sp, error_bound=tol, verbosity=0, use_local_solve=True)
    t0 = time.perf_counter()
    try:
        val = float(sonc.solve(verbosity=0))
        out["sonc"] = {
            "status": "success",
            "lower_bound": val,
            "runtime_sec": float(time.perf_counter() - t0),
            "error": None,
        }
    except Exception as exc:
        out["sonc"] = {
            "status": "fail",
            "lower_bound": None,
            "runtime_sec": float(time.perf_counter() - t0),
            "error": str(exc),
        }

    gp_p = OptimizationProblem(sa)
    gp_p.set_objective(poly)
    gp = GPRelaxations(gp_p, auto_transform=True, verbosity=0)
    t1 = time.perf_counter()
    try:
        val = float(gp.solve())
        out["gp"] = {
            "status": "success",
            "lower_bound": val,
            "runtime_sec": float(time.perf_counter() - t1),
            "error": None,
        }
    except Exception as exc:
        out["gp"] = {
            "status": "fail",
            "lower_bound": None,
            "runtime_sec": float(time.perf_counter() - t1),
            "error": str(exc),
        }

    return out


def write_summary(rows: list[dict[str, Any]], output_md: Path) -> None:
    lines = [
        "# Phase 3 CX-2 Transform Probe Summary",
        "",
        f"Total rows: {len(rows)}",
        "",
        "| case_id | family_coeff | tolerance | transform | sonc_status | sonc_lb | gp_status | gp_lb |",
        "|---|---:|---:|---|---|---:|---|---:|",
    ]

    for r in rows:
        s = r["results"]["sonc"]
        g = r["results"]["gp"]
        s_lb = "" if s["lower_bound"] is None else f"{s['lower_bound']:.6g}"
        g_lb = "" if g["lower_bound"] is None else f"{g['lower_bound']:.6g}"
        lines.append(
            f"| {r['case_id']} | {r['family_coeff']:.3g} | {r['tolerance']:.0e} | {r['transform_tag']} | {s['status']} | {s_lb} | {g['status']} | {g_lb} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `family_coeff` is the tail coefficient swept in each family.",
            "- `pre` vs `post` compares solver behavior before/after the linear transform used in the manuscript identity.",
            "- SONC/GP statuses are treated as stress diagnostics; failures are recorded with raw solver messages in JSONL.",
        ]
    )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CX-2 transform stress probes")
    parser.add_argument(
        "--output-jsonl",
        default="MeansResearch/results/phase3_runs_clean_cx2_transform_probe.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--output-md",
        default="MeansResearch/results/phase3_pilot_summary_cx2_transform_probe.md",
        help="Output markdown summary",
    )
    parser.add_argument(
        "--tolerances",
        default="1e-6,1e-8",
        help="Comma-separated tolerances",
    )
    parser.add_argument(
        "--q-tail-coeffs",
        default="4.0,3.95,3.9,3.8",
        help="Comma-separated tail coefficients for Q-family",
    )
    parser.add_argument(
        "--rhat-tail-coeffs",
        default="2.0,1.98,1.95,1.9,1.85,1.8",
        help="Comma-separated tail coefficients for Robinson-hat family",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_jsonl = Path(args.output_jsonl)
    output_md = Path(args.output_md)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    tolerances = [float(x.strip()) for x in args.tolerances.split(",") if x.strip()]
    q_tail_coeffs = [float(x.strip()) for x in args.q_tail_coeffs.split(",") if x.strip()]
    rhat_tail_coeffs = [float(x.strip()) for x in args.rhat_tail_coeffs.split(",") if x.strip()]

    cases = build_cases(q_tail_coeffs=q_tail_coeffs, rhat_tail_coeffs=rhat_tail_coeffs)
    sa = SemigroupAlgebra(CommutativeSemigroup(["x", "y", "z", "w"]))
    rows: list[dict[str, Any]] = []

    for case in cases:
        for tol in tolerances:
            row: dict[str, Any] = {
                "case_id": case.id,
                "label": case.label,
                "kind": case.kind,
                "transform_tag": case.transform_tag,
                "family_coeff": case.family_coeff,
                "tolerance": tol,
                "method_family": "SONC+GP",
                "results": run_methods(sa, case.polynomial, tol),
                "structure": structure_diagnostics(sa, case.polynomial),
            }
            rows.append(row)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    write_summary(rows, output_md)
    print(f"Wrote {len(rows)} rows to {output_jsonl}")
    print(f"Wrote summary to {output_md}")


if __name__ == "__main__":
    main()
