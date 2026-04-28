#!/usr/bin/env python3
"""OP2 theorem-stage helper: evaluate a geometry-first Delta_support criterion.

This script consumes:
1) frozen prototype table (d in {4,5}, nondegenerate), and
2) d=6 structural SONC diagnostics (robust-failure stress slice),

and computes a simple support-interiority margin:

    delta_support = min(p, d - p) / d

Interpretation:
- delta_support == 0: boundary exponent path (p in {0,d})
- delta_support > 0: interior exponent path

For the current tested slices, we classify:
- predicted R (robust-failure) if delta_support > 0
- predicted F (feasible) otherwise
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OP2 Delta_support criterion")
    parser.add_argument(
        "--frozen-csv",
        default="MeansResearch/results/op2_prototype_classification_table.csv",
        help="Frozen d=4/5 prototype classification table",
    )
    parser.add_argument(
        "--stress-jsonl",
        default="MeansResearch/results/phase3_sonc_diagnostics_d6_structural.jsonl",
        help="d=6 structural SONC diagnostics JSONL",
    )
    parser.add_argument(
        "--clean-jsonl",
        default="MeansResearch/results/phase3_runs_clean_d6_lc2.jsonl",
        help="d=6 L-C2 clean-pilot JSONL for scope audit",
    )
    parser.add_argument(
        "--output-csv",
        default="MeansResearch/results/op2_delta_support_table.csv",
        help="Combined output table",
    )
    parser.add_argument(
        "--output-json",
        default="MeansResearch/results/op2_delta_support_summary.json",
        help="Summary metrics JSON",
    )
    return parser.parse_args()


def delta_support(d: int, p: int) -> float:
    if d <= 0:
        return 0.0
    return min(p, d - p) / float(d)


def predict_class(d: int, p: int) -> str:
    return "R" if delta_support(d, p) > 0.0 else "F"


def load_frozen_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = int(row["d"])
            p = int(row["p"])
            observed = str(row["observed_class"]).strip()
            pred = predict_class(d, p)
            out.append(
                {
                    "slice": "frozen_d45",
                    "d": d,
                    "n": int(row["n"]),
                    "p": p,
                    "alpha_template": row["alpha_template"],
                    "support_class": row["support_class"],
                    "delta_support": f"{delta_support(d, p):.6f}",
                    "predicted_class": pred,
                    "observed_class": observed,
                    "match": int(pred == observed),
                }
            )
    return out


def load_stress_rows(path: Path) -> list[dict[str, Any]]:
    by_case: dict[tuple[Any, ...], dict[str, Any]] = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case = row.get("case") or {}
            key = (
                int(case.get("d", 0)),
                int(case.get("n", 0)),
                int(case.get("p", 0)),
                tuple(int(x) for x in case.get("alpha", [])),
                str(case.get("alpha_template") or "unknown"),
            )
            by_case.setdefault(key, {"statuses": set(), "case": case})
            by_case[key]["statuses"].add(str(row.get("status") or "unknown"))

    out: list[dict[str, Any]] = []
    for (d, n, p, _alpha, alpha_template), entry in sorted(by_case.items()):
        statuses = entry["statuses"]
        observed = "R" if statuses == {"fail"} else "U"
        pred = predict_class(d, p)
        out.append(
            {
                "slice": "stress_d6_structural",
                "d": d,
                "n": n,
                "p": p,
                "alpha_template": alpha_template,
                "support_class": "simplex-like",
                "delta_support": f"{delta_support(d, p):.6f}",
                "predicted_class": pred,
                "observed_class": observed,
                "match": int(pred == observed),
            }
        )
    return out


def load_clean_rows(path: Path) -> list[dict[str, Any]]:
    by_case: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("method") != "SONCRelaxations":
                continue
            desc = row.get("polynomial_family_descriptor") or {}
            if str(desc.get("support_class") or "") == "degenerate":
                continue

            d = int(desc.get("d", 0))
            n = int(desc.get("n", 0))
            p = int(desc.get("p", 0))
            alpha = tuple(int(x) for x in desc.get("alpha", []))
            alpha_template = str(desc.get("alpha_template") or "unknown")
            status = str(row.get("status") or "unknown")

            key = (d, n, p, alpha, alpha_template)
            if key not in by_case:
                by_case[key] = {"statuses": set()}
            by_case[key]["statuses"].add(status)

    out: list[dict[str, Any]] = []
    for (d, n, p, _alpha, alpha_template), entry in sorted(by_case.items()):
        statuses = entry["statuses"]
        if statuses == {"success"}:
            observed = "F"
        elif statuses == {"fail"}:
            observed = "R"
        else:
            observed = "U"

        pred = predict_class(d, p)
        out.append(
            {
                "slice": "clean_d6_lc2",
                "d": d,
                "n": n,
                "p": p,
                "alpha_template": alpha_template,
                "support_class": "simplex-like",
                "delta_support": f"{delta_support(d, p):.6f}",
                "predicted_class": pred,
                "observed_class": observed,
                "match": int(pred == observed),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"by_slice": {}}
    slices = sorted({r["slice"] for r in rows})
    for sl in slices:
        subset = [r for r in rows if r["slice"] == sl]
        total = len(subset)
        matches = sum(int(r["match"]) for r in subset)
        pred_counts = Counter(r["predicted_class"] for r in subset)
        obs_counts = Counter(r["observed_class"] for r in subset)
        by_p: dict[str, dict[str, int]] = {}
        for p in sorted({int(r["p"]) for r in subset}):
            p_rows = [r for r in subset if int(r["p"]) == p]
            by_p[str(p)] = {
                "total": len(p_rows),
                "matches": sum(int(r["match"]) for r in p_rows),
                "pred_R": sum(1 for r in p_rows if r["predicted_class"] == "R"),
                "obs_R": sum(1 for r in p_rows if r["observed_class"] == "R"),
            }

        summary["by_slice"][sl] = {
            "rows": total,
            "matches": matches,
            "accuracy": (matches / total) if total else None,
            "predicted_counts": dict(pred_counts),
            "observed_counts": dict(obs_counts),
            "by_p": by_p,
        }
    return summary


def main() -> None:
    args = parse_args()
    frozen_csv = Path(args.frozen_csv)
    stress_jsonl = Path(args.stress_jsonl)
    clean_jsonl = Path(args.clean_jsonl)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    rows = load_frozen_rows(frozen_csv) + load_stress_rows(stress_jsonl) + load_clean_rows(clean_jsonl)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "slice",
                "d",
                "n",
                "p",
                "alpha_template",
                "support_class",
                "delta_support",
                "predicted_class",
                "observed_class",
                "match",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for sl, metrics in summary["by_slice"].items():
        acc = metrics["accuracy"]
        acc_str = "n/a" if acc is None else f"{acc:.3f}"
        print(f"{sl}: rows={metrics['rows']} matches={metrics['matches']} accuracy={acc_str}")

    print(f"Wrote table: {output_csv}")
    print(f"Wrote summary: {output_json}")


if __name__ == "__main__":
    main()
