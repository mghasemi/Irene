#!/usr/bin/env python3
"""Build combined SOS/SONC/GP classification table from clean phase3 JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.phase3_benchmarks import BenchmarkCase, mean_polynomial, support_class_from_objective


METHOD_SDP = "SDPRelaxations"
METHOD_SONC = "SONCRelaxations"
METHOD_GP = "GPRelaxations"


def recompute_support_class(desc: dict[str, Any]) -> str:
    try:
        case = BenchmarkCase(
            item_id=str(desc.get("item_id") or "L-C1"),
            d=int(desc["d"]),
            n=int(desc["n"]),
            p=int(desc["p"]),
            alpha=tuple(int(x) for x in desc.get("alpha", [])),
            alpha_template=str(desc.get("alpha_template") or "unknown"),
        )
        _sg, _alg, objective = mean_polynomial(case)
        return support_class_from_objective(objective)
    except Exception:
        return str(desc.get("support_class") or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined classification table from clean runs")
    parser.add_argument(
        "--inputs",
        default="MeansResearch/results/phase3_runs_clean.jsonl,MeansResearch/results/phase3_runs_clean_d5.jsonl",
        help="Comma-separated clean JSONL paths",
    )
    parser.add_argument(
        "--output-csv",
        default="MeansResearch/results/phase3_classification_table.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-md",
        default="MeansResearch/results/phase3_classification_table.md",
        help="Output Markdown path",
    )
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}

    for row in rows:
        desc = row.get("polynomial_family_descriptor") or {}
        key = (
            desc.get("d"),
            desc.get("n"),
            desc.get("p"),
            tuple(desc.get("alpha", [])),
            desc.get("alpha_template"),
        )
        if key not in groups:
            support = recompute_support_class(desc)
            groups[key] = {
                "d": desc.get("d"),
                "n": desc.get("n"),
                "p": desc.get("p"),
                "alpha": tuple(desc.get("alpha", [])),
                "alpha_template": desc.get("alpha_template"),
                "support_class": support,
                METHOD_SDP: set(),
                METHOD_SONC: set(),
                METHOD_GP: set(),
                "pair_diffs": [],
            }

        g = groups[key]
        method = row.get("method")
        status = row.get("status")
        if method in (METHOD_SDP, METHOD_SONC, METHOD_GP):
            g[method].add(status)

    # second pass for paired differences
    pair_map: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: {METHOD_SONC: [], METHOD_GP: []})
    for row in rows:
        desc = row.get("polynomial_family_descriptor") or {}
        key = (
            desc.get("d"),
            desc.get("n"),
            desc.get("p"),
            tuple(desc.get("alpha", [])),
            desc.get("alpha_template"),
            row.get("tolerance_setting"),
        )
        method = row.get("method")
        if method == METHOD_SONC and row.get("status") == "success":
            val = (row.get("bounds") or {}).get("sonc_lower_bound")
            if isinstance(val, (int, float)):
                pair_map[key][METHOD_SONC].append(float(val))
        if method == METHOD_GP and row.get("status") == "success":
            val = (row.get("bounds") or {}).get("gp_lower_bound")
            if isinstance(val, (int, float)):
                pair_map[key][METHOD_GP].append(float(val))

    for key, vals in pair_map.items():
        if vals[METHOD_SONC] and vals[METHOD_GP]:
            d, n, p, alpha, template, _tol = key
            gkey = (d, n, p, alpha, template)
            if gkey in groups:
                diff = vals[METHOD_GP][0] - vals[METHOD_SONC][0]
                groups[gkey]["pair_diffs"].append(diff)

    out: list[dict[str, Any]] = []
    for g in groups.values():
        sdp_status = "success" if g[METHOD_SDP] == {"success"} else "/".join(sorted(g[METHOD_SDP]))
        sonc_status = "success" if g[METHOD_SONC] == {"success"} else "/".join(sorted(g[METHOD_SONC]))
        gp_status = "success" if g[METHOD_GP] == {"success"} else "/".join(sorted(g[METHOD_GP]))
        if g["pair_diffs"]:
            min_diff = min(g["pair_diffs"])
            max_diff = max(g["pair_diffs"])
            pair_range = f"[{min_diff:.2e}, {max_diff:.2e}]"
        else:
            pair_range = "n/a"
        out.append(
            {
                "d": g["d"],
                "n": g["n"],
                "p": g["p"],
                "alpha": str(g["alpha"]),
                "alpha_template": g["alpha_template"],
                "support_class": g["support_class"],
                "sdp_status": sdp_status,
                "sonc_status": sonc_status,
                "gp_status": gp_status,
                "gp_minus_sonc_range": pair_range,
            }
        )

    out.sort(key=lambda r: (r["d"], r["n"], r["p"], r["alpha_template"], r["alpha"]))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 3 Classification Table", "", f"Rows: {len(rows)}", ""]
    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]
    rows = load_rows(paths)
    table = aggregate(rows)
    write_csv(Path(args.output_csv), table)
    write_md(Path(args.output_md), table)
    print(f"Wrote {len(table)} rows to {args.output_csv} and {args.output_md}")


if __name__ == "__main__":
    main()
