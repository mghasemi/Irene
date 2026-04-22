#!/usr/bin/env python3
"""Build manuscript-native LaTeX tables from the phase 3 classification CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


STATUS_ABBREV = {
    "success": "S",
    "fail": "F",
    "inconclusive": "I",
    "unknown": "?",
}

METHOD_COLUMNS = ("sdp_status", "sonc_status", "gp_status")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript-native phase 3 tables")
    parser.add_argument(
        "--input-csv",
        default="MeansResearch/results/phase3_classification_table.csv",
        help="Classification CSV input",
    )
    parser.add_argument(
        "--output-tex",
        default="MeansResearch/results/phase3_classification_tables.tex",
        help="Output TeX snippet",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_alpha(alpha: str) -> tuple[int, ...]:
    stripped = alpha.strip().strip("()")
    if not stripped:
        return ()
    return tuple(int(part.strip()) for part in stripped.split(",") if part.strip())


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def alpha_to_tex(alpha: tuple[int, ...]) -> str:
    return "$({})$".format(", ".join(str(value) for value in alpha))


def status_triplet(rows: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for column in METHOD_COLUMNS:
        values = {row.get(column, "unknown") for row in rows}
        value = sorted(values)[0] if len(values) == 1 else "unknown"
        parts.append(STATUS_ABBREV.get(value, "?"))
    return "/".join(parts)


def representative_alphas(rows: list[dict[str, str]]) -> str:
    by_n: dict[int, tuple[int, ...]] = {}
    for row in sorted(rows, key=lambda item: int(item["n"])):
        n = int(row["n"])
        by_n.setdefault(n, parse_alpha(row["alpha"]))
    pieces = [f"$n={n}$: {alpha_to_tex(alpha)}" for n, alpha in sorted(by_n.items())]
    return "; ".join(pieces)


def build_degree_table(rows: list[dict[str, str]], degree: int) -> list[str]:
    degree_rows = [row for row in rows if int(row["d"]) == degree]
    template_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in degree_rows:
        template_groups[row["alpha_template"]].append(row)

    p_values = sorted({int(row["p"]) for row in degree_rows})
    p_headers = [f"$p={p}$" if p != degree else "$p=d$" for p in p_values]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Pilot classification summary for $d={degree}$ (degree ${2 * degree}$). Entries report $(\mathrm{{SDP}}/\mathrm{{SONC}}/\mathrm{{GP}})$ using $\mathrm{{S}}=$ success, $\mathrm{{F}}=$ fail, and $\mathrm{{I}}=$ inconclusive.}}",
        rf"\label{{tab:phase3-classification-d{degree}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{p{1.45cm}p{4.8cm}p{2.15cm}c" + "c" * len(p_values) + r"}",
        r"\hline",
        "template & representative $\\alpha$ & support class & $n$ values & " + " & ".join(p_headers) + " \\\\",
        r"\hline",
    ]

    template_order = ["uniform", "mixed", "boundary", "sparse"]
    for template in template_order:
        group = template_groups.get(template)
        if not group:
            continue
        support_values = sorted({row["support_class"] for row in group})
        support = tex_escape(support_values[0] if len(support_values) == 1 else "mixed")
        n_values = ", ".join(sorted({row["n"] for row in group}, key=int))
        alpha_repr = representative_alphas(group)
        cells: list[str] = []
        for p in p_values:
            slice_rows = [row for row in group if int(row["p"]) == p]
            cells.append(status_triplet(slice_rows) if slice_rows else "--")
        line = " & ".join(
            [
                tex_escape(template),
                alpha_repr,
                support,
                "$\\{" + n_values + "\\}$",
                *cells,
            ]
        ) + " \\\\"
        lines.append(line)

    placeholder = " & ".join(
        [
            "$d\\geq 6$",
            "--",
            "placeholder",
            "$\\{3,4\\}$ planned",
            *([r"\multicolumn{1}{c}{deferred}" for _ in p_values]),
        ]
    ) + " \\\\"
    lines.append(r"\hline")
    lines.append(placeholder)
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )
    return lines


def build_future_note() -> list[str]:
    return [
        r"\paragraph{Deferred higher-degree slice.}",
        r"The $d\geq 6$ rows are retained as placeholders for a future extension of the same protocol. No higher-degree computations are included in the present manuscript snapshot.",
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_tex)
    rows = load_rows(input_path)

    degrees = sorted({int(row["d"]) for row in rows})
    lines = ["% Auto-generated by scripts/phase3_build_manuscript_tables.py", ""]
    for degree in degrees:
        lines.extend(build_degree_table(rows, degree))
        lines.append("")
    lines.extend(build_future_note())
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote manuscript tables to {output_path}")


if __name__ == "__main__":
    main()