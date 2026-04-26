#!/usr/bin/env python3
"""Phase 3 benchmark runner for mean-polynomial experiments.

This script implements the first executable slice of the Phase 3 matrix:
- L-C1: SDP/SOS pilot checks for M_{2d,p}(X, alpha)
- L-C2: SONC/GP behavior on the same families

Logs are written as JSONL records to support reproducibility and ledger updates.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Irene.geometric import GPRelaxations
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxations import SDPRelaxations
from Irene.sonc import SONCRelaxations


DEFAULT_TOLERANCES = (1e-6, 1e-8)


@dataclass(frozen=True)
class BenchmarkCase:
    item_id: str
    d: int
    n: int
    p: int
    alpha: tuple[int, ...]
    alpha_template: str

    @property
    def q(self) -> int:
        return 2 * self.d


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def support_class_from_objective(objective: Any) -> str:
    if objective is None or not hasattr(objective, "content"):
        return "degenerate"

    support: set[tuple[int, ...]] = set()
    generators = [gen.ext_rep[0].name for gen in objective.semigroup.generators]

    for coeff, mono in getattr(objective, "content", []):
        try:
            if abs(float(coeff)) <= 1e-14:
                continue
        except Exception:
            continue

        exp_map = {sym.name: int(power) for sym, power in mono.array_form}
        exp = tuple(exp_map.get(name, 0) for name in generators)
        support.add(exp)

    if len(support) <= 1:
        return "degenerate"

    points = np.array(sorted(support), dtype=float)
    centered = points - points[0]
    affine_dim = int(np.linalg.matrix_rank(centered))
    if affine_dim <= 0:
        return "degenerate"

    if affine_dim == 1:
        num_vertices = 2
    else:
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            basis = vh[:affine_dim].T
            projected = centered @ basis
            hull = ConvexHull(projected)
            num_vertices = len(hull.vertices)
        except (np.linalg.LinAlgError, QhullError, ValueError):
            return "degenerate"

    if num_vertices == affine_dim + 1:
        return "simplex-like"
    if num_vertices > affine_dim + 1:
        return "non-simplex"
    return "degenerate"


def is_structurally_degenerate(case: BenchmarkCase) -> bool:
    # One-hot alpha yields M_{2d,p} == 0 for all supported p in this generator.
    return sum(1 for a in case.alpha if a != 0) <= 1


def alpha_templates(total_degree: int, n: int) -> dict[str, tuple[int, ...]]:
    base = total_degree // n
    rem = total_degree % n
    uniform = [base] * n
    for i in range(rem):
        uniform[i] += 1

    sparse = [0] * n
    sparse[0] = total_degree

    boundary = [0] * n
    boundary[0] = max(total_degree - 1, 0)
    if n > 1:
        boundary[1] = 1

    mixed = [0] * n
    if n == 1:
        mixed[0] = total_degree
    elif n == 2:
        mixed[0] = total_degree // 2
        mixed[1] = total_degree - mixed[0]
    else:
        mixed[0] = total_degree // 2
        mixed[1] = total_degree // 3
        mixed[2] = total_degree - mixed[0] - mixed[1]

    out = {
        "uniform": tuple(uniform),
        "sparse": tuple(sparse),
        "boundary": tuple(boundary),
        "mixed": tuple(mixed),
    }
    # Keep only valid templates and remove duplicates while preserving first label.
    dedup: dict[tuple[int, ...], str] = {}
    for name, alpha in out.items():
        if len(alpha) != n or sum(alpha) != total_degree or any(a < 0 for a in alpha):
            continue
        dedup.setdefault(alpha, name)
    return {name: alpha for alpha, name in dedup.items()}


def divisors_up_to_d(q: int, d: int) -> list[int]:
    return [p for p in range(1, d + 1) if q % p == 0]


def build_case_family(item_id: str, d_values: list[int], n_values: list[int]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for d in d_values:
        q = 2 * d
        p_values = [0] + divisors_up_to_d(q, d)
        for n in n_values:
            templates = alpha_templates(q, n)
            for template_name, alpha in templates.items():
                for p in p_values:
                    cases.append(
                        BenchmarkCase(
                            item_id=item_id,
                            d=d,
                            n=n,
                            p=p,
                            alpha=alpha,
                            alpha_template=template_name,
                        )
                    )
    return cases


def monomial_from_alpha(variables: list[Any], alpha: tuple[int, ...]) -> Any:
    mono = 1
    for v, power in zip(variables, alpha):
        mono = mono * (v ** int(power))
    return mono


def mean_polynomial(case: BenchmarkCase) -> tuple[CommutativeSemigroup, SemigroupAlgebra, Any]:
    names = [f"x{i+1}" for i in range(case.n)]
    semigroup = CommutativeSemigroup(names)
    algebra = SemigroupAlgebra(semigroup)
    variables = [algebra[name] for name in names]

    q = case.q
    first = 0
    for ai, xi in zip(case.alpha, variables):
        first = first + ai * (xi ** q)
    first = (1.0 / q) * first

    if case.p == 0:
        second = monomial_from_alpha(variables, case.alpha)
    else:
        inner = 0
        for ai, xi in zip(case.alpha, variables):
            inner = inner + ai * (xi ** case.p)
        second = ((1.0 / q) * inner) ** (q // case.p)

    return semigroup, algebra, (first - second)


def create_problem(case: BenchmarkCase) -> tuple[OptimizationProblem, Any]:
    _semigroup, algebra, objective = mean_polynomial(case)
    problem = OptimizationProblem(algebra)
    problem.set_objective(objective)
    return problem, objective


def run_sdp(case: BenchmarkCase, tolerance: float, solver: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "method": "SDPRelaxations",
        "solver_config": {
            "solver": solver.upper(),
            "error_tolerance": tolerance,
            "parallel": False,
        },
        "objective": None,
    }

    start = time.perf_counter()
    try:
        problem, objective = create_problem(case)
        entry["objective"] = str(objective)
        relax = SDPRelaxations.from_problem(problem, name="phase3_lc1")
        relax.ErrorTolerance = tolerance
        relax.Parallel = False
        relax.SetSDPSolver(solver)
        relax.InitSDP()
        f_min = relax.Minimize()

        gram_rank = None
        decomposition_ok = False
        decomposition_size = None
        decomp_error = None
        if relax.Solution is not None and hasattr(relax.Solution, "MomentMatrix"):
            try:
                gram_rank = int(np.linalg.matrix_rank(relax.Solution.MomentMatrix))
            except Exception:
                gram_rank = None

        if relax.Info.get("status") == "Optimal":
            try:
                sos = relax.Decompose()
                decomposition_ok = True
                decomposition_size = {str(k): len(v) for k, v in sos.items()}
            except Exception as exc:
                decomp_error = str(exc)

        entry.update(
            {
                "status": "success" if relax.Info.get("status") == "Optimal" else "infeasible",
                "bounds": {
                    "sdp_lower_bound": f_min,
                    "primal": getattr(relax.Solution, "Primal", None),
                    "dual": getattr(relax.Solution, "Dual", None),
                },
                "runtime_sec": float(time.perf_counter() - start),
                "decomposition_diagnostics": {
                    "decompose_ok": decomposition_ok,
                    "decompose_error": decomp_error,
                    "decompose_terms": decomposition_size,
                    "gram_rank": gram_rank,
                    "moment_order": relax.MmntOrd,
                    "status_message": relax.Info.get("Message"),
                },
            }
        )
    except Exception as exc:
        entry.update(
            {
                "status": "fail",
                "bounds": None,
                "runtime_sec": float(time.perf_counter() - start),
                "decomposition_diagnostics": None,
                "notes": str(exc),
            }
        )
    return entry


def run_sonc(case: BenchmarkCase, tolerance: float) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "method": "SONCRelaxations",
        "solver_config": {
            "error_bound": tolerance,
            "use_local_solve": True,
        },
        "objective": None,
    }
    start = time.perf_counter()
    try:
        problem, objective = create_problem(case)
        entry["objective"] = str(objective)
        sonc = SONCRelaxations(problem, error_bound=tolerance, verbosity=0, use_local_solve=True)
        value = sonc.solve(verbosity=0)
        entry.update(
            {
                "status": "success",
                "bounds": {"sonc_lower_bound": value},
                "runtime_sec": float(time.perf_counter() - start),
                "decomposition_diagnostics": {
                    "solution_available": sonc.solution is not None,
                },
            }
        )
    except Exception as exc:
        entry.update(
            {
                "status": "fail",
                "bounds": None,
                "runtime_sec": float(time.perf_counter() - start),
                "decomposition_diagnostics": None,
                "notes": str(exc),
            }
        )
    return entry


def run_gp(case: BenchmarkCase) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "method": "GPRelaxations",
        "solver_config": {
            "auto_transform": True,
        },
        "objective": None,
    }
    start = time.perf_counter()
    try:
        problem, objective = create_problem(case)
        entry["objective"] = str(objective)
        gp = GPRelaxations(problem, auto_transform=True, verbosity=0)
        value = gp.solve()
        entry.update(
            {
                "status": "success",
                "bounds": {"gp_lower_bound": value},
                "runtime_sec": float(time.perf_counter() - start),
                "decomposition_diagnostics": {
                    "solution_available": gp.solution is not None,
                },
            }
        )
    except Exception as exc:
        entry.update(
            {
                "status": "fail",
                "bounds": None,
                "runtime_sec": float(time.perf_counter() - start),
                "decomposition_diagnostics": None,
                "notes": str(exc),
            }
        )
    return entry


def make_inconclusive_result(method: str, note: str) -> dict[str, Any]:
    return {
        "method": method,
        "status": "inconclusive",
        "bounds": None,
        "runtime_sec": 0.0,
        "decomposition_diagnostics": None,
        "notes": note,
        "objective": None,
        "solver_config": {"skipped": True},
    }


def build_record_base(case: BenchmarkCase, commit_hash: str, tolerance: float) -> dict[str, Any]:
    support = "unknown"
    generation_error = None

    if is_structurally_degenerate(case):
        support = "degenerate"
        generation_error = "structurally_degenerate_family"
    else:
        try:
            _problem, objective = create_problem(case)
            support = support_class_from_objective(objective)
        except Exception as exc:
            generation_error = str(exc)

    anomaly_flags: list[str] = []
    if generation_error is not None:
        anomaly_flags.append("family_generation_issue")

    return {
        "id": f"{case.item_id}-d{case.d}-n{case.n}-p{case.p}-a{'_'.join(str(x) for x in case.alpha)}-tol{tolerance}",
        "timestamp": now_iso(),
        "commit_hash": commit_hash,
        "polynomial_family_descriptor": {
            "item_id": case.item_id,
            "q": case.q,
            "d": case.d,
            "n": case.n,
            "p": case.p,
            "alpha": list(case.alpha),
            "alpha_template": case.alpha_template,
            "support_class": support,
        },
        "tolerance_setting": tolerance,
        "anomaly_flags": anomaly_flags,
        "family_generation_error": generation_error,
    }


def append_record(path: Path, row: dict[str, Any]) -> None:
    """Append a single record to a JSONL file immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def load_existing_ids(path: Path) -> set[str]:
    """Return the set of record IDs already present in output file."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_items(value: str) -> list[str]:
    return [v.strip().upper() for v in value.split(",") if v.strip()]


def parse_csv_solvers(value: str) -> list[str]:
    solvers = [v.strip().lower() for v in value.split(",") if v.strip()]
    seen: set[str] = set()
    deduped: list[str] = []
    for s in solvers:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 mean-polynomial benchmarks")
    parser.add_argument("--items", default="L-C1,L-C2", help="Comma-separated item IDs (L-C1,L-C2)")
    parser.add_argument("--degrees", default="4", help="Comma-separated d values, e.g. 4 or 4,5,6")
    parser.add_argument("--variables", default="3,4", help="Comma-separated n values, e.g. 3,4")
    parser.add_argument("--tolerances", default="1e-6,1e-8", help="Comma-separated numeric tolerances")
    parser.add_argument(
        "--case-offset",
        type=int,
        default=0,
        help="Skip the first N generated cases before applying --max-cases",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap on number of generated cases")
    parser.add_argument("--sdp-solver", default="cvxopt", help="SDP solver name accepted by Irene")
    parser.add_argument(
        "--sdp-solver-seq",
        default="",
        help="Optional comma-separated solver fallback sequence (e.g. cvxopt,dsdp,sdpa,csdp). If set, overrides --sdp-solver.",
    )
    parser.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Include structurally degenerate families (one-hot alpha) instead of logging them as inconclusive.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases whose ID is already present in the output file (resume support).",
    )
    parser.add_argument(
        "--solve-timeout",
        type=int,
        default=0,
        help="Per-solve wall-clock timeout in seconds (0 = no limit). Timed-out solves are logged as status=timeout.",
    )
    parser.add_argument(
        "--output",
        default="MeansResearch/results/phase3_runs.jsonl",
        help="JSONL output path (relative to repo root)",
    )
    return parser.parse_args()


def _run_sdp_worker(
    case: BenchmarkCase,
    tol: float,
    solver: str,
    result_queue: "multiprocessing.Queue[dict[str, Any]]",
) -> None:
    try:
        result_queue.put(run_sdp(case, tol, solver))
    except Exception as exc:
        result_queue.put(
            {
                "method": "SDPRelaxations",
                "status": "fail",
                "bounds": None,
                "runtime_sec": 0.0,
                "decomposition_diagnostics": None,
                "notes": str(exc),
                "objective": None,
                "solver_config": {"solver": solver.upper()},
            }
        )


def run_sdp_with_timeout(
    case: BenchmarkCase,
    tol: float,
    solver: str,
    timeout_sec: int,
) -> dict[str, Any]:
    """Run SDP in a subprocess so it can be killed if it exceeds the timeout."""
    if timeout_sec <= 0:
        return run_sdp(case, tol, solver)

    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_run_sdp_worker, args=(case, tol, solver, q))
    p.start()
    p.join(timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return {
            "method": "SDPRelaxations",
            "status": "timeout",
            "bounds": None,
            "runtime_sec": float(timeout_sec),
            "decomposition_diagnostics": None,
            "notes": f"Solve exceeded timeout of {timeout_sec}s.",
            "objective": None,
            "solver_config": {"solver": solver.upper(), "error_tolerance": tol},
        }
    return q.get()


def run_sdp_with_fallbacks(
    case: BenchmarkCase,
    tol: float,
    solver_sequence: list[str],
    timeout_sec: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for solver in solver_sequence:
        result = run_sdp_with_timeout(case, tol, solver, timeout_sec)
        attempt_status = result.get("status")
        attempts.append(
            {
                "solver": solver.upper(),
                "status": attempt_status,
                "notes": result.get("notes"),
                "runtime_sec": result.get("runtime_sec"),
            }
        )
        # Stop fallback chain only on successful SDP solve.
        if attempt_status == "success":
            result["solver_attempts"] = attempts
            return result

    # No success: return the last attempt with full attempt history.
    if attempts:
        result["solver_attempts"] = attempts
    return result


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    output = (repo_root / args.output).resolve()
    item_ids = parse_csv_items(args.items)
    d_values = parse_csv_ints(args.degrees)
    n_values = parse_csv_ints(args.variables)
    tolerances = tuple(float(v.strip()) for v in args.tolerances.split(",") if v.strip())
    if not tolerances:
        tolerances = DEFAULT_TOLERANCES

    solver_sequence = parse_csv_solvers(args.sdp_solver_seq) if args.sdp_solver_seq else [args.sdp_solver.lower()]
    if not solver_sequence:
        solver_sequence = ["cvxopt"]

    valid_items = {"L-C1", "L-C2"}
    item_ids = [itm for itm in item_ids if itm in valid_items]
    if not item_ids:
        raise ValueError("No valid items selected. Supported: L-C1,L-C2")

    all_cases: list[BenchmarkCase] = []
    for item_id in item_ids:
        all_cases.extend(build_case_family(item_id, d_values, n_values))

    if args.case_offset > 0:
        all_cases = all_cases[args.case_offset :]

    if args.max_cases > 0:
        all_cases = all_cases[: args.max_cases]

    existing_ids: set[str] = set()
    if args.skip_existing:
        existing_ids = load_existing_ids(output)
        if existing_ids:
            print(f"Skipping {len(existing_ids)} already-completed IDs.")

    commit_hash = git_commit_hash(repo_root)
    total_written = 0
    for case in all_cases:
        for tol in tolerances:
            base = build_record_base(case, commit_hash, tol)
            record_id = base["id"]

            if record_id in existing_ids:
                continue

            methods: list[dict[str, Any]] = []

            if is_structurally_degenerate(case) and not args.include_degenerate:
                skip_note = "Skipped structurally degenerate family (one-hot alpha yields zero objective)."
                if case.item_id == "L-C1":
                    methods.append(make_inconclusive_result("SDPRelaxations", skip_note))
                elif case.item_id == "L-C2":
                    methods.append(make_inconclusive_result("SONCRelaxations", skip_note))
                    methods.append(make_inconclusive_result("GPRelaxations", skip_note))
            else:
                if case.item_id == "L-C1":
                    methods.append(
                        run_sdp_with_fallbacks(case, tol, solver_sequence, args.solve_timeout)
                    )
                elif case.item_id == "L-C2":
                    methods.append(run_sonc(case, tol))
                    methods.append(run_gp(case))

            for method_result in methods:
                row = dict(base)
                row.update(method_result)
                if row.get("status") not in {"success", "inconclusive", "timeout"}:
                    row["anomaly_flags"].append("method_failure")
                append_record(output, row)
                total_written += 1

    print(f"Wrote {total_written} records to {output}")


if __name__ == "__main__":
    main()
