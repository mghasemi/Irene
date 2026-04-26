#!/usr/bin/env python3
"""CLI for exact symbolic mathematics using SymPy."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _require_sympy() -> Any:
    try:
        import sympy  # noqa: PLC0415
        return sympy
    except ImportError:
        _fail("SymPy is not installed. Run: pip install sympy")


def _fail(message: str, fmt: str = "text") -> None:
    if fmt == "json":
        print(json.dumps({"success": False, "error": message}))
    else:
        print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _result(data: dict[str, Any], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(data, ensure_ascii=False))
    else:
        print(data.get("result", ""))
        if data.get("latex"):
            print(f"LaTeX: {data['latex']}")


def _to_latex(sp: Any, expr: Any) -> str:
    try:
        return sp.latex(expr)
    except Exception:
        return str(expr)


def cmd_solve(sp: Any, args: argparse.Namespace) -> dict[str, Any]:
    var = sp.Symbol(args.var)
    expr = sp.sympify(args.expression)
    solutions = sp.solve(expr, var)
    return {
        "command": "solve",
        "expression": args.expression,
        "var": args.var,
        "result": str(solutions),
        "latex": _to_latex(sp, solutions),
        "success": True,
    }


def cmd_diff(sp: Any, args: argparse.Namespace) -> dict[str, Any]:
    var = sp.Symbol(args.var)
    expr = sp.sympify(args.expression)
    result = sp.diff(expr, var, args.n)
    return {
        "command": "diff",
        "expression": args.expression,
        "var": args.var,
        "n": args.n,
        "result": str(result),
        "latex": _to_latex(sp, result),
        "success": True,
    }


def cmd_integrate(sp: Any, args: argparse.Namespace) -> dict[str, Any]:
    var = sp.Symbol(args.var)
    expr = sp.sympify(args.expression)
    if args.limits:
        parts = args.limits.split()
        if len(parts) != 2:
            _fail("--limits must be two space-separated values, e.g. '0 pi'")
        lo = sp.sympify(parts[0])
        hi = sp.sympify(parts[1])
        result = sp.integrate(expr, (var, lo, hi))
    else:
        result = sp.integrate(expr, var)
    return {
        "command": "integrate",
        "expression": args.expression,
        "var": args.var,
        "limits": args.limits,
        "result": str(result),
        "latex": _to_latex(sp, result),
        "success": True,
    }


def cmd_dsolve(sp: Any, args: argparse.Namespace) -> dict[str, Any]:
    func_sym = sp.Function(args.func)
    x = sp.Symbol("x")
    ode_expr = sp.sympify(args.expression, locals={args.func: func_sym, "x": x})
    result = sp.dsolve(ode_expr, func_sym(x))
    return {
        "command": "dsolve",
        "expression": args.expression,
        "func": args.func,
        "result": str(result),
        "latex": _to_latex(sp, result),
        "success": True,
    }


def cmd_simplify(sp: Any, args: argparse.Namespace) -> dict[str, Any]:
    expr = sp.sympify(args.expression)
    result = sp.simplify(expr)
    return {
        "command": "simplify",
        "expression": args.expression,
        "result": str(result),
        "latex": _to_latex(sp, result),
        "success": True,
    }


def cmd_factor(sp: Any, args: argparse.Namespace) -> dict[str, Any]:
    expr = sp.sympify(args.expression)
    result = sp.factor(expr)
    return {
        "command": "factor",
        "expression": args.expression,
        "result": str(result),
        "latex": _to_latex(sp, result),
        "success": True,
    }


def cmd_expand(sp: Any, args: argparse.Namespace) -> dict[str, Any]:
    expr = sp.sympify(args.expression)
    result = sp.expand(expr)
    return {
        "command": "expand",
        "expression": args.expression,
        "result": str(result),
        "latex": _to_latex(sp, result),
        "success": True,
    }


def cmd_latex(sp: Any, args: argparse.Namespace) -> dict[str, Any]:
    expr = sp.sympify(args.expression)
    latex_str = sp.latex(expr)
    return {
        "command": "latex",
        "expression": args.expression,
        "result": latex_str,
        "latex": latex_str,
        "success": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SymPy symbolic computation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="Solve expr=0 for a variable")
    p_solve.add_argument("expression")
    p_solve.add_argument("--var", default="x")
    p_solve.add_argument("--format", choices=["text", "json"], default="text")

    p_diff = sub.add_parser("diff", help="Differentiate expression")
    p_diff.add_argument("expression")
    p_diff.add_argument("--var", default="x")
    p_diff.add_argument("--n", type=int, default=1, help="Order of differentiation")
    p_diff.add_argument("--format", choices=["text", "json"], default="text")

    p_int = sub.add_parser("integrate", help="Integrate expression")
    p_int.add_argument("expression")
    p_int.add_argument("--var", default="x")
    p_int.add_argument("--limits", default=None, help="Two space-separated bounds, e.g. '0 pi'")
    p_int.add_argument("--format", choices=["text", "json"], default="text")

    p_dsolve = sub.add_parser("dsolve", help="Solve an ODE")
    p_dsolve.add_argument("expression", help="ODE expression, e.g. f(x).diff(x) - f(x)")
    p_dsolve.add_argument("--func", default="f", help="Function symbol name (default: f)")
    p_dsolve.add_argument("--format", choices=["text", "json"], default="text")

    p_simp = sub.add_parser("simplify", help="Simplify expression")
    p_simp.add_argument("expression")
    p_simp.add_argument("--format", choices=["text", "json"], default="text")

    p_fac = sub.add_parser("factor", help="Factor expression")
    p_fac.add_argument("expression")
    p_fac.add_argument("--format", choices=["text", "json"], default="text")

    p_exp = sub.add_parser("expand", help="Expand expression")
    p_exp.add_argument("expression")
    p_exp.add_argument("--format", choices=["text", "json"], default="text")

    p_lat = sub.add_parser("latex", help="Convert expression to LaTeX string")
    p_lat.add_argument("expression")
    p_lat.add_argument("--format", choices=["text", "json"], default="text")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fmt = getattr(args, "format", "text")
    sp = _require_sympy()

    dispatch = {
        "solve": cmd_solve,
        "diff": cmd_diff,
        "integrate": cmd_integrate,
        "dsolve": cmd_dsolve,
        "simplify": cmd_simplify,
        "factor": cmd_factor,
        "expand": cmd_expand,
        "latex": cmd_latex,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        _fail(f"Unknown command: {args.command}", fmt)

    try:
        data = handler(sp, args)
    except Exception as exc:
        _fail(str(exc), fmt)
        return

    _result(data, fmt)


if __name__ == "__main__":
    main()
