#!/usr/bin/env python3
"""API inventory for a given Irene package instance.

Usage: python3 api_inventory.py /path/to/package_root [output.json]

Walks all public modules in the Irene package, extracts classes, functions,
methods and their signatures, and dumps a JSON structure.
"""
import importlib
import inspect
import json
import os
import sys
import pkgutil

PKG_DIR = os.path.abspath(sys.argv[1])
sys.path.insert(0, PKG_DIR)
OUT = sys.argv[2] if len(sys.argv) > 2 else None

try:
    import Irene as pkg
except Exception as exc:  # pragma: no cover
    print(json.dumps({"error": f"import failed: {exc}"}))
    sys.exit(1)

MODULES = [
    "base", "sdp", "relaxations", "sosonc", "sonc", "geometric",
    "grouprings", "program", "matrices", "invariant", "dsdp",
    "border_basis", "newton_polytope", "sparsity", "correlative_sparsity",
    "unified_reductions", "nonpopsdp", "relaxation_api", "cvxpy_solver",
    "symbolic_engine", "telemetry",
]


def sig_of(obj):
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "<no signature>"


def members_of(module, name):
    """Public classes + functions defined in module `name`."""
    out = {"classes": {}, "functions": {}}
    try:
        mod = importlib.import_module(f"Irene.{name}")
    except Exception as exc:
        out["_import_error"] = str(exc)
        return out
    for mname, mobj in inspect.getmembers(mod):
        if mname.startswith("_"):
            continue
        if inspect.isclass(mobj) and mobj.__module__ == f"Irene.{name}":
            out["classes"][mname] = sig_of(mobj)
        elif inspect.isfunction(mobj) and mobj.__module__ == f"Irene.{name}":
            out["functions"][mname] = sig_of(mobj)
    # Methods of each class
    for cname, _ in out["classes"].items():
        cls = getattr(mod, cname)
        methods = {}
        for meth_name, meth in inspect.getmembers(cls):
            if meth_name.startswith("_"):
                continue
            if callable(meth):
                try:
                    methods[meth_name] = sig_of(meth)
                except Exception:
                    pass
        out["classes"][cname] = {"__init__": sig_of(cls), "methods": methods}
    return out


result = {"package": str(getattr(pkg, "__version__", "?")), "modules": {}}
for name in MODULES:
    result["modules"][name] = members_of(pkg, name)

if OUT:
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    print(f"wrote {OUT}")
else:
    print(json.dumps(result, indent=1, sort_keys=True))
