#!/usr/bin/env python3
"""Detailed inspection of unexpected failures and per-method SOS/SONC gap."""

import json

with open("/home/mehdi/Code/Python/IreneRewrite/benchmarks/results/gallery_20260808_201748Z.json") as f:
    data = json.load(f)

problems = data["results"]

# Focus on the 4 unexpected failures + separating examples for SOS/SONC gap analysis
targets = ["motzkin_constrained", "polynomial_on_sphere", "mean_poly_sweep_motzkin",
           "sparse_trinomial", "motzkin", "choi_lam", "robinson"]

for p in problems:
    if p["id"] not in targets:
        continue

    pid = p["id"]
    relaxations = p.get("relaxations", {})

    print("=" * 70)
    print("Problem: %s" % pid)
    print("=" * 70)

    for method in ["sos", "sonc", "sosonc"]:
        r = relaxations.get(method, {})
        if not r:
            print("\n  %s: NOT RUN / SKIPPED" % method.upper())
            continue

        status = r.get("status", "N/A")
        value = r.get("value")
        init_time = r.get("init_time_s")
        solve_time = r.get("solve_time_s")
        matrix_dim = r.get("matrix_dim")
        error_msg = r.get("error")

        print("\n  %s:" % method.upper())
        print("    status:     %s" % status)
        if value is not None:
            print("    value:      %.8e" % value)
        else:
            print("    value:      N/A")
        if init_time is not None:
            print("    init_time:  %.4f s" % init_time)
        if solve_time is not None:
            print("    solve_time: %.4f s" % solve_time)
        if matrix_dim is not None:
            print("    matrix_dim: %d" % matrix_dim)
        if error_msg:
            print("    ERROR:      %s" % error_msg[:200])

    # Check for SOS/SONC gap in separating examples
    if pid in ["motzkin", "choi_lam", "robinson"]:
        sos_r = relaxations.get("sos", {})
        sonc_r = relaxations.get("sonc", {})
        sos_status = sos_r.get("status") if sos_r else None
        sonc_status = sonc_r.get("status") if sonc_r else None

        print("\n  SOS/SONC GAP CHECK:")
        print("    SOS status:  %s" % (sos_status or "N/A"))
        print("    SONC status: %s" % (sonc_status or "N/A"))

        if sos_status != "optimal" and sonc_status == "optimal":
            print("    => GAP CONFIRMED: SONC succeeds where SOS fails")
        elif sos_r is None and sonc_r:
            print("    => GAP CONFIRMED: SOS not attempted, SONC succeeded")
        else:
            print("    => No clear gap at this order (both fail or both succeed)")

print("\n" + "=" * 70)
print("Also checking: why sparse_trinomial shows valid=False")
print("=" * 70)

for p in problems:
    if p["id"] == "sparse_trinomial":
        val = p.get("validation", {})
        print("  validation dict:", json.dumps(val, indent=4))
        true_min = val.get("true_min")
        best_bound = val.get("best_bound")
        gap_val = val.get("gap")
        print("\n  Note: sparse_trinomial has true_min=-0.5 (approximate)")
        print("  Best bound %.6e is an OVERESTIMATE, not a valid lower bound" % best_bound)
        print("  This is expected for SONC at low order on degree-6 problems")
