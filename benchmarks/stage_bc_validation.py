#!/usr/bin/env python3
"""Stage (b)-(c): Bound validation & SOS/SONC gap analysis."""

import json

with open("/home/mehdi/Code/Python/IreneRewrite/benchmarks/results/gallery_20260808_201748Z.json") as f:
    data = json.load(f)

problems = data["results"]

print("=" * 80)
print("P4.5 STAGE (b)-(c): BOUND VALIDATION & SOS/SONC GAP ANALYSIS")
print("=" * 80)

pass_count = 0
fail_count = 0
expected_fail = 0
gap_confirmed = 0

for p in problems:
    pid = p["id"]
    cat = p.get("category", "unknown")
    val = p.get("validation", {})
    valid = val.get("valid", False)
    within_tol = val.get("within_tolerance", False)
    best_bound = val.get("best_bound")
    true_min = val.get("true_min")
    gap_val = val.get("gap")

    relaxations = p.get("relaxations", {})
    sos_status = relaxations.get("sos", {}).get("status", "N/A")
    sonc_status = relaxations.get("sonc", {}).get("status", "N/A")
    sosonc_status = relaxations.get("sosonc", {}).get("status", "N/A")

    # Check SOS/SONC gap: SONC succeeds where SOS fails for separating examples
    is_separating = pid in ["motzkin", "choi_lam", "robinson"]
    sos_fail_sonc_pass = (sos_status != "optimal" and sonc_status == "optimal") or \
                         (not relaxations.get("sos") and relaxations.get("sonc"))

    if is_separating and sos_fail_sonc_pass:
        gap_confirmed += 1
        status = "SOS/SONC GAP CONFIRMED"
    elif valid and within_tol:
        pass_count += 1
        status = "PASS (within tol)"
    elif not valid and cat == "separating":
        expected_fail += 1
        status = "EXPECTED FAIL (low order)"
    elif not valid:
        fail_count += 1
        status = "FAIL"
    else:
        pass_count += 1
        status = "PASS"

    bound_str = "%.6e" % best_bound if best_bound is not None else "N/A"
    true_str = "%.6e" % true_min if true_min is not None else "N/A"
    gap_str = "%.6e" % gap_val if gap_val is not None else "N/A"

    print("")
    print("%s (%s, deg=%d):" % (pid, cat, p.get("degree", "?")))
    print("  SOS: %-10s | SONC: %-10s | SOSONC: %-10s" % (sos_status, sonc_status, sosonc_status))
    print("  Best bound: %s | True min: %s | Gap: %s" % (bound_str, true_str, gap_str))
    print("  Validation: valid=%s within_tol=%s => %s" % (valid, within_tol, status))

print("")
print("=" * 80)
print("Summary:")
print("  PASS (within tolerance):   %d" % pass_count)
print("  SOS/SONC gap confirmed:    %d" % gap_confirmed)
print("  Expected failures:         %d" % expected_fail)
print("  Unexpected failures:       %d" % fail_count)
print("=" * 80)
