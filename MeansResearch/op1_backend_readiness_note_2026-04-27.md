# OP1 Backend Readiness Note (Session 1)

Date: 2026-04-27
Scope: Session 1 backend gating for OP1 E1 pilot (d=6, n=5, L-C1).

## Decision Summary

- MOSEK is not available in the current environment (user-confirmed, also absent as Python module).
- SDPA-GMP is not detected under expected CLI names in current PATH (`sdpa_gmp`, `sdpa-gmp`) and could not be installed from currently available conda/pip sources in this environment.
- DSDP is installed at `/home/mehdi/miniforge3/envs/sage/bin/dsdp5` and is runnable.
- Irene recognizes DSDP as available when PATH is prefixed with `/home/mehdi/miniforge3/envs/sage/bin`.
- Result: at least one stronger backend (DSDP) is runnable with environment-path integration, so Session 1 gate is satisfied.

## Environment Evidence

Probe method:
- Python module checks using venv interpreter.
- CLI binary checks via PATH lookup.

Probe result snapshot:
- Python modules:
  - mosek: false
  - dsdp: false
  - pydsdp: false
  - sdpa: false
  - cvxopt: true
- CLI binaries:
  - sdpa_gmp: not found
  - sdpa-gmp: not found
  - sdpa: /usr/bin/sdpa
  - csdp: /usr/bin/csdp
  - dsdp5: not found in current PATH
  - dsdp: not found in current PATH

Additional location probes:
- DSDP executable found: `/home/mehdi/miniforge3/envs/sage/bin/dsdp5`
- DSDP smoke check: binary executes (expects problem file input)
- Irene discovery check with prefixed PATH:
  - `which(dsdp5)` resolves to `/home/mehdi/miniforge3/envs/sage/bin/dsdp5`
  - `AvailableSDPSolvers()` returns `['CVXOPT', 'DSDP', 'SDPA', 'CSDP']`

SDPA-GMP installation attempts (2026-04-27):
- `conda install -n sage -c conda-forge sdpa-multiprecision` -> `PackagesNotFoundError`.
- `pip install sdpa-multiprecision` in `sage` env -> `No matching distribution found`.
- `apt-cache search sdpa` lists `sdpa`, `libsdpa-dev`, `sdpam` but no SDPA-GMP package.

Interpretation:
- Baseline stack components SDPA and CSDP are present.
- SDPA-GMP remains unavailable under current channels/package indexes used in this workspace.
- DSDP is accessible for OP1 runs once PATH is prefixed to include the discovered dsdp5 location.

## Locked E1 Pilot Manifest (6 representative nondegenerate cases)

Selection rule:
- 2 p-values (p=1 and p=4) x 3 template families (uniform, boundary, mixed), all at d=6, n=5, tol=1e-6.

Locked IDs:
1. L-C1-d6-n5-p1-a3_3_2_2_2-tol1e-06 (uniform)
2. L-C1-d6-n5-p4-a3_3_2_2_2-tol1e-06 (uniform)
3. L-C1-d6-n5-p1-a11_1_0_0_0-tol1e-06 (boundary)
4. L-C1-d6-n5-p4-a11_1_0_0_0-tol1e-06 (boundary)
5. L-C1-d6-n5-p1-a6_4_2_0_0-tol1e-06 (mixed)
6. L-C1-d6-n5-p4-a6_4_2_0_0-tol1e-06 (mixed)

## JSONL Schema Parity Check

Reference artifacts:
- MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_uniform.jsonl
- MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_boundary.jsonl
- MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_mixed.jsonl

Observed top-level keys (consistent across all three files):
- anomaly_flags
- bounds
- commit_hash
- decomposition_diagnostics
- family_generation_error
- id
- method
- notes
- objective
- polynomial_family_descriptor
- runtime_sec
- solver_attempts
- solver_config
- status
- timestamp
- tolerance_setting

Parity conclusion:
- E1 pilot should retain this exact schema and keep template/p metadata in polynomial_family_descriptor as currently implemented.

## Gate Outcome for Session 1

- [x] At least one stronger backend is runnable.
- [ ] Stronger backends unavailable in current environment (MOSEK, SDPA-GMP, DSDP).

Immediate next action:
- Start Session 2 E1 pilot using DSDP in solver sequence with PATH prefixed while keeping CSDP in fallback chain:
  - `export PATH=/home/mehdi/miniforge3/envs/sage/bin:$PATH`
- Keep MOSEK as unavailable and keep SDPA-GMP as unavailable for current workflow unless a dedicated binary/source build path is provided.
