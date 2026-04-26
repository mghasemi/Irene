# d=6, n=5 L-C1 Escalation Pass2 Command Set

Goal: rerun unresolved nondegenerate pass1 cases at 1200s per solver (`cvxopt,sdpa,csdp`) while excluding structurally degenerate sparse slices.

## Uniform template (offsets 0-5)

```bash
/home/mehdi/Code/Irene/.venv/bin/python scripts/phase3_benchmarks.py \
  --items L-C1 --degrees 6 --variables 5 --tolerances 1e-6 \
  --sdp-solver-seq cvxopt,sdpa,csdp --solve-timeout 1200 \
  --case-offset 0 --max-cases 6 \
  --output MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_uniform.jsonl
```

## Boundary template (offsets 12-17)

```bash
/home/mehdi/Code/Irene/.venv/bin/python scripts/phase3_benchmarks.py \
  --items L-C1 --degrees 6 --variables 5 --tolerances 1e-6 \
  --sdp-solver-seq cvxopt,sdpa,csdp --solve-timeout 1200 \
  --case-offset 12 --max-cases 6 \
  --output MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_boundary.jsonl
```

## Mixed template (offsets 18-23)

```bash
/home/mehdi/Code/Irene/.venv/bin/python scripts/phase3_benchmarks.py \
  --items L-C1 --degrees 6 --variables 5 --tolerances 1e-6 \
  --sdp-solver-seq cvxopt,sdpa,csdp --solve-timeout 1200 \
  --case-offset 18 --max-cases 6 \
  --output MeansResearch/results/phase3_runs_clean_d6_lc1_escalation_pass2_mixed.jsonl
```
