# OP2 Prototype Validation Summary (phi/support v1)

Date: 2026-04-26

## Frozen Slice Used

- Source: `MeansResearch/results/phase3_classification_table.csv`
- Filter: d in {4,5}, nondegenerate rows, SDP=success, GP=success, SONC in {success, fail}
- Evaluated records: 48

## Prototype-1 Descriptor Set

- `phi_degenerate`: support class is degenerate
- `phi_failure_band_p12`: p in {1,2}
- `phi_simplex_like`: support class is simplex-like
- `phi_template_family`: one of uniform/boundary/mixed
- `phi_boundary_p`: p in {0,d}

Prediction rule (v1):
- if degenerate -> U
- else if p in {1,2} -> R
- else if simplex-like and p in {0,d} -> F
- else -> U

## Predicted vs Observed (Aggregate)

- Observed counts: F=24, R=24, U=0
- Predicted counts: F=24, R=24, U=0
- Accuracy: 48/48 = 1.000
- Degree-only baseline accuracy: 24/48 = 0.500

Confusion (observed -> predicted):
- F->F: 24
- F->R: 0
- F->U: 0
- R->F: 0
- R->R: 24
- R->U: 0
- U->F: 0
- U->R: 0
- U->U: 0

## Gate Evaluation (G1-G3)

- G1 (nontrivial separation vs degree-only baseline): PASS
  - Criterion: accuracy improves over degree-only and predicts both F and R classes.
  - Result: 1.000 vs baseline 0.500; predicted F=24, R=24.
- G2 (robustness consistency on p in {1,2} failures): PASS
  - Criterion: predicted R aligns with observed robust-failure family on tracked p in {1,2} slice.
  - Result on frozen table: predicted R 24/24; observed R 24/24.
  - External diagnostics check: d=4 has 24/24 failures under each of 4 configs; d=5 has 24/24 under each of 4 configs (zero recoveries).
  - Robust failure cases represented in diagnostics summaries: 48.
- G3 (reproducibility): PASS
  - Criterion: rerun with identical inputs reproduces identical class assignments.
  - Result: deterministic rerun equality = True.

## Artifacts

- `MeansResearch/results/op2_prototype_classification_table.csv`
- `MeansResearch/results/op2_prototype_validation_summary.md`
