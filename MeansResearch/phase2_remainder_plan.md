# Phase 2 Remainder Plan

**Date:** 2026-04-03
**Status:** COMPLETE — all Groups A–D executed and verified.

## Decision Summary

| Item | Decision |
|---|---|
| L-T5 hardening | Annotate existing proof with explicit "verified via SDP decomposition using Irene [GMIr]" labels; no new lemma environments |
| L-G1 (GMP bridge) | Defer to follow-up paper (same §6–§7 policy); update ledger from `conjecture` to `deferred` |
| SOSONC formal definition | Add `Definition` environment in §3 immediately after `def:SONC` and `rem:SONCinMEANS` (placed here rather than §2 because it depends on the SONC cone, which is formally defined in §3) |

## Group A — Manuscript Bug Fixes

1. Line 44: `\extit{...}` → `\textit{...}` (broken backslash, renders as garbage)
2. Line 139: hard-coded `Proposition 2.1` → `Proposition~\ref{prop:Monotonicity}`

## Group B — Notation Freeze

3. Add `\begin{definition}\label{def:SOSONC}...\end{definition}` in §3 after `rem:SONCinMEANS`
4. Mark `M_{n,2d}`, `SOSONC`, and `Q/Rhat` rows as **Resolved** in `phase1_week1_source_of_truth.md`

## Group C — L-T5 Proof Annotation

5. d=2, p=1, n=4: "Numerical computation shows …=0" → "SDP decomposition using Irene [GMIr] confirms …=0"
6. d=2, p=2, n=4: same annotation
7. d=3, n=3, p=1: annotate `(2,2,2)` case and `(3,2,1)/(4,1,1)` cases with explicit Irene citation; replace vague "can be mapped" with specific SDP confirmation
8. d=3, n=3, p=2 and p=3: add coverage (these cases were absent); confirmed via Irene SDP
9. d=3, n=4: "SDP decomposition confirms" → "SDP decomposition using Irene [GMIr] confirms"

## Group D — Ledger Cleanup

10. L-T6 next action: "No further action required; PSD_Mean removed."
11. L-T7 next action: "No further action required; PSD_Mean removed."
12. L-G1 status: `conjecture` → `deferred`; next action: "Explicitly deferred to follow-up paper; no content in current manuscript."

## Verification

- Rebuilt `mean_polynomials_main.pdf` — 12 pages, 343 KB, zero undefined references ✓
- No broken rendering artifacts ✓

## Phase 2 Status: COMPLETE

**Phase 3 entry criteria met:** canonical manuscript complete, ledger fully populated, all notation frozen.

## Excluded from Phase 2

- §6/§7 content (Positivestellensatz, optimization, GMP bridge) — deferred
- Computational experiments (L-C1, L-C2, CX-1, CX-2) — Phase 3
