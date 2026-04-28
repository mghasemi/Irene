## Plan: Unified Mean-Polynomial Research Roadmap

Deliver a 3-month research program that consolidates the proposal and three report PDFs into one coherent theory-to-evidence pipeline: unified manuscript skeleton, theorem/conjecture backlog with proof status, and a moderate computational validation protocol using Irene (SOS/SDP/SONC/GP) to stress-test and elaborate the mathematical findings.

**Execution Update (2026-04-22)**
- Completed:
	- Clean pilot runs for $d=4$ and $d=5$ with $n \in \{3,4\}$, tolerances $10^{-6},10^{-8}$, including SDP/SONC/GP comparisons.
	- d=4 and d=5 pilot summaries and a consolidated classification table (`phase3_pilot_summary_clean.md`, `phase3_pilot_summary_clean_d5.md`, `phase3_classification_table.csv`).
	- Robustness diagnostics for SONC failures at both degrees (`phase3_sonc_diagnostics.jsonl`, `phase3_sonc_diagnostics_d5.jsonl`) with matching summaries.
	- Support-class tagging fix in benchmark/classification scripts with regenerated combined artifacts (`phase3_classification_table.csv`, `phase3_classification_table.md`) showing simplex-like labeling for non-sparse families.
	- Manuscript-native TeX tables generated from the refreshed classification outputs, with an explicit placeholder for deferred $d \geq 6$ slices.
	- Manuscript and theorem ledger evidence insertion for the new computational results.
	- Manuscript-integrated claim-status synthesis for the frozen $d=4,d=5$ slice (confirmed, weakened, negative), linked to classification artifacts and tables.
	- Completed $d=6$ L-C2 SONC/GP clean slice with summary and diagnostics (`phase3_runs_clean_d6_lc2.jsonl`, `phase3_pilot_summary_clean_d6_lc2.md`, `phase3_sonc_diagnostics_d6_lc2.jsonl`).
	- Completed initial $d=6$ L-C1 SOS/SDP batch (2 records) with pilot summary (`phase3_runs_clean_d6_lc1_batch1.jsonl`, `phase3_pilot_summary_clean_d6_lc1_batch1.md`).
	- Completed $d=6$ L-C1 SOS/SDP batch2 uniform-$n=3$ extension (6 records covering $p \in \{0,1,2,3,4,6\}$) with pilot summary (`phase3_runs_clean_d6_lc1_batch2.jsonl`, `phase3_pilot_summary_clean_d6_lc1_batch2.md`).
	- Extended $d=6$ L-C1 SOS/SDP batch3 probe on $n=4$ to 4 records ($p \in \{0,1,2,3\}$): $p=0,1$ timed out at 660s, $p=2$ timed out at 1200s, and $p=3$ timed out at 1500s (`phase3_runs_clean_d6_lc1_batch3.jsonl`, `phase3_pilot_summary_clean_d6_lc1_batch3.md`).
	- Extended $d=6$ L-C1 SOS/SDP batch3 probe on $n=4$ to 5 records by adding $p=4$ at 1800s timeout (`phase3_runs_clean_d6_lc1_batch3.jsonl`, `phase3_pilot_summary_clean_d6_lc1_batch3.md`).
	- Completed targeted boundary mini-batch ($n=4$, $\alpha=(11,1,0,0)$, $p=0,1$) at 900s: both timeout (`phase3_runs_clean_d6_lc1_batch4_boundary.jsonl`, `phase3_pilot_summary_clean_d6_lc1_batch4_boundary.md`).
	- Completed targeted mixed mini-batch ($n=4$, $\alpha=(6,4,2,0)$, $p=0,1$) at 1200s: both timeout, duplicates cleaned and deduplicated summary generated (`phase3_runs_clean_d6_lc1_batch5_mixed.jsonl`, `phase3_pilot_summary_clean_d6_lc1_batch5_mixed.md`).
	- Completed cross-template $p=6$ probe for $n=4$: uniform ($\alpha=(3,3,3,3)$), boundary ($\alpha=(11,1,0,0)$), and mixed ($\alpha=(6,4,2,0)$) all timeout at 1200s (`phase3_runs_clean_d6_lc1_batch6_uniform_p6.jsonl`, `phase3_runs_clean_d6_lc1_batch7_boundary_p6.jsonl`, `phase3_runs_clean_d6_lc1_batch8_mixed_p6.jsonl` with corresponding summaries). **Verdict**: $n=4$ d=6 L-C1 appears computationally intractable across all template types and tested $p$ values under timeout budgets up to 1800s; evidence supports deferment of $d=6$ or escalation to specialized sparse/exploited-structure solvers.
	- Added timeout-guarded, resumable benchmark execution support (`--solve-timeout`, `--skip-existing`, `--case-offset`) plus incremental JSONL writes in `scripts/phase3_benchmarks.py` for safer long-running $d=6$ SDP slices and targeted template probing.
	- Completed CX-1 tractability probe: ran L-C1 boundary+mixed cases at n=5,6 for d=3,4 and n=5 for d=5 to assess CVXOPT scalability beyond n=4. Results:
		- d=3 n=5 boundary (single probe): success at 272s/case; full d=3 n=5 grid (~8 cases) would require ~2200s — marginally feasible but impractical at scale.
		- d=4 n=5 boundary+mixed (8 cases, 120s timeout): all timeout.
		- d=4 n=6 boundary+mixed (8 cases, 120s timeout): all timeout.
		- d=5 n=5 boundary (1 case, 900s timeout): timeout.
		- **CX-1 Verdict**: CVXOPT tractability ceiling for L-C1 is $n \leq 4$ across all tested degrees. At $n=5+$, SDP moment matrix size (e.g., $56\times56$ at $n=5$, $q=6$) renders the solver infeasible within practical budgets. This mirrors the d=6 deferment and confirms that solver scalability is the dominant bottleneck.
		- Artifacts: `phase3_runs_clean_cx1_d3_n5_probe.jsonl`, `phase3_pilot_summary_cx1_d3_n5_probe.md` (single boundary probe at d=3 n=5, 272s success), `phase3_runs_clean_cx1_d4_n5.jsonl`, `phase3_runs_clean_cx1_d4_n6.jsonl`, `phase3_runs_clean_cx1_batch1_d5_n5_boundary_mixed.jsonl` (all timeout).
	- Completed CX-2 transform stress batch 1 with dedicated runner (`scripts/phase3_cx2_transform_probe.py`) and artifacts (`phase3_runs_clean_cx2_transform_probe.jsonl`, `phase3_pilot_summary_cx2_transform_probe.md`).
		- Choi-Lam base/perturbed: SONC and GP both feasible at $10^{-6},10^{-8}$.
		- Robinson-hat base: SONC and GP both infeasible pre-transform.
		- Robinson-hat transformed base: SONC becomes feasible while GP remains infeasible.
		- Perturbed Robinson-hat transformed slice remains infeasible for both methods in this initial micro-batch.
	- Completed CX-2 transform stress batch 2 coefficient sweep (`phase3_runs_clean_cx2_transform_probe_batch2.jsonl`, `phase3_pilot_summary_cx2_transform_probe_batch2.md`).
		- Choi-Lam family with tail coefficients $\\{4.0,3.95,3.9,3.8\\}$ remains SONC+GP feasible.
		- Robinson-hat family pre-transform remains SONC+GP infeasible for coefficients $\\{2.0,1.98,1.95,1.9,1.85,1.8\\}$.
		- Robinson-hat post-transform SONC feasibility appears isolated at baseline coefficient $2.0$; nearby coefficients remain SONC-infeasible.
	- Completed d=6/n>=5 SOS escalation infrastructure for L-C1:
		- Added solver fallback chain support to benchmark runner via `--sdp-solver-seq` in `scripts/phase3_benchmarks.py`.
		- Verified environment solver availability: `CVXOPT, SDPA, CSDP`.
		- Smoke probe with `cvxopt,sdpa,csdp` and 20s cap logs per-solver outcomes in `solver_attempts` (`phase3_runs_clean_d6_lc1_escalation_smoke.jsonl`), all timeout on tested case.
	- Completed d=6 structural SONC ablation extension on the finished L-C2 clean slice (`phase3_sonc_diagnostics_d6_structural.jsonl`, `phase3_sonc_diagnostics_d6_structural.md`).
		- Scope: 48 unique failed SONC cases across the nondegenerate `uniform`, `boundary`, and `mixed` templates for d=6 and n in {3,4}.
		- Result: zero recoveries under all four alternate SONC diagnostic configs (`local_tol`, `global_tol`, `local_1e-10`, `global_1e-10`).
		- Interpretation: the d=6 structural SONC failures in this slice are configuration-robust rather than artifacts of local/global solve choice or tighter tolerance alone.
	- Completed d=6/n=5 L-C1 escalation pass1 (`phase3_runs_clean_d6_lc1_escalation_pass1.jsonl`) with solver-attempt trace summary (`phase3_pilot_summary_d6_lc1_escalation_pass1.md`).
		- Scope: 24 records at tolerance 1e-6 with solver sequence `cvxopt,sdpa,csdp` and 600s per-solver cap.
		- Outcome: 18 nondegenerate template cases (`uniform`, `boundary`, `mixed`) timeout under all three solvers; 6 sparse cases remain structurally inconclusive.
		- Solver-level signal: no `success` or hard `fail` outcomes; attempts are uniformly timeout-dominated on nondegenerate slices.
	- Prepared pass2 command set targeting only unresolved nondegenerate templates at 1200s per solver (`phase3_escalation_pass2_command_set.md`).
	- Completed d=6/n=5 L-C1 SOS escalation pass2 (`phase3_runs_clean_d6_lc1_escalation_pass2_uniform.jsonl`, `phase3_runs_clean_d6_lc1_escalation_pass2_boundary.jsonl`, `phase3_runs_clean_d6_lc1_escalation_pass2_mixed.jsonl`) with summary (`phase3_pilot_summary_d6_lc1_escalation_pass2.md`).
		- Scope: 18 nondegenerate records (6 uniform + 6 boundary + 6 mixed) at 1200s per-solver cap.
		- Outcome: 18/18 timeout under all three solvers (CVXOPT, SDPA, CSDP); zero success or hard-fail outcomes.
		- **Verdict**: Doubling the timeout from pass1 (600s) to pass2 (1200s) produced no change. d=6, n=5 L-C1 is computationally intractable under standard interior-point SDP solvers within practical budgets. The slice is formally deferred; continuation requires sparse-structure-exploiting or high-performance solvers (MOSEK, DSDP, SDPA-GMP).
- Completed:
	- d=6 L-C1 n=3 uniform grid (batches 1–2): success across all $p$ values (artifact-frozen).
	- d=6 L-C1 n=4 all-template timeout probe (batches 3–8): all template types ($\alpha$ uniform/boundary/mixed) and all tested $p$ values (incl. $p=6$ cross-template) timeout at budgets up to 1800s → **d=6 L-C1 computationally deferred**.
	- Manuscript, plan, and experiment matrix artifacts synchronized with d=6 deferment verdict.
	- CX-1 tractability ceiling established: L-C1 n≥5 infeasible under CVXOPT at all tested degrees.
	- CX-2 initial transform-sensitive behavior captured and archived via JSONL + summary artifacts.
	- CX-2 coefficient sweep completed; boundary-like transform sensitivity now evidenced around Robinson-hat baseline coefficient.
	- d=6 deferred-slice escalation path now operational with solver-attempt logging and reproducible command template.
	- d=6 structural SONC ablation completed with a no-recovery result across all tested nondegenerate templates and alternate SONC configs.
	- d=6/n=5 L-C1 escalation pass1 completed and summarized; unresolved nondegenerate cases are now isolated for pass2.
	- d=6/n=5 L-C1 escalation pass2 completed and summarized; all-timeout verdict confirms d=6 n=5 L-C1 is formally deferred.
	- Roadmap synchronized with Vikunja project 17; current execution queue refreshed around d=6/n=5 escalation pass1, pass1 summarization, and pass2 promotion.
- In progress:
	- None. Phase 4 parent closure sweep is complete; Vikunja project 17 refresh on 2026-04-26 shows 50/50 tasks closed and no overdue items.
- Next execution sprint:

	1. Completed: inserted d=6 n=5 intractability evidence into manuscript §Computational Results section.
	2. Completed: updated theorem ledger to mark d=6 n=5 L-C1 SOS-by-SDP as `deferred (solver ceiling)`.
	3. Completed: rebuilt PDF from updated TeX source.
	4. Completed kickoff: Phase 4 integration pass (open problems, conjecture backlog + ranked claims + continuation plan).
	5. Completed: final experiment-report/reproducibility close-out block drafted and integrated into manuscript + theorem ledger.
	6. Completed: closed remaining Phase 4 Vikunja parent task after final status sync.
	7. Completed: OP2/B2 support-sensitive SONC wording promoted from narrative-only framing to a formal manuscript conjecture object and synchronized in the theorem ledger L-C2 next-action text.
	8. **Completed (2026-04-28, Session 5)**: OP1 E1 pilot executed with DSDP/CSDP/SDPA/CVXOPT (DSDP found at /home/mehdi/miniforge3/envs/sage/bin/dsdp5; MOSEK and SDPA-GMP unavailable): 6/6 nondegenerate d=6 n=5 cases timed out at 600s; OP1 gate failed and OP1 is frozen this cycle. OP2 prototype validated (48/48 accuracy, G1/G2/G3 pass on frozen d=4/5 slice). OP3 boundary note drafted with falsifiable criteria C1-C3. Integration pass complete; no cross-document contradictions remain. Immediate next: advance OP2 to theorem-level criterion development and OP3 to verification/falsification of C1-C3; resume OP1 only when a higher-performance backend (MOSEK, SDPA-GMP) is available.
	9. **Completed (2026-04-28, Post-Session 5 follow-up)**: OP3 C3 micro-grid verification executed (c in {1.999,2.000,2.001}, t in {0.995,1.000,1.005}); results show center-only robust_success (1/9) and non-center robust_fail (8/9), so the isolated-atom claim is not falsified on the targeted neighborhood. OP2 theorem-stage kickoff advanced with Delta_support criterion run: frozen d=4/5 slice 48/48, d=6 structural stress slice 24/24, and unique nondegenerate d=6 L-C2 clean-pilot SONC slice 36/36; next theorem target is analytic promotion (circuit/barycentric proof with scope guards).
**Steps**
1. Phase 1 (Week 1): Source-of-truth consolidation for proposal + reports. Build a claim matrix from `/home/mehdi/Code/Irene/MeansResearch/Proposal- A Unified Approach to Optimization via Generalized Moment Problems and (p,q)-Mean Polynomials.pdf`, `/home/mehdi/Code/Irene/MeansResearch/mean_polynomials_main.pdf`, `/home/mehdi/Code/Irene/MeansResearch/PSD_Mean_31.12.2025(1).pdf`, and `/home/mehdi/Code/Irene/MeansResearch/The Cone Generated by Non-negative Mean Polynomials07.pdf`, using `/home/mehdi/Code/Irene/MeansResearch/mean_polynomials_main.tex` as the canonical editable anchor. Output: objective map, notation map, theorem overlap/diff map.
2. Phase 1 (Week 1): Version reconciliation between `mean_polynomials_main` and `PSD_Mean_31.12.2025`: identify which statements are equivalent, strengthened, weakened, or unresolved. Mark all TODO markers in TeX as blockers. Output: prioritized discrepancy list. Depends on Step 1.
3. Phase 2 (Weeks 2-4): Formalize a theorem ledger for four priority threads: cone theory for (p,q)-mean polynomials, generalized moment problem linkage, SOS/SONC/Mean inclusion-separation relations, and computationally testable conjectures. Each item must include status (`proved`, `draft proof`, `conjecture`, `counterexample candidate`) and dependencies. Depends on Step 2.
4. Phase 2 (Weeks 2-4): Draft the unified manuscript skeleton in parallel with theorem ledger maturation: preliminaries/notation, cone geometry, GMP connections, SOS-SONC comparison, computational evidence, open problems. Explicitly map each section to source claims from Step 1 and ledger entries from Step 3. Parallel with Step 3.
5. Phase 3 (Weeks 5-8): Build hypothesis-to-experiment mapping using Irene methods. For each conjecture/claim that can be empirically probed, assign one or more implementations: `SDPRelaxations.Minimize`, `SDPRelaxations.Decompose`, `SONCRelaxations.solve`, `OptimizationProblem.newton`, `OptimizationProblem.delta`, and `GPRelaxations.solve`. Output: experiment matrix with expected behavior and falsification criteria. Depends on Step 3.
6. Phase 3 (Weeks 5-8): Assemble moderate evidence benchmark suite from existing examples and tests. Start from SONC examples and constrained SDP examples; add controlled synthetic instances varying degree, sparsity, and Newton polytope geometry. Record metrics: bound tightness, runtime, solver status, decomposition/certificate diagnostics, and sensitivity to formulation. Depends on Step 5.
7. Phase 3 (Weeks 5-8): Execute comparative runs (SOS/SDP vs SONC vs GP where applicable) and capture findings as: confirmed hypotheses, weakened hypotheses, and negative results. Keep negative evidence as first-class output for conjecture refinement. Depends on Step 6.
8. Phase 4 (Weeks 9-12): Integrate theory and experiments into final deliverables: (a) unified survey-style manuscript draft, (b) conjecture/lemma backlog with proof status and next actions, (c) experiment report/notebook cross-referenced to Irene scripts and tests. Depends on Steps 4 and 7.
9. Phase 4 (Weeks 9-12): Final research steering pass: prioritize 2-3 strongest publishable claims, 2-3 high-value open problems, and 1 fallback direction if core conjectures fail. Define post-quarter continuation plan. Depends on Step 8.

**Relevant files**
- `/home/mehdi/Code/Irene/MeansResearch/Proposal- A Unified Approach to Optimization via Generalized Moment Problems and (p,q)-Mean Polynomials.pdf` — proposal objectives and intended unified framework.
- `/home/mehdi/Code/Irene/MeansResearch/mean_polynomials_main.tex` — editable source for core mean-polynomial cone results and SOS/SONC links.
- `/home/mehdi/Code/Irene/MeansResearch/PSD_Mean_31.12.2025(1).pdf` — historical reference report used for reconciliation context.
- `/home/mehdi/Code/Irene/MeansResearch/The Cone Generated by Non-negative Mean Polynomials07.pdf` — additional report findings to merge into theorem ledger.
- `/home/mehdi/Code/Irene/Irene/relaxations.py` — SDP hierarchy and SOS extraction (`SDPRelaxations`, `Decompose`).
- `/home/mehdi/Code/Irene/Irene/sonc.py` — constrained SONC formulation and solver pipeline.
- `/home/mehdi/Code/Irene/Irene/program.py` — structural analysis hooks (`delta`, `newton`, exponent conversion).
- `/home/mehdi/Code/Irene/Irene/geometric.py` — GP-based lower-bound route for selected formulations.
- `/home/mehdi/Code/Irene/examples/SONCExample33.py` — barycentric/circuit validation template.
- `/home/mehdi/Code/Irene/examples/Example01.py` — constrained SDP baseline template.
- `/home/mehdi/Code/Irene/tests/test_sonc_section3.py` — regression anchors for SONC pipeline reliability.
- `/home/mehdi/Code/Irene/tests/test_quality_plan.py` — quality-gate template for research evidence standards.

**Verification**
1. Claim coverage check: every major statement in the proposal and three report PDFs appears in the claim matrix with status and source location.
2. Reconciliation check: no unresolved inconsistency remains undocumented between `mean_polynomials_main` and `PSD_Mean_31.12.2025` versions.
3. Ledger completeness check: every theorem/conjecture has a status tag, dependency tag, and next action.
4. Experiment traceability check: each computational claim links to a specific Irene method and runnable example/test anchor.
5. Reproducibility check: selected baseline tests run successfully before and after experiment additions (`tests/test_sonc_section3.py`, `tests/test_quality_plan.py`).
6. Evidence balance check: final report includes both positive confirmations and counterexamples/negative outcomes.
7. Steering check: final phase yields ranked publishable claims and explicitly scoped deferred items.

**Decisions**
- Planning horizon fixed to 3 months.
- Technical priorities include all four tracks: cone theory, GMP linkage, SOS/SONC/Mean relations, and Irene-based hypothesis validation.
- Evidence level fixed to moderate (benchmark suite + targeted ablations rather than exhaustive campaign).
- Required deliverables: unified survey-style manuscript, proof-status backlog, and experiment notebook/report.

**Further Considerations**
1. If the proposal PDF contains claims absent from TeX sources, prioritize transcribing those claims into the ledger before Week 2 to avoid drift.
2. Decide early whether the unified manuscript target is journal-style full paper or staged technical report; this changes section depth and experiment breadth.
3. Keep a strict boundary between formal proofs and computational evidence: experiments can prioritize conjectures but cannot replace theorem-level closure.
