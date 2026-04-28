# OP2 Prototype Plan: Support-Sensitive SONC Criterion

Date: 2026-04-26  
Scope: Prototype a support-sensitive criterion that explains persistent SONC failures on p in {1,2} slices while GP remains feasible on the same tested nondegenerate families.

## 1) Prototype Objective

Construct a criterion template that predicts SONC behavior from support geometry descriptors rather than degree alone, and evaluate whether it separates:
- SONC-feasible slices, and
- robust SONC-failure slices (especially p in {1,2})
within the current tested families.

The prototype is explanatory and computationally falsifiable; it is not yet a theorem-level characterization.

## 2) Baseline Evidence for Prototype Design

Observed pattern to explain:
- On tested nondegenerate d in {4,5}, n in {3,4} grids, GP is broadly feasible while SONC is support-sensitive.
- SONC failures on p in {1,2} are robust under local/global solver modes and tightened tolerances.
- d=6 structural SONC diagnostics also show configuration-robust failure persistence for tracked failed slices.

Primary evidence artifacts:
- MeansResearch/results/phase3_pilot_summary_clean.md
- MeansResearch/results/phase3_pilot_summary_clean_d5.md
- MeansResearch/results/phase3_sonc_diagnostics_summary.md
- MeansResearch/results/phase3_sonc_diagnostics_summary_d5.md
- MeansResearch/results/phase3_sonc_diagnostics_d6_structural.md

## 3) Criterion Prototype Template

Define a score or rule set over support descriptors phi(support) and classify each instance into:
- class F: SONC-feasible expected,
- class R: robust SONC-failure expected,
- class U: undecided/near-boundary.

Candidate descriptor blocks for phi(support):
- simplex-cover compatibility of negative-term supports,
- barycentric deficit of candidate circuit representations,
- overlap and multiplicity structure among tail supports,
- distance-to-circuit envelope under fixed monomial basis,
- optional transform sensitivity tag for known boundary families.

Prototype form options:
- weighted score threshold,
- decision-list rule set,
- two-stage rule: hard exclusion tests then soft score.

## 4) Minimal Data Protocol

Unit of evaluation:
- one benchmark instance with fixed degree, n, p, template family, and support metadata.

Required labels:
- SONC outcome class under diagnostic protocol (success or robust failure),
- GP feasibility flag on same instance,
- robustness flag across local/global and tolerance diagnostics.

Split policy:
- Development slice: d in {4,5}, n in {3,4} nondegenerate cases.
- Stress slice: d=6 structural SONC diagnostics as out-of-slice robustness check.

## 5) Validation Plan

Validation V1 (separation quality on development slice):
- Measure how well prototype separates SONC-feasible vs robust SONC-failure groups.
- Report confusion table and per-template breakdown.

Validation V2 (robustness to diagnostic configuration):
- Confirm predicted robust-failure class aligns with local/global and tightened-tolerance persistence.

Validation V3 (cross-slice sanity check):
- Apply prototype to d=6 structural diagnostic artifacts and report agreement/disagreement patterns.

## 6) Acceptance Gates for OP2 Progression

Gate G1 (minimum viability):
- Prototype must recover a nontrivial separation signal on development slice (better than degree-only baseline).

Gate G2 (robustness consistency):
- Predicted robust-failure class must align with observed diagnostic-persistent failures in the majority of tracked p in {1,2} cases.

Gate G3 (artifact reproducibility):
- Re-running prototype evaluation with same inputs must reproduce identical classification outputs.

Failure rule:
- If G1 fails, keep OP2 open and revise descriptor set before any manuscript-level promotion.

## 7) Deliverables and Artifact-First Outputs

Required deliverables for OP2 prototype cycle:
- One prototype specification note (this file).
- One results table artifact summarizing predicted vs observed classes by template and p.
- One short interpretation memo stating what the prototype explains and where it fails.

Suggested output filenames:
- MeansResearch/results/op2_prototype_classification_table.csv
- MeansResearch/results/op2_prototype_validation_summary.md

## 8) Manuscript and Ledger Policy While OP2 Is Open

- Keep language as support-sensitive conjectural program, not theorem-level criterion.
- Continue separating confirmed empirical behavior from explanatory prototype claims.
- Treat prototype outputs as evidence-weighting tools until formal proof dependencies are established.

## 9) Immediate Next Action

Completed on 2026-04-28:
- Instantiated first phi(support) descriptor set on frozen d in {4,5} nondegenerate slice.
- Produced predicted vs observed class table and validation summary.
- Evaluated G1-G3 with PASS/PASS/PASS.

## 10) 2026-04-28 Theorem-Stage Kickoff Update

Prototype status transition:
- OP2 moves from prototype-instantiation to theorem-stage criterion development.

Cross-slice stress check (d=6 structural failures):
- Source: MeansResearch/results/phase3_sonc_diagnostics_d6_structural.jsonl
- Unique cases: 24 (all robust SONC failures across p in {1,2,3,4} and template families uniform/boundary/mixed)
- Delta_support criterion run (2026-04-28):
	- definition: Delta_support(case) = min(p, d-p) / d
	- prediction rule: Delta_support > 0 -> R, Delta_support = 0 -> F
	- frozen d=4/5 slice: 48/48 matches (1.000)
	- d=6 structural stress slice: 24/24 matches (1.000)
	- d=6 L-C2 clean pilot (unique nondegenerate SONC cases): 36/36 matches (1.000)
	- artifacts: MeansResearch/results/op2_delta_support_table.csv, MeansResearch/results/op2_delta_support_summary.json

Interpretation:
- A geometry-first interiority margin now explains both frozen and stress slices empirically.
- The remaining theorem-stage task is to justify this margin analytically (circuit/barycentric argument) and identify explicit scope conditions.

## 11) Immediate Next Action (Updated)

1. Attempt a proof skeleton for uniform template families using Delta_support as an interiority/circuit-margin surrogate.
2. Extend the same argument to boundary and mixed templates, recording any additional assumptions.
3. Completed (2026-04-28): scope check on nondegenerate d=6 L-C2 clean-pilot rows beyond stress subset (36/36 at unique-case level).
4. If analytic promotion fails, isolate the minimal counterexample family where Delta_support and SONC status diverge.

## 12) Immediate Attention (Formalized Statement for Sign-off)

Proposed manuscript-level OP2 statement:

For executed nondegenerate slices of $M_{2d,p}(X,\alpha)$, define
\(\Delta_{\mathrm{support}}(d,p)=\min\{p,d-p\}/d\).
Then empirical SONC status aligns with this criterion:
- \(\Delta_{\mathrm{support}}=0\) -> SONC-feasible,
- \(\Delta_{\mathrm{support}}>0\) -> robust SONC-infeasible
under executed local/global and tightened-tolerance diagnostics.

Executed-slice evidence for this statement:
- frozen d=4/5 clean slice: 48/48
- d=6 structural stress slice: 24/24
- unique nondegenerate d=6 L-C2 clean-pilot SONC slice: 36/36

Author decision (2026-04-28):
- Approved.
- The Delta-support statement is now the primary B2 formal conjecture wording in manuscript and theorem ledger, with explicit slice-local scope caveat.

Immediate execution after approval:
1. Start uniform-family proof skeleton using Delta_support as the interiority proxy.
2. Record the first dependency list for analytic promotion (circuit margin identity, barycentric admissibility conditions, and any required nondegeneracy assumptions).

## 13) Uniform-Family Proof Skeleton (First Draft)

Target statement (uniform template):
- For nondegenerate uniform-template families in the executed slices, SONC status follows
  the sign of \(\Delta_{\mathrm{support}}(d,p)=\min\{p,d-p\}/d\).

Proof-outline structure:
1. Setup and normalization:
	- Fix \(d\), \(n\), and a uniform-template \(\alpha\) with \(|\alpha|=2d\).
	- Write \(M_{2d,p}(X,\alpha)\) in coefficient-normalized form separating diagonal and tail blocks.
2. Boundary branch (\(\Delta_{\mathrm{support}}=0\), i.e., \(p\in\{0,d\}\)):
	- Show the tail support aligns with admissible circuit structure from Theorem~\ref{thm:Circuit_PSD}.
	- Verify circuit-number inequality in the nonnegative direction, yielding SONC-feasible status.
3. Interior branch (\(\Delta_{\mathrm{support}}>0\)):
	- Show the corresponding tail support induces a strict circuit-margin deficit in the executed slices.
	- Use this deficit to rule out SONC certificate existence in the tested formulation class.
4. Scope closure:
	- Explicitly state that this is currently slice-local (executed grids), pending symbolic closure for all families.

First dependency checklist (for implementation/proof notes):
- D-OP2-1 (circuit margin identity): explicit formula linking \(p\) and circuit margin sign for uniform template.
- D-OP2-2 (barycentric admissibility): characterization of when tail exponent remains in admissible convex hull with positive barycentric weights.
- D-OP2-3 (nondegeneracy guard): conditions excluding sparse/degenerate one-hot families from the theorem statement.
- D-OP2-4 (solver-to-structure bridge): argument that robust diagnostic failure pattern corresponds to structural margin sign in executed slices.

Execution-ready next substeps:
1. Derive candidate closed-form expression for D-OP2-1 on \(n=3\) uniform template, then lift to \(n=4\).
2. Cross-check D-OP2-2 against the barycentric condition used in Theorem~\ref{thm:Circuit_PSD}.
3. Map D-OP2-3 directly to existing `support_class != degenerate` filter used in artifacts.

## 14) D-OP2-1 Candidate Identity (Uniform Template, First Pass)

Status: drafted as a theorem-stage working identity for analytic promotion. This is
not yet a completed proof and remains scoped to the executed family definition.

Candidate normalized interiority margin:

\[
\mu_{\mathrm{unif}}(d,p) := \frac{\min\{p,d-p\}}{d} = \Delta_{\mathrm{support}}(d,p).
\]

Operational branch rule used by current evidence:
- boundary branch: \(\mu_{\mathrm{unif}}(d,p)=0\) (equivalently \(p\in\{0,d\}\))
- interior branch: \(\mu_{\mathrm{unif}}(d,p)>0\) (equivalently \(0<p<d\))

Candidate circuit-deficit proxy:

\[
\delta_{\mathrm{circ}}^{\star}(d,p) := \mu_{\mathrm{unif}}(d,p).
\]

Interpretation for theorem-stage work:
- \(\delta_{\mathrm{circ}}^{\star}=0\): admissible boundary circuit regime (observed SONC-feasible on executed slices).
- \(\delta_{\mathrm{circ}}^{\star}>0\): strict interior deficit proxy (observed robust SONC-failure on executed slices).

Minimal derivation note (uniform template):
1. In the current parametrization, the support-position variable is indexed by \(p\) with reflection symmetry \(p \leftrightarrow d-p\).
2. The nearest boundary distance on this index line is \(\min\{p,d-p\}\).
3. Normalizing by degree \(d\) yields a scale-free interiority score in \([0,1/2]\), giving \(\mu_{\mathrm{unif}}\).
4. The threshold \(\mu_{\mathrm{unif}}=0\) exactly identifies boundary cases \(p\in\{0,d\}\), while \(\mu_{\mathrm{unif}}>0\) identifies interior cases \(0<p<d\).

Immediate validation hooks for this identity:
1. Check monotonicity on the half-range \(0 \le p \le d/2\): \(\mu_{\mathrm{unif}}=(p/d)\), then apply reflection symmetry.
2. Verify branch agreement against `op2_delta_support_table.csv` for all uniform-template rows in frozen/stress/clean slices.
3. Attach the first symbolic bridge in manuscript notes by rewriting this proxy as a barycentric-distance surrogate (dependency D-OP2-2).

## 15) D-OP2-2 Barycentric Admissibility Bridge (Uniform First Pass)

Status: drafted as a theorem-stage bridge from the index-based interiority proxy
to circuit-style admissibility language. This is a scoped working note, not yet
a full formal proof.

Working setup:
- Let \(V=\{v_0,\ldots,v_m\}\) denote the outer (even-exponent) support vertices
  used by the circuit model in the uniform template slice.
- Let \(\beta(p)\) denote the tail exponent selected by index \(p\).
- Define barycentric coordinates by
  \[
  \beta(p)=\sum_{i=0}^{m}\lambda_i(p)\,v_i,\qquad \sum_{i=0}^{m}\lambda_i(p)=1.
  \]

Admissibility criterion (working form):
- strict interior admissibility: \(\lambda_i(p)>0\) for all \(i\),
- boundary admissibility: at least one \(\lambda_i(p)=0\), with the rest nonnegative.

Candidate bridge to Delta-support:
\[
\mu_{\mathrm{unif}}(d,p)=\frac{\min\{p,d-p\}}{d}=0
\iff
\beta(p)\in\partial\operatorname{conv}(V),
\]
\[
\mu_{\mathrm{unif}}(d,p)>0
\iff
\beta(p)\in\operatorname{relint}(\operatorname{conv}(V)).
\]

Equivalent barycentric slack proxy:
\[
s_{\min}(p):=\min_i \lambda_i(p),
\]
with working correspondence
\[
s_{\min}(p)=0 \Longleftrightarrow \mu_{\mathrm{unif}}(d,p)=0,
\qquad
s_{\min}(p)>0 \Longleftrightarrow \mu_{\mathrm{unif}}(d,p)>0.
\]

Interpretation for OP2 theorem-stage use:
1. Boundary branch (\(\mu_{\mathrm{unif}}=0\)): admissible boundary placement of the
	tail exponent, consistent with observed SONC-feasible rows.
2. Interior branch (\(\mu_{\mathrm{unif}}>0\)): strictly interior placement with
	positive barycentric slack, matched empirically with robust SONC-failure in executed slices.

What remains to close D-OP2-2:
1. Compute \(\lambda_i(p)\) explicitly for the concrete uniform-template support map
	used by the current implementation (first for \(n=3\), then \(n=4\)).
2. Prove (or delimit) the equivalence between \(s_{\min}\)-sign and
	\(\mu_{\mathrm{unif}}\)-sign for that concrete map.
3. Integrate this bridge with the circuit-number inequality dependency from
	Theorem~\ref{thm:Circuit_PSD} to determine whether interior admissibility implies
	certificate obstruction in the current formulation class.

## 16) D-OP2-2 Explicit \(n=3\) Barycentric Slice (Edge-Circuit Form)

Status: first explicit \(\lambda\)-formula pass completed for a scoped, implementation-aligned
uniform-template slice. This is an edge-circuit reduction used as a proof scaffold,
not yet a full characterization of all support points produced when \(p>0\).

Scoped construction:
- Degree parameter: \(q=2d\).
- Outer vertices (diagonal terms):
	\[
	v_1=(2d,0,0),\quad v_2=(0,2d,0).
	\]
- Edge tail exponent path indexed by \(p\in\{0,\ldots,d\}\):
	\[
	\beta_{\mathrm{edge}}(p)=(2(d-p),2p,0).
	\]

Barycentric decomposition on this edge:
\[
\beta_{\mathrm{edge}}(p)=\lambda_1(p) v_1+\lambda_2(p) v_2,
\quad
\lambda_1(p)=\frac{d-p}{d},\;\lambda_2(p)=\frac{p}{d},
\quad
\lambda_1+\lambda_2=1.
\]

Explicit slack identity:
\[
s_{\min}^{\mathrm{edge}}(p):=\min\{\lambda_1(p),\lambda_2(p)\}
=\frac{\min\{p,d-p\}}{d}
=\Delta_{\mathrm{support}}(d,p).
\]

Immediate consequences on the edge slice:
1. \(p\in\{0,d\}\Rightarrow s_{\min}^{\mathrm{edge}}=0\) (boundary of the edge segment).
2. \(0<p<d\Rightarrow s_{\min}^{\mathrm{edge}}>0\) (relative interior of the edge segment).
3. The branch threshold used by OP2 is recovered exactly on this slice:
	 \(\Delta_{\mathrm{support}}=0\) vs \(\Delta_{\mathrm{support}}>0\).

Scope note:
- For \(p>0\), the implemented family generator expands
	\(\big(\frac{1}{q}\sum_i \alpha_i x_i^p\big)^{q/p}\), producing many support points.
	The edge-circuit formulas above capture one analytically tractable slice of that
	support, used as the first symbolic bridge to Theorem~\ref{thm:Circuit_PSD}.

Next closure step from this explicit slice:
1. Lift from edge-circuit slice to a full-support admissibility statement for \(n=3\)
	 uniform-template runs (identify which support strata control SONC obstruction).
2. Repeat with the analogous \(n=4\) edge/simplex slices, then consolidate into D-OP2-3 guards.

## 17) D-OP2-2 Full-Support Lift (\(n=3\), Uniform Template)

Status: drafted as the first full-support lift note from edge-circuit formulas to the
implemented support expansion.

Implementation-aligned support description (for \(p>0\), \(p\mid 2d\)):
- Set \(m:=2d/p\).
- The second-term expansion support used by the benchmark generator is
	\[
	S_{d,p}^{(3)}=\{\gamma= p\,k : k\in\mathbb{N}_0^3,\; k_1+k_2+k_3=m\}.
	\]

Simplex coordinates and slack:
- Use outer simplex vertices \(v_i=2d\,e_i\) for \(i\in\{1,2,3\}\).
- Any \(\gamma\in S_{d,p}^{(3)}\) admits barycentric coordinates
	\[
	\lambda_i(\gamma)=\frac{\gamma_i}{2d}=\frac{k_i}{m},\qquad \sum_i\lambda_i=1.
	\]
- Define pointwise slack
	\[
	s_{\min}(\gamma):=\min_i \lambda_i(\gamma)=\frac{\min_i k_i}{m}.
	\]

Support-strata decomposition:
1. Vertex stratum \(\mathcal{V}\): two zero \(k_i\) values (simplex vertices), \(s_{\min}=0\).
2. Edge stratum \(\mathcal{E}\): exactly one zero \(k_i\), \(s_{\min}=0\).
3. Interior stratum \(\mathcal{I}\): all \(k_i>0\), \(s_{\min}>0\).

Working lift relation to OP2 branch split:
- Boundary cases \((p\in\{0,d\})\): no strict interior support stratum appears in the
	tested \(n=3\) uniform slice; observed SONC status is feasible.
- Interior cases \((0<p<d)\): strict interior support points appear in the tested
	\(n=3\) uniform slice; observed SONC status is robust-failure.

Theorem-stage interpretation (working, not final):
- The full-support lift target is to show that the presence of \(\mathcal{I}\) with
	positive slack is the structural trigger linked to SONC obstruction in the current
	formulation class, while \(\mathcal{V}\cup\mathcal{E}\) alone is compatible with feasibility.

Remaining proof obligations after this lift draft:
1. Convert the stratum statement above into a circuit-number inequality argument
	 tied to Theorem~\ref{thm:Circuit_PSD} (or identify the exact gap where this fails).
2. Replicate the same support-strata/slack analysis for \(n=4\) uniform templates.
3. Record explicit nondegeneracy guards (D-OP2-3) for sparse/one-hot and near-degenerate slices.

## 18) D-OP2-2 Full-Support Lift (\(n=4\), Uniform Template)

Status: first \(n=4\) lift drafted with explicit strata and a caveat that one slack
scalar alone may be insufficient for all interior \(p\) classes.

Implementation-aligned support description (for \(p>0\), \(p\mid 2d\)):
- Set \(m:=2d/p\).
- The second-term expansion support is
	\[
	S_{d,p}^{(4)}=\{\gamma= p\,k : k\in\mathbb{N}_0^4,\; k_1+k_2+k_3+k_4=m\}.
	\]

Simplex coordinates and slack:
- Outer simplex vertices: \(v_i=2d\,e_i\), \(i\in\{1,2,3,4\}\).
- For \(\gamma\in S_{d,p}^{(4)}\):
	\[
	\lambda_i(\gamma)=\frac{\gamma_i}{2d}=\frac{k_i}{m},\qquad \sum_i\lambda_i=1.
	\]
- Primary slack proxy:
	\[
	s_{\min}(\gamma):=\min_i \lambda_i(\gamma)=\frac{\min_i k_i}{m}.
	\]

Support-strata decomposition in the 3-simplex:
1. Vertex stratum \(\mathcal{V}\): three zero \(k_i\) values.
2. Edge stratum \(\mathcal{E}\): two zero \(k_i\) values.
3. Face-interior stratum \(\mathcal{F}\): exactly one zero \(k_i\), remaining positive.
4. Full-interior stratum \(\mathcal{I}_4\): all \(k_i>0\).

Structural observations for executed \(d=6\), \(n=4\), uniform slice:
- \(p=6\Rightarrow m=2\): only \(\mathcal{V}\cup\mathcal{E}\) are available.
- \(p=4\Rightarrow m=3\): \(\mathcal{F}\) is available but \(\mathcal{I}_4\) is not.
- \(p\in\{1,2,3\}\Rightarrow m\in\{12,6,4\}\): \(\mathcal{I}_4\) is available.

Implication for theorem-stage modeling:
- Because \(p=4\) is empirically robust-failure while \(\mathcal{I}_4\) is absent,
	the final obstruction criterion for \(n=4\) cannot rely only on "full-interior present".
- Candidate refinement: use a stratified slack profile (minimum and second-minimum
	barycentric coordinates, or active-support cardinality) rather than a single \(s_{\min}\).

Immediate \(n=4\) closure tasks:
1. Define a minimal stratified-slack functional that separates \(p=6\) (feasible in data)
	 from \(p=4\) (robust-failure in data) and \(p\in\{1,2,3\}\) (robust-failure).
2. Tie that functional to circuit-number obstruction arguments in the current SONC formulation class.

## 19) D-OP2-3 Nondegeneracy Guards (First Draft)

Status: initial theorem-scope guards drafted to avoid overgeneralization from executed slices.

Guard set (working):
1. Exclude structurally degenerate one-hot/sparse families:
	 - enforce `support_class != degenerate` (already used in OP2 artifacts).
2. Require admissible expansion index:
	 - \(p=0\) handled as geometric-tail branch; for \(p>0\), require \(p\mid 2d\) so the
		 implemented expansion and support map are well-defined.
3. Require simplex-like outer support frame for barycentric formulas:
	 - apply D-OP2-1/D-OP2-2 lifts only where diagonal exponents are the canonical
		 \(2d e_i\) frame used in the benchmark construction.
4. Record stratum-availability metadata before claiming obstruction mechanism:
	 - for each \((d,n,p)\), log whether \(\mathcal{I}_n\), \(\mathcal{F}\), \(\mathcal{E}\), \(\mathcal{V}\)
		 are present in \(S_{d,p}^{(n)}\).

Current use in theorem-stage text:
- Delta-support branch statements remain slice-local empirical claims.
- Analytic promotion attempts must explicitly cite guards (1)-(4) and identify any
	additional assumption needed for circuit-number obstruction.

## 20) D-OP2-4 Minimal \(n=4\) Stratified-Slack Functional (Defined + Checked)

Status: candidate functional defined and checked on current OP2 artifacts.

For \(n=4\), \(p>0\), \(p\mid 2d\), define \(m:=2d/p\) and stratum indicators
\(\chi_{\mathcal{F}},\chi_{\mathcal{I}_4}\in\{0,1\}\) by support availability in
\(S_{d,p}^{(4)}\):
- \(\chi_{\mathcal{F}}=1\) iff face-interior points are available (equivalently \(m\ge 3\)).
- \(\chi_{\mathcal{I}_4}=1\) iff full-interior points are available (equivalently \(m\ge 4\)).

Define the minimal stratified activation functional
\[
\Phi_4(d,p):=\mathbf{1}\{\chi_{\mathcal{F}}+\chi_{\mathcal{I}_4}>0\}=\mathbf{1}\{m\ge 3\}.
\]
For the \(p=0\) geometric-tail branch, keep the existing boundary classification branch.

Working class rule (current formulation class):
- predict feasible class \(F\) if \(p=0\) or \(\Phi_4(d,p)=0\),
- predict robust-failure class \(R\) if \(\Phi_4(d,p)=1\).

Checks against existing OP2 table artifact (`op2_delta_support_table.csv`):
1. All simplex-like \(d=6,n=4\) rows: 30/30 matches.
2. Per-\(p\) on \(d=6,n=4\):
	 - \(p=0\): 3/3 (F), \(p=6\): 3/3 (F),
	 - \(p\in\{1,2,3,4\}\): 24/24 (R).
3. Stress-only \(d=6,n=4\) subset: 12/12 matches.

Circuit-obstruction linkage (delimited claim, not yet theorem closure):
1. \(\Phi_4=0\) confines support to \(\mathcal{V}\cup\mathcal{E}\), i.e., edge-only strata with no
	forced face/full-interior barycentric activation.
2. \(\Phi_4=1\) activates at least one face-interior/full-interior stratum, introducing
	strictly positive multi-coordinate barycentric profiles absent in the edge-only regime.
3. On executed slices under D-OP2-3 guards, this activation aligns with robust SONC-failure;
	we treat this as a scoped structural marker pending a full circuit-number inequality proof.

Immediate next dependency after D-OP2-4:
1. Prove (or sharply delimit) a face-activation \(\Rightarrow\) circuit-deficit lemma in the
	uniform template for \(n=4\).
2. Test transfer of that lemma to boundary/mixed templates under the same guard set.
3. If transfer fails, record the minimal counterexample pattern and update B2 scope text.
