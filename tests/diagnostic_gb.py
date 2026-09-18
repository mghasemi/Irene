"""Inspect Groebner basis structure and variable elimination."""

from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxations import SDPRelaxations, RelaxationConfig

generator_names = ["x", "y", "u", "v", "d_vx", "d_vy", "d_vu"]
sg = CommutativeSemigroup(generator_names)
sga = SemigroupAlgebra(sg)

generators = {g.ext_rep[0].name: g for g in sg.generators}
x, y, u, v, d_vx, d_vy, d_vu = (sga[name] for name in generator_names)

# Derivatives
sga.add_derivative({generators["x"]: 1, generators["y"]: 0, generators["u"]: 0, generators["v"]: y * v})
sga.add_derivative({generators["x"]: 0, generators["y"]: 1, generators["u"]: 0, generators["v"]: x * v})
sga.add_derivative({generators["x"]: d_vx, generators["y"]: d_vy, generators["u"]: d_vu, generators["v"]: 1})

relations = [
    u**2 - (x * y + 1),
    v * y**2 * d_vx - (y - x * y * v * d_vy),
    v * x**2 * d_vy - (x - x * y * v * d_vx),
    2 * u * d_vu - (y * d_vx + x * d_vy),
]

prog = OptimizationProblem(sga, relations=relations)
prog.set_objective(1 - x**2 + y)
prog.add_constraints([
    u + x / 2 - x**2 + 2 * y,
    2 - y**2,
    y - x * v + 1,
])

config = RelaxationConfig(
    reduction_method="border_basis",
    quotient_basis="groebner",
    monomial_pruning=True,
    sparsity_detection=True,
    verbose_reduction=False,
)

sdp_relax = SDPRelaxations.from_problem(prog, config=config)

# ====================================================================
# Inspect the Groebner basis
# ====================================================================
print("=" * 70)
print("GROEBNER BASIS INSPECTION")
print("=" * 70)

gb = sdp_relax.Groebner
if gb:
    print(f"\nGroebner basis size: {len(gb)}")
    for i, g in enumerate(gb):
        deg = sg.degree(g) if hasattr(g, 'content') else '?'
        content_len = len(g.content) if hasattr(g, 'content') else '?'
        
        # Check which generators appear
        gens_present = []
        if hasattr(g, 'content'):
            for coeff, mono in g.content:
                if hasattr(mono, 'array_form'):
                    for gen, exp in mono.array_form:
                        if exp != 0 and gen.name not in gens_present:
                            gens_present.append(gen.name)
        
        print(f"  GB[{i}]: deg={deg}, terms={content_len}, generators={gens_present}")

# ====================================================================
# Check the SymDict — what does the SDP see as "variables"?
# ====================================================================
print("\n" + "=" * 70)
print("SYMBOL DICTIONARY (what SDP sees)")
print("=" * 70)

sym_dict = sdp_relax.SymDict
for name, val in sym_dict.items():
    print(f"  {name}: {type(val).__name__}")

# ====================================================================
# Check AuxSyms — auxiliary symbols from relations
# ====================================================================
print("\n" + "=" * 70)
print("AUXILIARY SYMBOLS")
print("=" * 70)

aux = sdp_relax.AuxSyms
for i, a in enumerate(aux):
    print(f"  AuxSym[{i}]: {a}")

# ====================================================================
# Check FreeRelations — how relations are stored
# ====================================================================
print("\n" + "=" * 70)
print("FREE RELATIONS")
print("=" * 70)

rels = sdp_relax.FreeRelations
for i, r in enumerate(rels):
    deg = sg.degree(r) if hasattr(r, 'content') else '?'
    print(f"  Rel[{i}]: degree={deg}")

# ====================================================================
# Key question: Can d_vx, d_vy, d_vu be eliminated?
# ====================================================================
print("\n" + "=" * 70)
print("VARIABLE ELIMINATION CHECK")
print("=" * 70)

# Check if any GB element has leading term in d_v* only
for i, g in enumerate(gb):
    if hasattr(g, 'content'):
        for coeff, mono in g.content:
            if hasattr(mono, 'array_form'):
                dv_terms = [(gen.name, exp) for gen, exp in mono.array_form 
                           if gen.name.startswith('d_v') and exp != 0]
                non_dv_terms = [(gen.name, exp) for gen, exp in mono.array_form 
                               if not gen.name.startswith('d_v') and exp != 0]
                if dv_terms and not non_dv_terms:
                    print(f"  GB[{i}] has pure d_v* term: {dv_terms}")

# ====================================================================
# Test: What if we remove the ADE relations entirely but keep v?
# ====================================================================
print("\n" + "=" * 70)
print("VARIANT F: Only u^2=xy+1, constraint y-x*v+1>=0 with v free")
print("=" * 70)

sg_f = CommutativeSemigroup(["x", "y", "u", "v"])
sga_f = SemigroupAlgebra(sg_f)
gens_f = {g.ext_rep[0].name: g for g in sg_f.generators}
x_f, y_f, u_f, v_f = sga_f["x"], sga_f["y"], sga_f["u"], sga_f["v"]

relations_f = [u_f**2 - (x_f * y_f + 1)]

prog_f = OptimizationProblem(sga_f, relations=relations_f)
prog_f.set_objective(1 - x_f**2 + y_f)
prog_f.add_constraints([
    u_f + x_f / 2 - x_f**2 + 2 * y_f,
    2 - y_f**2,
    y_f - x_f * v_f + 1,
])

sdp_f = SDPRelaxations.from_problem(prog_f, config=config)
sdp_f.MomentsOrd(2)
sdp_f.SetSDPSolver("cvxopt")
try:
    sdp_f.InitSDP()
    sdp_f.Minimize()
    sol_f = sdp_f.Solution
    if sol_f and hasattr(sol_f, 'Primal') and sol_f.Primal is not None:
        print(f"  SDP lower bound: {float(sol_f.Primal):.8f}")
    else:
        print("  No solution / infeasible")
except Exception as e:
    print(f"  Error: {e}")

# ====================================================================
# Test: What if we add v^2 constraint to bound v?
# ====================================================================
print("\n" + "=" * 70)
print("VARIANT G: u^2=xy+1, v bounded by v^2 <= 10")
print("=" * 70)

sg_g = CommutativeSemigroup(["x", "y", "u", "v"])
sga_g = SemigroupAlgebra(sg_g)
gens_g = {g.ext_rep[0].name: g for g in sg_g.generators}
x_g, y_g, u_g, v_g = sga_g["x"], sga_g["y"], sga_g["u"], sga_g["v"]

relations_g = [u_g**2 - (x_g * y_g + 1)]

prog_g = OptimizationProblem(sga_g, relations=relations_g)
prog_g.set_objective(1 - x_g**2 + y_g)
prog_g.add_constraints([
    u_g + x_g / 2 - x_g**2 + 2 * y_g,
    2 - y_g**2,
    y_g - x_g * v_g + 1,
    10 - v_g**2,  # bound v
])

sdp_g = SDPRelaxations.from_problem(prog_g, config=config)
sdp_g.MomentsOrd(2)
sdp_g.SetSDPSolver("cvxopt")
try:
    sdp_g.InitSDP()
    sdp_g.Minimize()
    sol_g = sdp_g.Solution
    if sol_g and hasattr(sol_g, 'Primal') and sol_g.Primal is not None:
        print(f"  SDP lower bound (v bounded): {float(sol_g.Primal):.8f}")
    else:
        print("  No solution / infeasible")
except Exception as e:
    print(f"  Error: {e}")
