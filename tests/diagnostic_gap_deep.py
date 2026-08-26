"""Deep diagnostic: Inspect Groebner quotient and moment matrix structure."""

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
sdp_relax.MomentsOrd(2)

# ====================================================================
# Inspect internal state after Groebner quotient computation
# ====================================================================
print("=" * 70)
print("INTERNAL STATE INSPECTION")
print("=" * 70)

# Check what attributes are available
attrs = [a for a in dir(sdp_relax) if not a.startswith('__')]
print(f"\nSDPRelaxations attributes ({len(attrs)} total):")
for a in sorted(attrs):
    val = getattr(sdp_relax, a, None)
    if val is not None and not callable(val):
        t = type(val).__name__
        if hasattr(val, '__len__') and t not in ('str', 'SemigroupAlgebraElement'):
            print(f"  {a}: {t}[{len(val)}]")
        else:
            print(f"  {a}: {t}")

# ====================================================================
# Check the quotient basis explicitly
# ====================================================================
print("\n" + "=" * 70)
print("QUOTIENT BASIS ANALYSIS")
print("=" * 70)

# The Groebner basis is computed in SDPRelaxations.__init__ via _compute_groebner_basis
if hasattr(sdp_relax, '_gb'):
    gb = sdp_relax._gb
    print(f"\nGroebner basis size: {len(gb)}")
    for i, g in enumerate(gb):
        print(f"  GB[{i}]: degree={sg.degree(g)}, content_len={len(g.content) if hasattr(g,'content') else '?'}")

# Check the monomial basis used for the moment matrix
if hasattr(sdp_relax, 'Monomials'):
    mons = sdp_relax.Monomials
    print(f"\nMoment monomial basis size: {len(mons)}")
    
    # Count how many involve d_vx, d_vy, d_vu
    dv_count = 0
    for m in mons:
        if hasattr(m, 'content'):
            for coeff, mono in m.content:
                if hasattr(mono, 'array_form'):
                    for gen, exp in mono.array_form:
                        if gen.name in ('d_vx', 'd_vy', 'd_vu'):
                            dv_count += 1
                            break
    print(f"Monomials involving d_v* variables: {dv_count}/{len(mons)}")

# ====================================================================
# Test: What happens without the ADE relations (only u^2 = xy+1)?
# ====================================================================
print("\n" + "=" * 70)
print("VARIANT D: Only algebraic lift, no ADE differential relations")
print("=" * 70)

sg_d = CommutativeSemigroup(["x", "y", "u"])
sga_d = SemigroupAlgebra(sg_d)
gens_d = {g.ext_rep[0].name: g for g in sg_d.generators}
x_d, y_d, u_d = sga_d["x"], sga_d["y"], sga_d["u"]

relations_d = [u_d**2 - (x_d * y_d + 1)]

prog_d = OptimizationProblem(sga_d, relations=relations_d)
prog_d.set_objective(1 - x_d**2 + y_d)
prog_d.add_constraints([
    u_d + x_d / 2 - x_d**2 + 2 * y_d,
    2 - y_d**2,
])

sdp_d = SDPRelaxations.from_problem(prog_d, config=config)
sdp_d.MomentsOrd(2)
sdp_d.SetSDPSolver("cvxopt")
sdp_d.InitSDP()
sdp_d.Minimize()
sol_d = sdp_d.Solution
if sol_d and hasattr(sol_d, 'Primal') and sol_d.Primal is not None:
    print(f"  SDP lower bound (no ADE): {float(sol_d.Primal):.8f}")

# ====================================================================
# Test: What happens with v but without d_v* variables?
# ====================================================================
print("\n" + "=" * 70)
print("VARIANT E: x,y,u,v with u^2=xy+1, no differential relations")
print("=" * 70)

sg_e = CommutativeSemigroup(["x", "y", "u", "v"])
sga_e = SemigroupAlgebra(sg_e)
gens_e = {g.ext_rep[0].name: g for g in sg_e.generators}
x_e, y_e, u_e, v_e = sga_e["x"], sga_e["y"], sga_e["u"], sga_e["v"]

relations_e = [u_e**2 - (x_e * y_e + 1)]

prog_e = OptimizationProblem(sga_e, relations=relations_e)
prog_e.set_objective(1 - x_e**2 + y_e)
prog_e.add_constraints([
    u_e + x_e / 2 - x_e**2 + 2 * y_e,
    2 - y_e**2,
    y_e - x_e * v_e + 1,
])

sdp_e = SDPRelaxations.from_problem(prog_e, config=config)
sdp_e.MomentsOrd(2)
sdp_e.SetSDPSolver("cvxopt")
sdp_e.InitSDP()
sdp_e.Minimize()
sol_e = sdp_e.Solution
if sol_e and hasattr(sol_e, 'Primal') and sol_e.Primal is not None:
    print(f"  SDP lower bound (v as free variable): {float(sol_e.Primal):.8f}")
