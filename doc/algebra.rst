=================================
Group-Ring Foundations
=================================

This chapter introduces the algebraic layer behind the optimization modules.

Algebraic Setting
=================================

Let :math:`S` be a finitely generated commutative semigroup and let
:math:`\mathbb{R}[S]` denote its semigroup algebra. In Irene, elements of
:math:`\mathbb{R}[S]` are represented as finite sums

.. math::

   f = \sum_{\alpha \in \mathrm{supp}(f)} c_\alpha \alpha,

where :math:`\alpha` is a semigroup element and :math:`c_\alpha \in \mathbb{R}`.
This perspective generalizes classical polynomial notation and keeps support,
degree, and structural operations explicit for optimization routines.

When relations are present, the algebra is effectively handled modulo those
relations through reduction in the semigroup representation. This is useful for
modeling quotient structures that arise naturally in symbolic formulations.

Commutative Semigroup and Semigroup Algebra
===========================================

The module ``grouprings.py`` provides:

1. ``CommutativeSemigroup``: generator-based semigroup representation.
2. ``SemigroupAlgebra``: algebra construction over that semigroup.
3. ``AtomicSGElement`` and ``SemigroupAlgebraElement``: monomial and polynomial-like elements.

This representation is used by higher layers to extract supports, exponents,
and structural information required by geometric and SONC relaxations.

Support and Geometry
=================================

The support of :math:`f` is central in both geometric and SONC constructions.
By converting semigroup monomials to exponent tuples, Irene can compute Newton
polytope information and barycentric relations directly from algebraic input.

This is the key bridge from symbolic algebra to convex-geometric objects used
in lower-bound certificates.

Symbolic Engine: SymEngine Primary with SymPy Fallback
------------------------------------------------------

IreneRewrite uses a dual-engine design for symbolic computation, implemented in
``symbolic_engine.py`` as the ``SymbolicEngine`` class (imported as ``engine``).

**Design Rationale.** SymEngine provides a C++ backend that is significantly faster
for polynomial expansion, numeric evaluation, and matrix construction. However, it
lacks several advanced APIs required by SDP relaxation pipelines — notably Gröbner
basis computation, the full ``Poly`` API (``as_dict()``, domain arithmetic),
``PolyMatrix``/``DomainMatrix``, and ``lambdify``. The dual-engine router resolves
this by attempting SymEngine first and falling back to SymPy transparently when an
operation is unsupported.

**Fallback Mechanism.** Every routed operation follows this pattern:

1. Attempt the SymEngine C++ path (e.g., ``se.expand()``, ``se.DenseMatrix``).
2. On ``AttributeError``, ``NotImplementedError``, or ``LibExpressionException``,
   cast all SymEngine inputs to SymPy via ``to_sympy()`` and call the native SymPy
   equivalent.
3. Return the result (SymPy objects are accepted downstream; no forced cast-back).

The ``to_sympy()`` helper short-circuits when the input is already a SymPy object,
avoiding redundant tree conversions. The ``fallback_log`` attribute records which
calls fell back, and ``engine.fallback_stats()`` returns per-operation counts for
profiling.

**Performance Profile.** Instrumented traces show that SymEngine handles the bulk of
expand/matrix/symbol operations, while Gröbner basis, ``Poly``, and ``lambdify``
always route to SymPy (these APIs simply do not exist in SymEngine). The net effect
is faster polynomial manipulation with no loss of advanced algebraic functionality.

**Usage Pattern.** Existing modules import the unified engine rather than raw
SymPy/SymEngine::

    from Irene.symbolic_engine import engine
    x = engine.Symbol('x')
    expr = engine.expand((x + 1)**2)       # SymEngine C++ expand
    g = engine.groebner([f1, f2], x)      # auto-fallback to SymPy
    p = engine.Poly(expr, x)              # SymPy Poly (SymEngine lacks this)

This pattern ensures that all symbolic code in IreneRewrite benefits from the
dual-engine routing without importing either backend directly.

**Selecting the Symbolic Backend.** Users can choose between SymEngine and pure
SymPy at runtime, either programmatically or via the environment::

    # Programmatic selection (applies to the default `engine` singleton):
    from Irene.symbolic_engine import engine, set_symbolic_backend, get_symbolic_backend
    set_symbolic_backend('symengine')    # or 'sympy' / 'auto'
    assert get_symbolic_backend() == 'symengine'

    # Environment variable (read once at import time):
    #   IRENE_SYMBOLIC_BACKEND=symengine python3 my_script.py
    #   IRENE_SYMBOLIC_BACKEND=sympy     python3 my_script.py
    #   IRENE_SYMBOLIC_BACKEND=auto      python3 my_script.py   (default)

Accepted values: ``symengine`` (default when installed), ``sympy``, and
``auto`` (prefer SymEngine when available, else SymPy). When ``symengine`` is
not installed the engine automatically runs in SymPy mode and
``engine.available_backends()`` reports ``['sympy']``; installing the package
with the optional extra ``pip install .[symengine]`` enables the C++ backend.
Operations that only exist in SymPy (Gröbner basis, ``Poly``, ``lambdify``,
``DomainMatrix``) are backend-independent and always run on SymPy.

Differential Operators
=================================

Beyond static algebraic representation, ``SemigroupAlgebra`` supports derivations.
Given a base map, derivatives are propagated through expressions using product-rule behavior.

For factors :math:`u` and :math:`v`, the implementation follows:

.. math::

   D(uv) = D(u)v + uD(v).

This enables a workflow where optimization is not only over algebraic objects,
but also over structures enriched with differential operators.

In code, the derivative path is organized as:

1. ``SemigroupAlgebra.add_derivative`` registers a derivation map.
2. ``SemigroupAlgebra.derivative`` selects a registered derivation by index.
3. ``SemigroupAlgebra.diff`` applies recursive product-rule expansion.

This method-level design makes differentiation explicit and extensible for
problem formulations where algebraic structure and operator behavior are coupled.

**Multiple Derivation Support.** The ``derivatives`` attribute is a list, so the
algebra can carry several independent derivation operators simultaneously. Each call
to ``add_derivative(base_map)`` appends a new map at index ``len(derivatives)``.
The ``derivative(expr, idx)`` method then selects operator ``idx`` by position:

.. math::

   d_0, d_1, \dots, d_{k-1} : \mathbb{R}[S] \to \mathbb{R}[S],

where each :math:`d_i` is registered independently via its own base map on the
generators. This supports systems with up to ~10 derivation generators (sufficient
for holonomic function representations and multi-variable differential constraints).

**Notation.** The derivation operators :math:`d_x`, :math:`d_y`, etc., are linear maps
on the semigroup algebra satisfying Leibniz rules. They are **not** Leibniz fractions
(:math:`dy/dx` is notation for a ratio of differentials; :math:`d_x` is an operator).
In code, ``derivative(expr, 0)`` applies the first registered derivation (e.g.,
:math:`d_x`), while ``derivative(expr, 1)`` applies the second (e.g., :math:`d_y`).

From a theoretical viewpoint, derivations are linear maps
:math:`D: \mathbb{R}[S] \to \mathbb{R}[S]` that satisfy Leibniz rules. Irene's
``add_derivative`` and ``diff`` pipeline implements this behavior directly on
semigroup-algebra elements.

Why This Matters for POP
=================================

The shift from polynomial-only notation to group-ring structures improves the
connection between symbolic representation and geometric computation:

1. Exponent vectors are directly accessible for convex hull and delta-set routines.
2. Algebraic reductions remain explicit and programmatic.
3. The same representation supports SDP, GP, and SONC method families.
