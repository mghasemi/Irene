#!/usr/bin/env python3
"""
Instrumented relaxation run v2 — directly patches engine methods BEFORE any Irene imports.
"""
import sys, os, time

sys.path.insert(0, '/home/mehdi/Code/Python/IreneRewrite')

# ── Step 1: Import the engine and instrument it BEFORE any Irene module imports ──
import Irene.symbolic_engine as se_mod
from Irene.symbolic_engine import SymbolicEngine

# Save reference to the module-level engine instance
real_engine = se_mod.engine

# Create counters
counts = {}
times = {}

# Wrap all engine methods by directly patching the instance
for attr_name in dir(real_engine):
    if attr_name.startswith('_'):
        continue
    attr = getattr(real_engine, attr_name)
    if not callable(attr):
        continue
    
    method_name = attr_name
    original_method = attr
    
    def make_wrapper(name, orig):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = orig(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1e6
                counts[name] = counts.get(name, 0) + 1
                times[name] = times.get(name, 0) + elapsed
                return result
            except Exception:
                elapsed = (time.perf_counter() - t0) * 1e6
                counts[f"{name}_ERR"] = counts.get(f"{name}_ERR", 0) + 1
                times[f"{name}_ERR"] = times.get(f"{name}_ERR", 0) + elapsed
                raise
        return wrapper
    
    try:
        setattr(real_engine, method_name, make_wrapper(method_name, original_method))
    except AttributeError:
        pass  # skip read-only properties like Equality, GreaterThan, etc.

# Also instrument to_sympy/to_symengine
import Irene.symbolic_engine as sm
_orig_to_sympy = sm.to_sympy
_orig_to_symengine = sm.to_symengine
_conv_calls = {'to_sympy': 0, 'to_symengine': 0}
_conv_time = {'to_sympy': 0.0, 'to_symengine': 0.0}

def _wrapped_to_sympy(obj):
    t0 = time.perf_counter()
    r = _orig_to_sympy(obj)
    _conv_time['to_sympy'] += (time.perf_counter() - t0) * 1e6
    _conv_calls['to_sympy'] += 1
    return r

def _wrapped_to_symengine(obj):
    t0 = time.perf_counter()
    r = _orig_to_symengine(obj)
    _conv_time['to_symengine'] += (time.perf_counter() - t0) * 1e6
    _conv_calls['to_symengine'] += 1
    return r

sm.to_sympy = _wrapped_to_sympy
sm.to_symengine = _wrapped_to_symengine

# ── Step 2: NOW import Irene and run ──
from Irene.grouprings import CommutativeSemigroup, SemigroupAlgebra
from Irene.program import OptimizationProblem
from Irene.relaxation_api import RelaxationEngine

print("=" * 70)
print("INSTRUMENTED MOTZKIN RELAXATION — order 1 SOS")
print("=" * 70)

sg = CommutativeSemigroup(["x", "y"])
sga = SemigroupAlgebra(sg)
x, y = sga["x"], sga["y"]
motzkin = x**4 * y**2 + x**2 * y**4 + 1 - 3 * x**2 * y**2
prog = OptimizationProblem(sga)
prog.set_objective(motzkin)

t0 = time.perf_counter()
engine = RelaxationEngine(prog, order=1, verbosity=0)
result = engine.solve("sos")
total_time = (time.perf_counter() - t0) * 1000

print(f"\nTotal wall time: {total_time:.1f} ms")
print(f"SOS bound: {result.value}")
print(f"Status: {result.status}")

print(f"\n{'─'*65}")
print(f"{'engine.<method> call counts':<45} {'calls':>7} {'total µs':>10}")
print(f"{'─'*65}")

# Categorize
sympy_fallback = {'Poly', 'groebner', 'reduced', 'lambdify', 'sympify', 'latex', 'Function',
                  'DomainMatrix', 'PolyMatrix', 'QQ'}
symengine_native = {'Symbol', 'symbols', 'expand', 'zeros', 'Matrix', 'sqrt', 'Abs'}
prop_access = {'Equality', 'GreaterThan', 'LessThan', 'StrictGreaterThan', 'StrictLessThan',
               'PolynomialError'}

total_engine = 0
total_fallback = 0
total_native = 0
total_prop = 0
total_other = 0

for name in sorted(counts.keys(), key=lambda n: -times.get(n, 0)):
    c = counts[name]
    t = times[name]
    total_engine += t
    cat = ''
    if name in sympy_fallback:
        cat = ' [SymPy fallback]'
        total_fallback += t
    elif name in symengine_native:
        cat = ' [SymEngine]'
        total_native += t
    elif name in prop_access:
        cat = ' [property]'
        total_prop += t
    else:
        cat = ''
        total_other += t
    print(f"  {name:<43} {c:>7} {t:>10.1f}{cat}")

print(f"\n{'─'*65}")
print(f"  Engine total:                     {total_engine:>10.1f} µs")
print(f"  SymPy-fallback ops (Poly/groebner/reduced/etc): {total_fallback:>10.1f} µs")
print(f"  SymEngine-native ops (Symbol/expand/Matrix):    {total_native:>10.1f} µs")
print(f"  Property access:                                 {total_prop:>10.1f} µs")
print(f"  Other:                                           {total_other:>10.1f} µs")

conv_total = _conv_time['to_sympy'] + _conv_time['to_symengine']
print(f"\n  to_sympy() calls:    {_conv_calls['to_sympy']:>7}  total: {_conv_time['to_sympy']:>10.1f} µs")
print(f"  to_symengine() calls:{_conv_calls['to_symengine']:>7}  total: {_conv_time['to_symengine']:>10.1f} µs")
print(f"  Conversion total:                             {conv_total:>10.1f} µs = {conv_total/1000:.2f} ms")
print(f"  Conversion as % of wall time: {conv_total/1000/total_time*100:.1f}%")

print(f"\n  All engine ops + conversion: {total_engine + conv_total:.0f} µs = {(total_engine + conv_total)/1000:.1f} ms")
print(f"  That's {(total_engine + conv_total)/1000/total_time*100:.1f}% of total wall time")
print(f"  Remaining {(total_time - (total_engine + conv_total)/1000):.1f} ms is SDP solve + numpy/scipy overhead")

# Fallback log
print(f"\n{'─'*65}")
print("FALLBACK LOG:")
fb = real_engine.fallback_stats()
for op, c in sorted(fb.items(), key=lambda x: -x[1]):
    print(f"  {op}: {c}x fell back to SymPy")
if not fb:
    print("  (none)")

print(f"\n{'─'*65}")
print("DIAGNOSIS:")
if total_fallback > total_native:
    print(f"  SymPy-fallback ops use {total_fallback:.0f} µs vs SymEngine-native {total_native:.0f} µs")
    print(f"  >>> {(total_fallback/(total_fallback+total_native)*100):.0f}% of engine time is in SymPy-fallback operations")
if conv_total > total_native:
    print(f"  Conversion overhead ({conv_total:.0f} µs) exceeds SymEngine-native ops ({total_native:.0f} µs)")
    print(f"  >>> Paying more for conversions than we save from SymEngine speed")
if total_fallback + conv_total > total_engine * 0.5:
    print(f"  Fallback + conversion = {total_fallback+conv_total:.0f} µs out of {total_engine:.0f} µs engine time")
    print(f"  >>> Eliminating engine layer would save ~{total_fallback+conv_total:.0f} µs")
