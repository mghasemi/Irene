import pytest
from sympy import symbols, simplify
from Irene.mean_certificates import MeanCertificate

def test_choi_lam_certificate():
    X,Y,Z,W = symbols('X Y Z W')
    variables = [X**4, Y**4, Z**4, W**4]
    weights = [1]*4
    
    mc = MeanCertificate(variables, weights)
    cert = mc.certificate(q=1,p=0)
    
    # At minimum (Q=0), certificate must be ≥ 0:
    substitution_min = {X:1,Y:1,Z:1,W:1}
    assert simplify(cert.subs(substitution_min)).evalf() >= 0
    
    # Test at X=2,Y/Z/W=1 → expected to be positive:
    substitution_test = {X:2,Y:1,Z:1,W:1}
    actual_value = cert.subs(substitution_test).evalf()
    assert actual_value >= -1e-8