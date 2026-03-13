from .base import Foundation
from .orthsys import OrthSystem



class RationalAprox(Foundation):
    """
    ``RationalAprox`` calculates a rational approximation for a given
    function. ``RationalApprox`` is the preferred public alias; this class
    name is retained for backward compatibility. It takes one argument
    `orth` which is an instance of ``OrthSystem`` and does all the
    computations in the scope of this object.
    """

    def __init__(self, orth):
        """
        initiate a rational approximation framework.
        """
        if not isinstance(orth, OrthSystem):
            raise TypeError("`orth` must be an `OrthSystem` instance.")
        self.OrthSys = orth

    def RatLSQ(self, m, n, f):
        """
        Calculates a rational function :math:`\frac{p}{q}` where 
        :math:`p, q` consists of the first :math:`m, n` elements of 
        the orthonormal basis :math:`||f-\frac{p}{q}||_2` is minimized.
        """
        from numpy import diag, array, zeros, vstack, hstack
        from scipy.linalg import LinAlgError, solve
        if not isinstance(m, int) or not isinstance(n, int):
            raise TypeError("`m` and `n` must be integers.")
        if m < 0 or n < 0:
            raise ValueError("`m` and `n` must be non-negative.")
        if self.OrthSys.OrthBase == []:
            raise ValueError("No orthonormal basis is available. Call `FormBasis()` on the orthogonal system first.")
        if m >= self.OrthSys.num_base or n >= self.OrthSys.num_base:
            raise ValueError(
                "`m` and `n` must both be smaller than the size of the orthonormal basis (%d)."
                % (self.OrthSys.num_base)
            )
        self.M = diag([1 for _ in range(m + 1)])
        self.Z = zeros((m + 1, n))
        self.S = zeros((n, n))
        self.F = array([self.OrthSys.inner(self.OrthSys.OrthBase[j], f)
                        for j in range(m + 1)])
        self.G = array(
            [-self.OrthSys.inner(self.OrthSys.OrthBase[k], f**2) for k in range(1, n + 1)])
        for j in range(m + 1):
            for k in range(1, n + 1):
                self.Z[j][
                    k - 1] = - self.OrthSys.inner(self.OrthSys.OrthBase[j], f * self.OrthSys.OrthBase[k])
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                self.S[j - 1][k - 1] = self.OrthSys.inner(
                    self.OrthSys.OrthBase[j], f**2 * self.OrthSys.OrthBase[k])
        H = vstack((hstack((self.M, self.Z)), hstack((self.Z.T, self.S))))
        r = hstack((self.F, self.G))
        try:
            ab = solve(H, r)
        except LinAlgError as error:
            raise ValueError(
                "Failed to solve the rational least-squares system. The basis or target function may lead to a singular system."
            ) from error
        a, b = ab[:m + 1], ab[m + 1:]
        numer = sum([a[i] * self.OrthSys.OrthBase[i] for i in range(m + 1)])
        denom = 1 + sum([b[i] * self.OrthSys.OrthBase[i + 1]
                         for i in range(n)])
        return numer / denom


RationalApprox = RationalAprox

