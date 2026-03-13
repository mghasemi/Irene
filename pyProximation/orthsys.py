from .base import Foundation
from .measure import Measure


class OrthSystem(Foundation):
    """
    ``OrthogonalSystem`` class produces an orthogonal system of functions
    according to a suggested basis of functions and a given measure
    supported on a given region.

    This basically performs a 'Gram-Schmidt' method to extract the
    orthogonal basis. The inner product is obtained by integration of
    the product of functions with respect to the given measure (more
    accurately, the distribution).

    To initiate an instance of this class one should provide a list of
    symbolic variables `variables` and the range of each variable as a
    list of lists ``var_range``.

            To initiate an orthogonal system of functions, one should provide
            a list of symbolic variables ``variables`` and the range of each
            these variables as a list of lists ``var_range``.
    """

    def __init__(self, variables, var_range, env='sympy'):
        """
        To initiate an orthogonal system of functions, one should provide
        a list of symbolic variables ``variables`` and the range of each
        these variables as a list of lists ``var_range``.
        """
        if not isinstance(variables, list) or not isinstance(var_range, list):
            raise TypeError(
                "OrthSystem expects two lists: `variables` and `var_range`."
            )
        if len(variables) == 0:
            raise ValueError("`variables` cannot be empty.")
        if len(variables) != len(var_range):
            raise ValueError("`variables` and `var_range` must have the same length.")
        if env != 'sympy':
            raise ValueError("Only 'sympy' is supported as symbolic environment.")
        self.Env = 'sympy'
        self.EnvAvail = ['sympy']
        self.Vars = variables
        self.num_vars = len(self.Vars)
        self.Domain = var_range
        self.measure = Measure(self.Domain, 1)
        self.OriginalBasis = []
        self.OrthBase = []
        self.Numerical = False
        self._zero_tol = 1e-12
        self.CommonSymFuncs(self.Env)

    def PolyBasis(self, n):
        """
        Generates a polynomial basis from variables consisting of all
        monomials of degree at most ``n``.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("`n` must be a non-negative integer.")
        from itertools import product
        B = []
        for o in product(range(n + 1), repeat=self.num_vars):
            if sum(o) <= n:
                T_ = 1
                for idx in range(self.num_vars):
                    T_ *= self.Vars[idx]**o[idx]
                B.append(T_)
        return B

    def FourierBasis(self, n):
        """
        Generates a Fourier basis from variables consisting of all
        :math:`sin` & :math:`cos` functions with coefficients at most `n`.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("`n` must be a non-negative integer.")
        for interval in self.Domain:
            if interval[0] == interval[1]:
                raise ValueError("Fourier bases require non-degenerate intervals.")
        from itertools import product
        B = []
        for o in product(range(n + 1), repeat=self.num_vars):
            if sum(o) <= n:
                SinCos = product(range(2), repeat=self.num_vars)
                for ex in SinCos:
                    T_ = 1
                    for idx in range(self.num_vars):
                        period = self.Domain[idx][1] - self.Domain[idx][0]
                        if o[idx] != 0:
                            if ex[idx] == 0:
                                T_ *= self.cos(2 * self.pi *
                                               o[idx] * self.Vars[idx] / period)
                            else:
                                T_ *= self.sin(2 * self.pi *
                                               o[idx] * self.Vars[idx] / period)
                    B.append(T_)
        return list(set(B))

    def TensorPrd(self, Bs):
        """
        Takses a list of symbolic bases, each one a list of symbolic 
        expressions and returns the tensor product of them as a list.
        """
        if not isinstance(Bs, list) or Bs == []:
            raise ValueError("Cannot compute a tensor product from an empty list of bases.")
        if any(base == [] for base in Bs):
            raise ValueError("Each basis in the tensor product must be non-empty.")
        from itertools import product
        TP = product(*Bs)
        TBase = []
        for itm in TP:
            t_prd = 1
            for ent in itm:
                t_prd = t_prd * ent
            TBase.append(self.expand(t_prd))
        return TBase

    def SetMeasure(self, M):
        """
        To set the measure which the orthogonal system will be computed,
        simply call this method with the corresponding distribution as
        its parameter `dm`; i.e, the parameter is `d(m)` where `m` is
        the original measure.
        """
        if not isinstance(M, Measure):
            raise TypeError("The argument must be a `Measure` instance.")
        self.measure = M

    def Basis(self, base_set):
        """
        To specify a particular family of function as a basis, one should
        call this method with a list ``base_set`` of linearly independent
        functions.
        """
        if not isinstance(base_set, list):
            raise TypeError("A list of symbolic functions is expected.")
        if base_set == []:
            raise ValueError("`base_set` cannot be empty.")
        self.OriginalBasis = base_set
        self.num_base = len(self.OriginalBasis)

    def inner(self, f, g):
        """
        Computes the inner product of the two parameters with respect to
        the measure ``measure``.
        """
        from sympy import lambdify
        F = lambdify(self.Vars, f * g, "numpy")
        try:
            m = self.measure.integral(F)
        except Exception as error:
            raise ValueError(
                "Failed to evaluate the inner product on the current measure."
            ) from error
        return m

    def project(self, f, g):
        """
        Finds the projection of ``f`` on ``g`` with respect to the inner
        product induced by the measure ``measure``.
        """
        denominator = self.inner(g, g)
        if abs(float(denominator)) < self._zero_tol:
            raise ZeroDivisionError("Cannot project onto a zero-norm basis element.")
        return g * self.inner(f, g) / denominator

    def FormBasis(self):
        """
        Call this method to generate the orthogonal basis corresponding
        to the given basis via ``Basis`` method.
        The result will be stored in a property called ``OrthBase`` which
        is a list of function that are orthogonal to each other with
        respect to the measure ``measure`` over the given range ``Domain``.
        """
        if self.OriginalBasis == []:
            raise ValueError("No basis has been set. Call `Basis()` first.")
        self.OrthBase = []
        for f in self.OriginalBasis:
            nf = 0
            for u in self.OrthBase:
                nf += self.project(f, u)
            nf = f - nf
            norm_sq = self.inner(nf, nf)
            if abs(float(norm_sq)) < self._zero_tol:
                raise ValueError(
                    "Cannot form an orthonormal basis from linearly dependent basis functions."
                )
            F = self.expand(nf / self.sqrt(norm_sq))
            self.OrthBase.append(F)
        self.num_base = len(self.OrthBase)

    def SetOrthBase(self, base):
        """
        Sets the orthonormal basis to be the given `base`.
        """
        if not isinstance(base, list) or base == []:
            raise ValueError("`base` must be a non-empty list.")
        self.OrthBase = base
        self.num_base = len(self.OrthBase)

    def Series(self, f):
        """
        Given a function `f`, this method finds and returns the
        coefficients of the	series that approximates `f` as a
        linear combination of the elements of the orthogonal basis.
        """
        if self.OrthBase == []:
            raise ValueError("No orthonormal basis is available. Call `FormBasis()` first.")
        cfs = []
        for b in self.OrthBase:
            cfs.append(self.inner(f, b))
        return cfs
