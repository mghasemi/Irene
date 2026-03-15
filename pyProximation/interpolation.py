class Interpolation:
    """
    The ``Interpolation`` class provides polynomial interpolation routines
    in multi variate case.

    `var` is the list of symbolic variables and `env` is the the symbolic tool.
    """

    def __init__(self, var, env="sympy"):
        """
        `var` is the list of symbolic variables.
        """
        if not isinstance(var, list) or var == []:
            raise TypeError("`var` must be a non-empty list of symbolic variables.")
        if env != 'sympy':
            raise ValueError("Only 'sympy' is supported as symbolic environment.")
        self.Env = 'sympy'
        self.Vars = var
        self.num_vars = len(var)
        self.degree = 1

    def Interpolate(self, points, vals):
        """
        Takes a list of points `points` and corresponding list of values `vals` and
        return the interpolant.

        Since in multivariate case, there is a constraint on the number of points, it
        checks for the valididty of the input. In case of failure, describes the type
        of error occured according to the inputs.
        """
        if not isinstance(points, list) or points == []:
            raise ValueError("Unable to interpolate over an empty set of points.")
        if not isinstance(vals, list):
            raise TypeError("`vals` must be provided as a list.")
        if len(points) != len(vals):
            raise ValueError("The number of points does not match the number of values.")
        # check dimension consistency
        for point in points:
            if not isinstance(point, (list, tuple)):
                raise TypeError("Each interpolation point must be a list or tuple.")
            if len(point) != self.num_vars:
                raise ValueError(
                    "Each interpolation point must be %d-dimensional." % (self.num_vars)
                )
        self.num_points = len(points)
        # the least number of extra points required
        mnp = self.MinNumPoints()
        # check the sufficiency of points
        if mnp > 0:
            raise ValueError("At least %d more points and values are required." % (mnp))
        self.Points = [tuple(point) for point in points]
        self.Monomials()
        delta = self.Delta()
        if delta == 0:
            raise ValueError("Cannot find a unique interpolant from the given points.")
        p = 0
        for i in range(self.num_points):
            p += vals[i] * self.Delta(i) / delta
        return p

    def MinNumPoints(self):
        """
        Returns the minimum number of points still required.
        """
        from scipy.special import binom
        m = self.num_vars
        n = 1
        r = binom(m + n, n)
        while r < self.num_points:
            n += 1
            r = binom(m + n, n)
        self.degree = n
        return int(r - self.num_points)

    def Monomials(self):
        """
        Generates the minimal set of monomials for interpolation.
        """
        from itertools import product
        B = []
        for o in product(range(self.degree + 1), repeat=self.num_vars):
            if sum(o) <= self.degree:
                T_ = 1
                for idx in range(self.num_vars):
                    T_ *= self.Vars[idx]**o[idx]
                B.append(T_)
        self.MonoBase = B

    def Delta(self, idx=-1):
        """
        Construct the matrix corresponding to `idx`'th point, if `idx>0`
        Otherwise returns the discriminant.
        """
        if idx < -1 or idx >= self.num_points:
            raise IndexError("`idx` must be -1 or the index of an interpolation point.")
        from sympy import Matrix
        M = []
        for j in range(self.num_points):
            pnt = self.Points[j]
            row = []
            for mon in self.MonoBase:
                if j != idx:
                    chng = {self.Vars[i]: pnt[i] for i in range(self.num_vars)}
                    row.append(mon.subs(chng))
                else:
                    row.append(mon)
            M.append(row)
        Mtx = Matrix(M)
        return Mtx.det()
