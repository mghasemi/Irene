from .base import Foundation


class Measure(Foundation):
    """
    An instance of this class is a measure on a given set `supp`. The support is either
            + a python variable of type `set`, or
            + a list of tuples which represents a box in euclidean space.
    Initializes a measure object according to the inputs:
            + *dom* must be either
                    - a list of 2-tuples
                    - a non-empty dictionary
            + *w* must be a
                    - a function if `dom` defines a region
                    - left blank (None) if `dom` is a dictionary
    """

    def __init__(self, dom, w=None):
        """
        Initializes a measure object according to the inputs:
                +`dom` must be either
                        - a list of 2-tuples
                        - a non-empty dictionary
                + `w` must be a
                        - a function if `dom` defines a region
                        - left blank (None) if `dom` is a dictionary
        """
        self.ErrorMsg = ""
        self.dim = 0
        self.card = 0
        if not self.check(dom, w):
            raise ValueError(self.ErrorMsg)

    def _call_on_point(self, func, point):
        try:
            return func(point)
        except TypeError as original_error:
            if isinstance(point, tuple):
                try:
                    return func(*point)
                except TypeError as expanded_error:
                    raise TypeError(
                        "Failed to evaluate the integrand on support point %r. "
                        "For tuple-valued support, provide a callable that accepts "
                        "either a single tuple or one positional argument per coordinate."
                        % (point,)
                    ) from expanded_error
            raise TypeError(
                "Failed to evaluate the integrand on support point %r." % (point,)
            ) from original_error

    def boxCheck(self, B):
        """
        Checks the structure of the box *B*.
        Returns `True` id `B` is a list of 2-tuples, otherwise it
        returns `False`.
        """
        flag = True
        for interval in B:
            flag = flag and isinstance(interval, tuple) and (len(interval) == 2)
            if flag:
                flag = flag and (interval[0] <= interval[1])
        return flag

    def check(self, dom, w):
        """
        Checks the input types and their consistency, according to the
        *__init__* arguments.
        """
        from numbers import Number
        if isinstance(dom, list):
            if not self.boxCheck(dom):
                self.ErrorMsg = "Each support interval must be a 2-tuple `(lower, upper)` with `lower <= upper`."
                return False
            self.DomType = "box"
            self.dim = len(dom)
            self.supp = dom
            if w is None:
                self.ErrorMsg = "A continuous measure requires a weight function or a numeric constant."
                return False
            if callable(w):
                self.weight = w
            elif isinstance(w, Number):
                self.weight = lambda *x: w
            else:
                self.ErrorMsg = "Weight must be either a callable or a numeric constant."
                return False
        elif isinstance(dom, dict):
            if len(dom) == 0:
                self.ErrorMsg = "A discrete measure cannot have an empty support."
                return False
            self.DomType = "set"
            self.card = len(dom)
            self.supp = tuple(dom.keys())
            self.weight = dom
        else:
            self.ErrorMsg = "The support must be either a list of intervals or a dictionary of point masses."
            return False
        return True

    def measure(self, S):
        """
        Returns the measure of the set `S`.
        `S` must be a list of 2-tuples.
        """
        m = 0
        if self.DomType == "set":
            if not isinstance(S, (set, list, tuple)):
                raise TypeError("A discrete subset must be a `set`, `list`, or `tuple`.")
            for p in S:
                if p in self.supp:
                    m += self.weight[p]
        else:
            if (not isinstance(S, list)) or (not self.boxCheck(S)):
                raise TypeError("A continuous subset must be a list of 2-tuples.")
            from scipy import integrate
            m = integrate.nquad(self.weight, S)[0]
        return m

    def integral(self, f):
        """
        Returns the integral of `f` with respect to the currwnt measure
        over the support.
        """
        m = 0
        if self.DomType == "set":
            if not isinstance(f, dict) and not callable(f):
                raise TypeError("For a discrete measure, the integrand must be a `dict` or a callable.")
            if isinstance(f, dict):
                for p in self.supp:
                    if p in f:
                        m += self.weight[p] * f[p]
            else:
                for p in self.supp:
                    m += self.weight[p] * self._call_on_point(f, p)
        else:
            if not callable(f):
                raise TypeError("For a continuous measure, the integrand must be callable.")
            from scipy import integrate
            fw = lambda *x: f(*x) * self.weight(*x)
            m = integrate.nquad(fw, self.supp)[0]
        return m

    def norm(self, p, f):
        """
        Computes the norm-`p` of the `f` with respect to the current measure.
        """
        from math import pow
        if p <= 0:
            raise ValueError("The norm order `p` must be positive.")
        if not callable(f):
            raise TypeError("The norm expects a callable integrand.")
        absfp = lambda *x: pow(abs(f(*x)), p)
        return pow(self.integral(absfp), 1. / p)

    def sample(self, num):
        """
        Samples from the support according to the measure.

        """
        if not isinstance(num, int) or num < 1:
            raise ValueError("Sample size must be a positive integer.")
        if self.DomType == 'box':
            from math import ceil, pow
            from itertools import product
            import random
            from random import uniform
            SubRegions = {}
            NumSample = {}
            points = []
            n = int(ceil(pow(num, (1. / self.dim))))
            delta = [(r[1] - r[0]) / float(n) for r in self.supp]
            for o in product(range(n), repeat=self.dim):
                SubRegions[o] = [(self.supp[i][0] + o[i] * delta[i], self.supp[i]
                                  [0] + (o[i] + 1) * delta[i]) for i in range(self.dim)]
            numpnts = max(num, len(SubRegions))
            muSupp = self.measure(self.supp)
            if muSupp <= 0:
                raise ValueError("Cannot sample from a continuous measure with non-positive total mass.")
            for o in SubRegions:
                NumSample[o] = int(ceil(numpnts * self.measure(SubRegions[o]) / float(muSupp)))
            for o in NumSample:
                pnts = []
                while len(pnts) < NumSample[o]:
                    v = []
                    for rng in SubRegions[o]:
                        v.append(uniform(rng[0], rng[1]))
                    v = tuple(v)
                    if v not in pnts:
                        pnts.append(v)
                points += pnts
            return random.sample(points, num)
        else:
            weights = [self.weight[p] for p in self.supp]
            if any(weight < 0 for weight in weights):
                raise ValueError("Cannot sample from a discrete measure with negative weights.")
            TotM = sum(weights)
            if TotM <= 0:
                raise ValueError("Cannot sample from a discrete measure with non-positive total mass.")
            from random import choices
            return choices(self.supp, weights=weights, k=num)
