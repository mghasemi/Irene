"""A module for working with commutative semigroups and their differential algebras.

This module provides classes and functions for working with commutative semigroups and their differential algebras.

A semigroup is a set S together with an associative binary operation on S.
A commutative semigroup is a semigroup in which the binary operation is commutative.

A semigroup algebra is a vector space over a field with a basis consisting of the elements of a semigroup
equipped with multiplication compatible with the semigroup's operation.

This module provides the following classes:

* CommutativeSemigroup: A class representing a commutative semigroup.
* AtomicSGElement: A class representing an atomic element of a semigroup algebra.
* SemigroupAlgebraElement: A class representing an element of a semigroup algebra.
* SemigroupAlgebra: A class representing a semigroup algebra.

This module also provides the following functions:

* _degree: Computes the degree of a given expression.
* diff: Computes the derivative of an expression in a semigroup algebra.
"""

from itertools import combinations_with_replacement
from typing import Any, Iterator

from sympy import Expr
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.free_groups import free_group, FreeGroupElement


def _degree(xprsn: Any) -> int:
    """Computes the degree of a given expression.

    The degree of an expression is the sum of the absolute values of the exponents of its terms.

    Args:
        xprsn: The expression whose degree is to be computed.

    Returns:
        int: The degree of the expression.
    """
    dg = 0
    if hasattr(xprsn, 'array_form'):
        for _ in xprsn.array_form:
            dg += abs(_[1])
    return dg


class CommutativeSemigroup(object):
    """A class representing a commutative semigroup.

    A semigroup is a set S together with an associative binary operation on S.
    A commutative semigroup is a semigroup in which the binary operation is commutative.

    Attributes:
        gens (list): The generators of the semigroup.
        is_semigroup (bool): True if the semigroup is a semigroup, False otherwise.
        is_abelian (bool): True if the semigroup is abelian, False otherwise.
        rels (list): The relations of the semigroup.
        aux_rels (list): The auxiliary relations of the semigroup.
        inverses (dict): The inverses of the generators of the semigroup.
        max_deg (int): The maximum degree of the relations of the semigroup.
        edges (list): The edges of the lattice of the semigroup.
        vertices (dict): The vertices of the lattice of the semigroup.
        symbols (list): The symbols of the semigroup.
    """

    def __init__(self, gens: list, is_semigroup: bool = True, is_abelian: bool = True):
        """Initializes a new instance of the CommutativeSemigroup class.

        Args:
            gens (list): The generators of the semigroup.
            is_semigroup (bool): True if the semigroup is a semigroup, False otherwise.
            is_abelian (bool): True if the semigroup is abelian, False otherwise.
        """
        self.is_semigroup = is_semigroup
        self.is_abelian = is_abelian
        self.rels = []
        self.aux_rels = []
        self.inverses = dict()
        self.max_deg = 0
        self.edges = None
        self.vertices = None
        self.symbols = gens
        if not isinstance(gens, list):
            raise TypeError("'gens' must be a list")
        self.F_gens = free_group(gens)
        self.FreeGroup = self.F_gens[0]
        self.generators = list(self.F_gens[1:])
        self.generators.sort()
        self.num_gens = len(self.generators)
        for e in self.generators:
            self.__setattr__(e.ext_rep[0].name, e)
        if self.is_abelian:
            for i in range(self.num_gens):
                for j in range(i + 1, self.num_gens):
                    self.rels.append(self.generators[i] * self.generators[j] * (self.generators[i] ** -1) * (
                            self.generators[j] ** -1))
        self.G = FpGroup(self.FreeGroup, self.rels)

    def add_relations(self, rels: list):
        """Adds relations to the semigroup.

        Args:
            rels (list): The relations to add to the semigroup.
        """
        if not isinstance(rels, list):
            raise TypeError("'rels' must be a list")
        self.rels += rels
        self.max_deg = max([max([c[1] for c in _.array_form]) for _ in rels])
        self.aux_rels += rels
        self.G = FpGroup(self.FreeGroup, self.rels)
        self._inverse()

    def _lst2elmnt(self, lst: list) -> Expr:
        """Converts a list of tuples to an element of the semigroup.

        Args:
            lst (list): The list of tuples to convert.

        Returns:
            sympy.Expr: The element of the semigroup.
        """
        xprsn = self.G.identity
        for _ in lst:
            xprsn *= _[0] ** _[1]
        return xprsn

    def _lst_prod(self, lst: list) -> Expr:
        """Computes the product of a list of elements of the semigroup.

        Args:
            lst (list): The list of elements of the semigroup to multiply.

        Returns:
            sympy.Expr: The product of the elements of the semigroup.
        """
        lmnt = self.G.identity
        for _ in lst:
            lmnt *= self.G.reduce(_)
        return self.G.reduce(lmnt)

    def _inverse(self):
        """Computes the inverses of the generators of the semigroup."""
        for e in self.aux_rels:
            for idx in range(len(e.array_form)):
                lst = list(e.array_form)
                lst = [(self.__getattribute__(_[0].name), _[1]) for _ in lst]
                symbl = e.array_form[idx][0].name
                expnt = e.array_form[idx][1]
                lst[idx] = (self.__getattribute__(symbl), expnt - 1)
                xprsn = self._lst2elmnt(lst)
                if symbl in self.inverses:
                    self.inverses[symbl] = min(self.inverses[symbl], xprsn) * self.G.identity
                else:
                    self.inverses[symbl] = xprsn * self.G.identity

    def _sort_exp(self, expr: Expr) -> Expr:
        """Sorts the terms of an expression in the semigroup.

        Args:
            expr (sympy.Expr): The expression to sort.

        Returns:
            sympy.Expr: The sorted expression.
        """
        arr = expr.array_form
        sorted_dict = {_: 0 for _ in self.generators}
        for _ in arr:
            sorted_dict[self.__getattribute__(_[0].name)] += _[1]
        lst = [(_, sorted_dict[_]) for _ in sorted_dict]
        return self._lst2elmnt(lst)

    def _reduce(self, xprsn: Expr, recur: bool = True) -> Expr:
        """Reduces an expression in the semigroup.

        Args:
            xprsn (sympy.Expr): The expression to reduce.
            recur (bool): True if the reduction should be recursive, False otherwise.

        Returns:
            sympy.Expr: The reduced expression.
        """
        arr_frm = xprsn.array_form
        new_arr_frm = list()
        for _ in arr_frm:
            if (_[1] < 0) and (_[0].name in self.inverses):
                rplcmnt = (self.inverses[_[0].name], -_[1])
                new_arr_frm.append(rplcmnt)
            else:
                new_arr_frm.append((self.__getattribute__(_[0].name), _[1]))
        idnt = self._sort_exp(self._lst2elmnt(new_arr_frm))
        if recur:
            cndd = self.G.reduce(idnt)
            return self._reduce(cndd, recur=False)
        else:
            return idnt

    def degree(self, expr) -> int:
        """Computes the degree of an expression in the semigroup.

        The degree of an expression is the sum of the absolute values of the exponents of its terms.

        Args:
            expr : The expression whose degree is to be computed.

        Returns:
            int: The degree of the expression.
        """
        xprsn = self._reduce(expr)
        dg = 0
        for _ in xprsn.array_form:
            dg += abs(_[1])
        return dg

    def positive_exp(self, expr: Expr) -> bool:
        """Checks if an expression in the semigroup has only positive exponents.

        Args:
            expr (sympy.Expr): The expression to check.

        Returns:
            bool: True if the expression has only positive exponents, False otherwise.
        """
        xprsn = self._reduce(expr)
        for _ in xprsn.array_form:
            if _[1] < 0:
                return False
        return True

    def identity(self) -> Expr:
        return self.G.identity

    def lattice_edges(self, degree: int) -> None:
        """Computes the edges of the lattice of the semigroup.

        Args:
            degree (int): The degree of the lattice.
        """
        elements = []
        lst = list(combinations_with_replacement(self.generators + [self.G.identity], degree))
        for tpl in lst:
            lmnt = self._reduce(self._lst_prod(tpl))
            if (self.degree(lmnt) <= degree) and (lmnt not in elements):
                elements.append(lmnt)
        elements.sort()
        self.edges = elements

    def lattice_vertices(self) -> None:
        """Computes the vertices of the lattice of the semigroup.

        The vertices of the lattice are the elements of the semigroup that are generated by the edges of the lattice.
        """
        if self.edges is None:
            raise ValueError("Lattice edges must be computed before computing vertices")
        all_vs = dict()
        N = len(self.edges)
        for i in range(N):
            for j in range(i, N):
                vi = self.edges[i]
                vj = self.edges[j]
                vij = self._reduce(vi * vj)
                if vij not in all_vs:
                    all_vs[vij] = set([])
                all_vs[vij].add((vi, vj))
        self.vertices = all_vs

    def element_sub_lattice(self, elm: Expr, ex: set = None) -> set:
        """Computes the sublattice of the lattice of the semigroup generated by an element.

        Args:
            elm (sympy.Expr): The element of the semigroup to generate the sublattice from.
            ex (set): The set of elements to exclude from the sublattice.

        Returns:
            set: The sublattice of the lattice of the semigroup generated by the element.
        """
        if ex is None:
            ex = set()
        _elm = self._reduce(elm)
        if self.edges is None:
            self.lattice_edges(self.degree(_elm))
        if self.vertices is None:
            self.lattice_vertices()
        vs = ex
        for cmp in self.vertices[_elm] - ex:
            vs = vs.union(self.element_sub_lattice(cmp[0], ex=vs))
        return vs


class AtomicSGElement(object):
    """An atomic element of a semigroup algebra.

    An atomic element is an element of the semigroup algebra that is not a sum of other elements.

    Attributes:
        semigroup (CommutativeSemigroup): The semigroup of the element.
        symbol (str): The symbol of the element.
        content (list): The content of the element.
    """

    def __init__(self, semigroup: CommutativeSemigroup, element: str):
        """Initializes a new instance of the AtomicSGElement class.

        Args:
            semigroup (CommutativeSemigroup): The semigroup of the element.
            element (str): The symbol of the element.
        """
        if not isinstance(semigroup, CommutativeSemigroup):
            raise TypeError("'semigroup' should be an instance of 'CommutativeSemigroup'.")
        self.semigroup = semigroup
        self.symbol = element
        if element not in self.semigroup.symbols:
            raise KeyError("'%s' is not an element of the given semigroup" % element)
        self.__setattr__(element, (1., self.semigroup.__getattribute__(element)))
        self.content = [(1., self.semigroup.__getattribute__(element))]

    def constant(self) -> float:
        return self[self.semigroup.G.identity]

    def support(self) -> list:
        return [self.content[0][1]]

    def LC(self) -> float:
        """Returns the leading coefficient of the element.

        The leading coefficient of an element is the coefficient of the term with the highest degree.

        Returns:
            float: The leading coefficient of the element.
        """
        return 1.

    def LM(self) -> Expr:
        """Returns the leading monomial of the element.

        The leading monomial of an element is the monomial with the highest degree.

        Returns:
            sympy.Expr: The leading monomial of the element.
        """
        return self.content[0][1]

    def LT(self) -> "SemigroupAlgebraElement":
        """Returns the leading term of the element.

        The leading term of an element is the term with the highest degree.

        Returns:
            SemigroupAlgebraElement: The leading term of the element.
        """
        return SemigroupAlgebraElement(self.content, self.semigroup)

    def lt_divisible_by(self, expr: Any) -> bool:
        """Checks if the leading term of the element is divisible by the leading term of another element or expression.

        Args:
            expr (int, float, AtomicSGElement, SemigroupAlgebraElement): The element or expression to check divisibility by.

        Returns:
            bool: True if the leading term of the element is divisible by the leading term of the other element or expression, False otherwise.
        """
        if not isinstance(expr, (int, float, AtomicSGElement, SemigroupAlgebraElement)):
            raise ArithmeticError("Division type missmatch!")
        if isinstance(expr, (int, float)):
            return True
        else:
            p = self.LM()
            q = expr.LM()
            p_d_q = self.semigroup.positive_exp(p * q)
            return p_d_q

    def divide(self, fs: list) -> tuple[list, "SemigroupAlgebraElement"]:
        """Divides the element by a list of elements or expressions.

        Args:
            fs (list): The list of elements or expressions to divide by.

        Returns:
            tuple: A tuple containing the quotient and remainder of the division.
        """
        p = SemigroupAlgebraElement(self.content, self.semigroup)
        return p.divide(fs)

    def __add__(self, other):
        content = []
        if not isinstance(other, (AtomicSGElement, SemigroupAlgebraElement, int, float)):
            raise TypeError("An 'AtomicSGElement' can not be added with '%s' object" % type(other))
        if isinstance(other, (int, float)):
            content = [(other, self.semigroup.G.identity), self.content[0]]
        elif isinstance(other, AtomicSGElement):
            if other.content[0][1] == self.content[0][1]:
                content = [(self.content[0][0] + other.content[0][0], self.content[0][1])]
            else:
                content = self.content + other.content
        elif isinstance(other, SemigroupAlgebraElement):
            content = []
            keys = set([])
            slf_k = self.content[0][1]
            for _ in other.content:
                keys.add(_[1])
                if slf_k == _[1]:
                    content.append((self.content[0][0] + _[0], _[1]))
                else:
                    content.append(_)
            if slf_k not in keys:
                content += self.content
        return SemigroupAlgebraElement(content, self.semigroup)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        content = [(-_[0], _[1]) for _ in self.content]
        return SemigroupAlgebraElement(content, self.semigroup)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        content = [(-_[0], _[1]) for _ in self.content]
        return SemigroupAlgebraElement(content, self.semigroup).__add__(other)

    def __mul__(self, other):
        content = []
        if not isinstance(other, (AtomicSGElement, SemigroupAlgebraElement, int, float)):
            raise TypeError("An 'AtomicSGElement' can not be multiplied with '%s' object" % type(other))
        if isinstance(other, (int, float)):
            content = [(other * self.content[0][0], self.content[0][1])]
        elif isinstance(other, AtomicSGElement):
            content = [(self.content[0][0] * other.content[0][0],
                        self.semigroup._reduce(self.content[0][1] * other.content[0][1]))]
        elif isinstance(other, SemigroupAlgebraElement):
            content = [
                (self.content[0][0] * _[0], self.semigroup._reduce(self.content[0][1] * self.semigroup._reduce(_[1])))
                for _ in other.content]
        return SemigroupAlgebraElement(content, self.semigroup)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, p: int):
        content = [(self.content[0][0] ** p, self.content[0][1] ** p)]
        return SemigroupAlgebraElement(content, self.semigroup)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, AtomicSGElement, SemigroupAlgebraElement)):
            raise ArithmeticError("Unsupported division.")
        if isinstance(other, (int, float)):
            return SemigroupAlgebraElement([(1. / other, self.content[0][1])], self.semigroup)
        q, r = self.divide([other])
        if isinstance(r, SemigroupAlgebraElement) and r.content:
            return None
        return q[0]

    def __floordiv__(self, other):
        if not isinstance(other, (int, float, AtomicSGElement, SemigroupAlgebraElement)):
            raise ArithmeticError("Unsupported division.")
        if isinstance(other, (int, float)):
            return SemigroupAlgebraElement([(1. / other, self.content[0][1])], self.semigroup)
        q, _ = self.divide([other])
        return q[0]

    def __mod__(self, other):
        if not isinstance(other, (int, float, AtomicSGElement, SemigroupAlgebraElement)):
            raise ArithmeticError("Unsupported division.")
        if isinstance(other, (int, float)):
            return SemigroupAlgebraElement([(0., self.semigroup.G.identity)], self.semigroup)
        _, r = self.divide([other])
        return r

    def __lt__(self, other) -> bool:
        if isinstance(other, AtomicSGElement):
            return self.content[0][1] < other.content[0][1]
        elif isinstance(other, (int, float)):
            return False
        elif isinstance(other, SemigroupAlgebraElement):
            M = other._max_content()
            return self.content[0][1] < M
        raise TypeError(f"Objects of type {type(other)} can not be compared with AtomicSGElement")

    def __le__(self, other) -> bool:
        if isinstance(other, AtomicSGElement):
            return self.content[0][1] <= other.content[0][1]
        elif isinstance(other, (int, float)):
            if self.content[0][1] == self.semigroup.G.identity:
                return True
            return False
        elif isinstance(other, SemigroupAlgebraElement):
            M = other._max_content()
            return self.content[0][1] <= M
        raise TypeError(f"Objects of type {type(other)} can not be compared with AtomicSGElement")

    def __gt__(self, other) -> bool:
        if isinstance(other, AtomicSGElement):
            return self.content[0][1] > other.content[0][1]
        elif isinstance(other, (int, float)):
            if self.content[0][1] == self.semigroup.G.identity:
                return False
            return True
        elif isinstance(other, SemigroupAlgebraElement):
            M = other._max_content()
            return self.content[0][1] > M
        raise TypeError(f"Objects of type {type(other)} can not be compared with AtomicSGElement")

    def __ge__(self, other) -> bool:
        if isinstance(other, AtomicSGElement):
            return self.content[0][1] >= other.content[0][1]
        elif isinstance(other, (int, float)):
            return True
        elif isinstance(other, SemigroupAlgebraElement):
            M = other._max_content()
            return self.content[0][1] >= M
        raise TypeError(f"Objects of type {type(other)} can not be compared with AtomicSGElement")

    def __eq__(self, other) -> bool:
        term = self.content[0]
        if isinstance(other, (int, float)):
            return (term[1] == self.semigroup.G.identity) and (term[0] == other)
        elif isinstance(other, AtomicSGElement):
            return term == other.content[0]
        elif isinstance(other, SemigroupAlgebraElement):
            return len(other.content) == 1 and term == other.content[0]
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __getitem__(self, item: Any) -> float:
        if isinstance(item, FreeGroupElement):
            if self.content[0][1] == item:
                return self.content[0][0]
        elif isinstance(item, AtomicSGElement):
            if self.content[0][1] == item.content[0][1]:
                return self.content[0][0]
        elif isinstance(item, SemigroupAlgebraElement):
            if len(item.content) > 1:
                raise TypeError("Cannot find the coefficient of the provided element.")
            else:
                if self.content[0][1] == item.content[0][1]:
                    return self.content[0][0]
        return 0.

    def __bool__(self) -> bool:
        return bool(self.content and self.content[0][0] != 0.)

    def __str__(self) -> str:
        return "%.3f * %s" % (self.content[0][0], self.content[0][1])


class SemigroupAlgebraElement(object):
    """An element of a semigroup algebra.

    A semigroup algebra is a vector space over a field with a basis consisting of the elements of a semigroup.

    Attributes:
        content (list): The content of the element.
        semigroup (CommutativeSemigroup): The semigroup of the element.
    """

    def __init__(self, terms: list, semigroup: CommutativeSemigroup):
        """Initializes a new instance of the SemigroupAlgebraElement class.

        Args:
            terms (list): The terms of the element.
            semigroup (CommutativeSemigroup): The semigroup of the element.
        """
        self.content = [(_[0], semigroup._reduce(_[1])) for _ in terms if _[0] != 0.]
        self.semigroup = semigroup

    def __iter__(self) -> Iterator[tuple[float, Expr]]:
        return iter(self.content)

    def __bool__(self) -> bool:
        return bool(self.content)

    def _max_content(self) -> Expr:
        """Returns the maximum content of the element.

        The maximum content of an element is the content of the term with the highest degree.

        Returns:
            sympy.Expr: The maximum content of the element.
        """
        M = max([_[1] for _ in self.content], key=lambda x: (self.semigroup.degree(x), str(x)))
        return M

    def constant(self) -> float:
        return self[self.semigroup.G.identity]

    def support(self) -> list[Expr]:
        sprt = list()
        for _ in self.content:
            sprt.append(_[1])
        return sprt

    def LC(self) -> float:
        """Returns the leading coefficient of the element.

        The leading coefficient of an element is the coefficient of the term with the highest degree.

        Returns:
            float: The leading coefficient of the element.
        """
        M = self._max_content()
        for _ in self.content:
            if _[1] == M:
                return _[0]
        return 0.

    def LM(self) -> Expr:
        """Returns the leading monomial of the element.

        The leading monomial of an element is the monomial with the highest degree.

        Returns:
            sympy.Expr: The leading monomial of the element.
        """
        return self._max_content()

    def LT(self) -> "SemigroupAlgebraElement":
        """Returns the leading term of the element.

        The leading term of an element is the term with the highest degree.

        Returns:
            SemigroupAlgebraElement: The leading term of the element.
        """
        return SemigroupAlgebraElement([(self.LC(), self._max_content())], self.semigroup)

    def lt_divisible_by(self, expr: Any) -> bool:
        """Checks if the leading term of the element is divisible by the leading term of another element or expression.

        Args:
            expr (int, float, AtomicSGElement, SemigroupAlgebraElement): The element or expression to check divisibility by.

        Returns:
            bool: True if the leading term of the element is divisible by the leading term of the other element or expression, False otherwise.
        """
        if not isinstance(expr, (int, float, AtomicSGElement, SemigroupAlgebraElement)):
            raise ArithmeticError("Division type missmatch!")
        if isinstance(expr, (int, float)):
            return True
        else:
            p = self.LM()
            q = expr.LM()
            p_d_q = self.semigroup.positive_exp(p * q)
            return p_d_q

    def divide(self, fs: list) -> tuple[list, "SemigroupAlgebraElement"]:
        """Divides the element by a list of elements or expressions.

        Args:
            fs (list): The list of elements or expressions to divide by.

        Returns:
            tuple: A tuple containing the quotient and remainder of the division.
        """
        s = len(fs)
        qs = [0.] * s
        r = 0.
        p = self
        while p.content:
            i = 0
            division_occurred = False
            while i < s:
                if not p.content:
                    break
                if p.lt_divisible_by(fs[i].LT()):
                    mono_div = self.semigroup._reduce(p.LM() * fs[i].LM() ** -1)
                    if _degree(mono_div) > _degree(p.LM()):
                        break
                    div = SemigroupAlgebraElement(
                        [(p.LC() / fs[i].LC(), self.semigroup._reduce(p.LM() * fs[i].LM() ** -1))], self.semigroup)
                    qs[i] = qs[i] + div
                    p = p - div * fs[i]
                    division_occurred = True
                else:
                    i += 1
            if not division_occurred:
                r = r + p.LT()
                p = p - p.LT()
        return qs, r

    def __truediv__(self, other):
        if not isinstance(other, (int, float, AtomicSGElement, SemigroupAlgebraElement)):
            raise ArithmeticError("Unsupported division.")
        if isinstance(other, (int, float)):
            return SemigroupAlgebraElement([(_[0] / other, _[1]) for _ in self.content], self.semigroup)
        q, r = self.divide([other])
        if isinstance(r, SemigroupAlgebraElement) and r.content:
            return None
        return q[0]

    def __floordiv__(self, other):
        if not isinstance(other, (int, float, AtomicSGElement, SemigroupAlgebraElement)):
            raise ArithmeticError("Unsupported division.")
        if isinstance(other, (int, float)):
            return SemigroupAlgebraElement([(_[0] / other, _[1]) for _ in self.content], self.semigroup)
        q, _ = self.divide([other])
        return q[0]

    def __mod__(self, other):
        if not isinstance(other, (int, float, AtomicSGElement, SemigroupAlgebraElement)):
            raise ArithmeticError("Unsupported division.")
        if isinstance(other, (int, float)):
            return SemigroupAlgebraElement([(0., self.semigroup.G.identity)], self.semigroup)
        _, r = self.divide([other])
        return r

    def __neg__(self):
        content = [(-_[0], _[1]) for _ in self.content]
        return SemigroupAlgebraElement(content, self.semigroup)

    def __add__(self, other):
        content = []
        if not isinstance(other, (AtomicSGElement, SemigroupAlgebraElement, int, float)):
            raise TypeError("An 'SemigroupAlgebraElement' can not be added with '%s' object" % type(other))
        if isinstance(other, (int, float)):
            content = []
            temp_keys = []
            for _ in self.content:
                temp_keys.append(_[1])
                if _[1] == self.semigroup.G.identity:
                    content.append((_[0] + other, _[1]))
                else:
                    content.append(_)
            if self.semigroup.G.identity not in temp_keys:
                content.append((other, self.semigroup.G.identity))
        elif isinstance(other, AtomicSGElement):
            content = []
            temp_keys = []
            for _ in self.content:
                temp_keys.append(_[1])
                if _[1] == other.content[0][1]:
                    content.append((_[0] + other.content[0][0], _[1]))
                else:
                    content.append(_)
            if other.content[0][1] not in temp_keys:
                content.append((other.content[0][0], other.content[0][1]))
        elif isinstance(other, SemigroupAlgebraElement):
            content = []
            dict1 = {_[1]: _[0] for _ in self.content}
            dict2 = {_[1]: _[0] for _ in other.content}
            all_keys = set(dict1.keys()).union(set(dict2.keys()))
            for k in all_keys:
                content.append((dict1.get(k, 0) + dict2.get(k, 0), k))
        return SemigroupAlgebraElement(content, self.semigroup)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        content = [(-_[0], _[1]) for _ in self.content]
        return SemigroupAlgebraElement(content, self.semigroup).__add__(other)

    def __mul__(self, other):
        content = []
        if not isinstance(other, (AtomicSGElement, SemigroupAlgebraElement, int, float)):
            raise TypeError("An 'SemigroupAlgebraElement' can not be added with '%s' object" % type(other))
        if isinstance(other, (int, float)):
            content = [(other * _[0], self.semigroup._reduce(_[1])) for _ in self.content]
        elif isinstance(other, AtomicSGElement):
            content = [(other.content[0][0] * _[0], self.semigroup._reduce(other.content[0][1] * _[1])) for _ in
                       self.content]
        elif isinstance(other, SemigroupAlgebraElement):
            keys1 = [self.semigroup._reduce(_[1]) for _ in self.content]
            keys2 = [self.semigroup._reduce(_[1]) for _ in other.content]
            mul_keys = set([])
            for i in keys1:
                for j in keys2:
                    mul_keys.add(self.semigroup._reduce(i * j))
            dict_content = {_: 0 for _ in mul_keys}
            for t1 in self.content:
                for t2 in other.content:
                    k = self.semigroup._reduce(t1[1] * t2[1])
                    cf = t1[0] * t2[0]
                    dict_content[k] += cf
            content = [(dict_content[_], _) for _ in dict_content]
        return SemigroupAlgebraElement(content, self.semigroup)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, p: int):
        if (type(p) is not int) or (p < 0):
            raise ValueError(f"The power {p} must be a positive integer.")
        elm = 1.
        for _ in range(p):
            elm = elm * self
        return elm

    def __lt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return False
        elif isinstance(other, AtomicSGElement):
            return self._max_content() < other.content[0][1]
        elif isinstance(other, SemigroupAlgebraElement):
            return self._max_content() < other._max_content()
        raise TypeError(f"Objects of type {type(other)} can not be compared with SemigroupAlgebraElement")

    def __le__(self, other) -> bool:
        if isinstance(other, (int, float)):
            if self._max_content() == self.semigroup.G.identity:
                return True
            return False
        elif isinstance(other, AtomicSGElement):
            return self._max_content() <= other.content[0][1]
        elif isinstance(other, SemigroupAlgebraElement):
            return self._max_content() <= other._max_content()
        raise TypeError(f"Objects of type {type(other)} can not be compared with SemigroupAlgebraElement")

    def __gt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            if self._max_content() == self.semigroup.G.identity:
                return False
            return True
        elif isinstance(other, AtomicSGElement):
            return self._max_content() > other.content[0][1]
        elif isinstance(other, SemigroupAlgebraElement):
            return self._max_content() > other._max_content()
        raise TypeError(f"Objects of type {type(other)} can not be compared with SemigroupAlgebraElement")

    def __ge__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return True
        elif isinstance(other, AtomicSGElement):
            return self._max_content() >= other.content[0][1]
        elif isinstance(other, SemigroupAlgebraElement):
            return self._max_content() >= other._max_content()
        raise TypeError(f"Objects of type {type(other)} can not be compared with SemigroupAlgebraElement")

    def __eq__(self, other) -> bool:
        if isinstance(other, (int, float)):
            if not self.content:
                return other == 0
            return len(self.content) == 1 and self.content[0][1] == self.semigroup.G.identity and self.content[0][0] == other
        elif isinstance(other, AtomicSGElement):
            return len(self.content) == 1 and self.content[0] == other.content[0]
        elif isinstance(other, SemigroupAlgebraElement):
            if len(self.content) != len(other.content):
                return False
            lhs = {mono: coeff for coeff, mono in self.content}
            rhs = {mono: coeff for coeff, mono in other.content}
            return lhs == rhs
        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __getitem__(self, item: Any) -> float:
        if isinstance(item, FreeGroupElement):
            for trm in self.content:
                if trm[1] == item:
                    return trm[0]
        elif isinstance(item, AtomicSGElement):
            for trm in self.content:
                if trm[1] == item.content[0][1]:
                    return trm[0]
        elif isinstance(item, SemigroupAlgebraElement):
            if len(item.content) > 1:
                raise TypeError("Cannot find the coefficient of the provided element.")
            else:
                for trm in self.content:
                    if trm[1] == item.content[0][1]:
                        return trm[0]
        return 0.

    def __str__(self) -> str:
        self.content.sort(reverse=True, key=lambda x: x[1])
        return " + ".join("%.3f * %s" % _ for _ in self.content)


class SemigroupAlgebra(object):
    """A semigroup algebra.

    A semigroup algebra is a vector space over a field with a basis consisting of the elements of a semigroup.

    Attributes:
        gens (list): The generators of the semigroup.
        semigroup (CommutativeSemigroup): The semigroup of the algebra.
        derivatives (list): A list of dictionaries mapping the elements of the semigroup to their derivatives.
    """

    def __init__(self, semigroup: CommutativeSemigroup):
        """"Initializes a new instance of the SemigroupAlgebra class.

        Args:
            semigroup (CommutativeSemigroup): The semigroup of the algebra.
        """
        if not isinstance(semigroup, CommutativeSemigroup):
            raise TypeError("'semigroup' should be an instance of 'CommutativeSemigroup'.")
        self.gens = semigroup.symbols
        self.semigroup = semigroup
        self.derivatives = list()
        self.one = SemigroupAlgebraElement([(1., self.semigroup.G.identity)], self.semigroup)

    def __getitem__(self, idx: str) -> AtomicSGElement:
        """Gets the element of the algebra corresponding to the given generator of the semigroup.

        Args:
            idx (str): The generator of the semigroup.

        Returns:
            AtomicSGElement: The element of the algebra corresponding to the given generator of the semigroup.
        """
        if idx in self.gens:
            return AtomicSGElement(self.semigroup, idx)
        else:
            raise KeyError(f"'{idx}' is not a generator of the given semigroup")

    def __len__(self) -> int:
        """Returns the number of generators of the semigroup.

        Returns:
            int: The number of generators of the semigroup.
        """
        return len(self.gens)

    def add_derivative(self, base_map: dict) -> None:
        """Adds a derivative to the algebra.

        Args:
            base_map (dict): A dictionary mapping the elements of the semigroup to their derivatives.
        """
        if not isinstance(base_map, dict):
            raise TypeError("'base_map' must be a dictionary.")
        self.derivatives.append(base_map)

    def derivative(self, expr: Any, idx: int):
        """Computes the derivative of an expression in the algebra.

        Args:
            expr (SemigroupAlgebraElement, AtomicSGElement, int, float): The expression to differentiate.
            idx (int): The index of the derivative to use.

        Returns:
            SemigroupAlgebraElement: The derivative of the expression.
        """
        if idx >= len(self.derivatives):
            return 0
        else:
            return self.diff(expr, self.derivatives[idx])

    def diff(self, expr: Any, base_map: dict):
        """Computes the derivative of an expression in the algebra.

        Args:
            expr (SemigroupAlgebraElement, AtomicSGElement, int, float): The expression to differentiate.
            base_map (dict): A dictionary mapping the elements of the semigroup to their derivatives.

        Returns:
            SemigroupAlgebraElement: The derivative of the expression.
        """
        semigroup = self.semigroup
        res = 0.
        if not isinstance(expr, (SemigroupAlgebraElement, AtomicSGElement, int, float)):
            raise TypeError(f"Can find the derivative of '{type(expr)}'")
        if isinstance(expr, (int, float)):
            return SemigroupAlgebraElement([(0, semigroup.G.identity)], semigroup)
        elif isinstance(expr, AtomicSGElement):
            return base_map[expr.content[0][1]]
        elif isinstance(expr, SemigroupAlgebraElement):
            for trm in expr.content:
                cf = trm[0]
                sprt = trm[1].array_form
                if not sprt:
                    pass
                elif len(sprt) == 1:
                    sym_chr = sprt[0][0].name
                    symb = self[sym_chr]
                    exp = sprt[0][1]
                    res = res + cf * exp * base_map[sym_chr] * symb ** (exp - 1)
                else:
                    sym_chr = sprt[0][0].name
                    symb = self[sym_chr]
                    exp = sprt[0][1]
                    rest_comp = semigroup.G.identity
                    for _ in sprt[1:]:
                        rest_comp *= semigroup.__getattribute__(_[0].name) ** _[1]
                    rest = SemigroupAlgebraElement([(1, semigroup._reduce(rest_comp))], semigroup)
                    res = res + cf * (exp * base_map[sym_chr] * symb ** (exp - 1) * rest +
                                      symb ** exp * self.diff(rest, base_map))
        return res
