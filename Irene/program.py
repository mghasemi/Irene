from math import ceil

from .grouprings import _degree, SemigroupAlgebraElement, SemigroupAlgebra, CommutativeSemigroup


class OptimizationProblem(object):
    """
    This class represents an optimization problem.

    Attributes:
        sga (SemigroupAlgebra): The semigroup algebra of the optimization problem.
        relations (list[SemigroupAlgebraElement]): The relations of the optimization problem.
        semigroup (CommutativeSemigroup): The semigroup of the optimization problem.
        objective (SemigroupAlgebraElement): The objective function of the optimization problem.
        constraints (list[SemigroupAlgebraElement]): The constraints of the optimization problem.
        objective_degree (int): The degree of the objective function.
        objective_half_degree (int): The half degree of the objective function.
        constraints_degree (list[int]): The degrees of the constraints.
        constraints_half_degree (list[int]): The half degrees of the constraints.
        objective_trms_with_positive_coefficient (list[SemigroupAlgebraElement]): The terms of the objective function with positive coefficients.
        objective_trms_with_negative_coefficient (list[SemigroupAlgebraElement]): The terms of the objective function with negative coefficients.
        objective_terms_with_even_exponent (list[SemigroupAlgebraElement]): The terms of the objective function with even exponents.
        objective_terms_with_odd_exponent (list[SemigroupAlgebraElement]): The terms of the objective function with odd exponents.
        constraint_trms_with_positive_coefficient (list[SemigroupAlgebraElement]): The terms of the constraints with positive coefficients.
        constraint_trms_with_negative_coefficient (list[SemigroupAlgebraElement]): The terms of the constraints with negative coefficients.
        constraint_terms_with_even_exponent (list[SemigroupAlgebraElement]): The terms of the constraints with even exponents.
        constraint_terms_with_odd_exponent (list[SemigroupAlgebraElement]): The terms of the constraints with odd exponents.
        total_degree (int): The total degree of the optimization problem.
    """

    def __init__(self, sga: SemigroupAlgebra = None,
                 relations: list[SemigroupAlgebraElement] = None):
        self.sga: SemigroupAlgebra = sga
        self.relations = relations
        self.semigroup: CommutativeSemigroup = CommutativeSemigroup([])
        if self.sga is not None:
            self.semigroup = self.sga.semigroup
        self.objective = None
        self.constraints = list()
        self.objective_degree = 0
        self.objective_half_degree = 0
        self.constraints_degree = list()
        self.constraints_half_degree = list()
        self.objective_trms_with_positive_coefficient = list()
        self.objective_trms_with_negative_coefficient = list()
        self.objective_terms_with_even_exponent = list()
        self.objective_terms_with_odd_exponent = list()
        self.constraint_trms_with_positive_coefficient = list()
        self.constraint_trms_with_negative_coefficient = list()
        self.constraint_terms_with_even_exponent = list()
        self.constraint_terms_with_odd_exponent = list()
        self.total_degree = 2

    def set_objective(self, obj: SemigroupAlgebraElement):
        """
        Sets the objective function for the optimization problem.
        :param obj: an `SemigroupAlgebraElement` expression to be optimized.
        :return: `None`
        """
        self.objective = obj
        if self.semigroup != obj.semigroup:
            self.semigroup = obj.semigroup
        self.objective_degree = _degree(obj.LM())
        self.objective_half_degree = int(ceil(self.objective_degree / 2.))

    def add_constraints(self, const: list[SemigroupAlgebraElement]):
        """
        Adds constraints to the optimization problem.

        Args:
            const (list[SemigroupAlgebraElement]): The constraints to add.
        """
        for exp in const:
            if self.semigroup != exp.semigroup:
                self.semigroup = exp.semigroup
            self.constraints.append(exp)
            exp_deg = _degree(exp.LM())
            exp_half_deg = int(ceil(exp_deg / 2.))
            self.constraints_degree.append(exp_deg)
            self.constraints_half_degree.append(exp_half_deg)

    def program_degree(self):
        """
        Computes the degree of the optimization problem.

        Returns:
            int: The degree of the optimization problem.
        """
        degs = self.constraints_degree + [self.objective_degree]
        dg = max(degs)
        if dg % 2 == 0:
            return dg
        return dg + 1

    def analyse_program(self):
        """
        Analyses the optimization problem.

        This method separates the terms of the objective function and the constraints based on the sign of their coefficients and the exponents of the semigroup content.
        """
        # Separate terms of objective and constraints based on the sign of their coefficients
        for trm in self.objective:
            if trm[0] > 0:
                self.objective_trms_with_positive_coefficient.append(trm)
            else:
                self.objective_trms_with_negative_coefficient.append(trm)
            if self.square_exponent(trm[1]):
                self.objective_terms_with_even_exponent.append(trm)
            else:
                self.objective_terms_with_odd_exponent.append(trm)
        # Separate terms of objective and constraints based on the exponents of the semigroup content
        for xprsn in self.constraints:
            for trm in xprsn.content:
                if trm[0] > 0:
                    self.constraint_trms_with_positive_coefficient.append(trm)
                else:
                    self.constraint_trms_with_negative_coefficient.append(trm)
                if self.square_exponent(trm[1]):
                    self.constraint_terms_with_even_exponent.append(trm)
                else:
                    self.constraint_terms_with_odd_exponent.append(trm)

    @staticmethod
    def square_exponent(xpnt):
        """
        Checks if the exponent is a square.

        Args:
            xpnt (SemigroupAlgebraElement): The exponent to check.

        Returns:
            bool: True if the exponent is a square, False otherwise.
        """
        for _ in xpnt.array_form:
            if _[1] % 2 != 0:
                return False
        return True

    @staticmethod
    def has_symbol(symb, mono):
        """
        Checks if the monomial contains the given symbol.

        Args:
            symb (str): The symbol to check for.
            mono (SemigroupAlgebraElement): The monomial to check.

        Returns:
            tuple[bool, int]: True if the monomial contains the symbol, False otherwise. If True, the index of the symbol in the monomial's array form is also returned.
        """
        idx = 0
        for _ in mono.array_form:
            if symb == _[0].name:
                return True, idx
            idx += 1
        return False, -1

    @staticmethod
    def omega(xprsn, deg):  # Add the term for 0
        """
        Computes the omega of the expression.

        Args:
            xprsn (SemigroupAlgebraElement): The expression to compute the omega of.
            deg (int): The degree of the omega.

        Returns:
            list[tuple[int, SemigroupAlgebraElement]]: The terms of the omega.
        """
        terms = list()
        for trm in xprsn.content:
            if len(trm[1].array_form) > 1:
                terms.append(trm)
                continue
            if not trm[1].array_form:
                continue
            if trm[1].array_form[0][1] == deg:
                continue
            else:
                terms.append(trm)
        return terms

    def delta(self, xprsn, deg):
        """
        Computes the delta of the expression.

        Args:
            xprsn (SemigroupAlgebraElement): The expression to compute the delta of.
            deg (int): The degree of the delta.

        Returns:
            dict[str, set[SemigroupAlgebraElement]]: The terms of the delta, divided into two sets: '=d' and '<d'.
        """
        terms = {'=d': set([]), '<d': set([])}
        omega = self.omega(xprsn, deg)
        for trm in omega:
            if not self.square_exponent(trm[1]) or trm[0] < 0:
                if self.semigroup.degree(trm[1]) == deg:
                    terms['=d'].add(trm[1])
                else:
                    terms['<d'].add(trm[1])
        return terms
