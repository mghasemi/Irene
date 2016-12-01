from base import base
from sdp import sdp


class SDPRelaxations(base):
    r"""
    This class defines a function space by taking a family of sympy
    symbolic functions and relations among them.
    Simply, it initiates a commutative free real algebra on the symbolic
    functions and defines the function space as the quotient of the free
    algebra by the ideal generated by the given relations.
    It takes two arguments:
        - `gens` which is a list of ``sympy`` symbols and function symbols,
        - `relations` which is a set of ``sympy`` expressions in terms of `gens` that defines an ideal.
    """
    GensError = r"""The `gens` must be a list of sympy functions or symbols"""
    RelsError = r"""The `relations` must be a list of relation among generators"""
    MonoOrdError = r"""`ord` must be one of 'lex', 'grlex', 'grevlex', 'ilex', 'igrlex', 'igrevlex'"""
    MmntOrdError = r"""The order of moments must be a positive integer"""
    SDPInpTypError = r"""The input of the SDP solver must be either a numpy matrix or ndarray"""
    # Monomial order: "lex", "grlex", "grevlex", "ilex", "igrlex", "igrevlex"
    MonomialOrder = 'lex'
    SDPSolver = 'cvxopt'
    Info = {}
    ErrorTolerance = 10**-8
    AvailableSolvers = []

    def __init__(self, gens, relations=[]):
        assert type(gens) is list, self.GensError
        assert type(gens) is list, self.RelsError
        from sympy import Function, Symbol, QQ, RR, groebner
        from sympy.core.relational import Equality, GreaterThan, LessThan, StrictGreaterThan, StrictLessThan
        self.EQ = Equality
        self.GEQ = GreaterThan
        self.LEQ = LessThan
        self.GT = StrictGreaterThan
        self.LT = StrictLessThan
        self.ExpTypes = [Equality, GreaterThan,
                         LessThan, StrictGreaterThan, StrictLessThan]
        self.Field = QQ
        self.Generators = []
        self.SymDict = {}
        self.RevSymDict = {}
        self.AuxSyms = []
        self.NumGenerators = 0
        self.FreeRelations = []
        self.Groebner = []
        self.MmntOrd = 0
        self.ReducedBases = {}
        #
        self.Constraints = []
        self.MomConst = []
        self.ObjDeg = 0
        self.ObjHalfDeg = 0
        self.CnsDegs = []
        self.CnsHalfDegs = []
        # check generators
        for f in gens:
            if isinstance(f, Function) or isinstance(f, Symbol):
                self.Generators.append(f)
                self.NumGenerators += 1
                t_sym = Symbol('X%d' % self.NumGenerators)
                self.SymDict[f] = t_sym
                self.RevSymDict[t_sym] = f
                self.AuxSyms.append(t_sym)
            else:
                raise TypeError(self.GensError)
        # check the relations
        # TBI
        for r in relations:
            t_rel = r.subs(self.SymDict)
            self.FreeRelations.append(t_rel)
        if self.FreeRelations != []:
            self.Groebner = groebner(
                self.FreeRelations, domain=self.Field, order=self.MonomialOrder)
        self.AvailableSolvers = self.AvailableSDPSolvers()

    def SetMonoOrd(self, ord):
        r"""
        Changes the default monomial order to `ord` which mustbe among
        `lex`, `grlex`, `grevlex`, `ilex`, `igrlex`, `igrevlex`.
        """
        assert ord in ['lex', 'grlex', 'grevlex', 'ilex',
                       'igrlex', 'igrevlex'], self.MonoOrdError
        from sympy import groebner
        self.MonomialOrder = ord
        if self.FreeRelations != []:
            self.Groebner = groebner(
                self.FreeRelations, domain=self.Field, order=self.MonomialOrder)

    def SetSDPSolver(self, solver):
        r"""
        Sets the default SDP solver. The followings are currently supported:
            - CVXOPT
            - DSDP
            - SDPA
            - CSDP

        The selected solver must be installed otherwise it cannot be called.
        The default solver is `CVXOPT` which has an interface for Python.
        `DSDP` is called through the CVXOPT's interface. `SDPA` and `CSDP`
        are called independently.
        """
        assert solver.upper() in ['CVXOPT', 'DSDP', 'SDPA',
                                  'CSDP'], "'%s' sdp solver is not supported" % solver
        self.SDPSolver = solver

    def ReduceExp(self, expr):
        r"""
        Takes an expression `expr`, either in terms of internal free symbolic
        variables or generating functions and returns the reduced expression
        in terms of internal symbolic variables, if a relation among generators
        is present, otherwise it just substitutes generating functions with
        their corresponding internal symbols.
        """
        from sympy import reduced
        T = expr.subs(self.SymDict)
        if self.Groebner != []:
            return reduced(T, self.Groebner)[1]
        else:
            return T

    def SetObjective(self, obj):
        r"""
        Takes the objective function `obj` as an algebraic combination 
        of the generating symbolic functions, replace the symbolic 
        functions with corresponding auxiliary symbols and reduce them 
        according to the given relations.
        """
        from math import ceil
        from sympy import Poly
        self.Objective = obj
        self.RedObjective = self.ReduceExp(obj)
        # self.CheckVars(obj)
        tot_deg = Poly(self.RedObjective).total_degree()
        self.ObjDeg = tot_deg
        self.ObjHalfDeg = int(ceil(tot_deg / 2.))

    def AddConstraint(self, cnst):
        r"""
        Takes an (in)equality as an algebraic combination of the 
        generating functions that defines the feasibility region.
        It reduces the defining (in)equalities according to the
        given relations.
        """
        from sympy import Poly
        from math import ceil
        CnsTyp = type(cnst)
        if CnsTyp in self.ExpTypes:
            if CnsTyp in [self.GEQ, self.GT]:
                non_red_exp = cnst.lhs - cnst.rhs
                expr = self.ReduceExp(non_red_exp)
                self.Constraints.append(expr)
                tot_deg = Poly(expr).total_degree()
                self.CnsDegs.append(tot_deg)
                self.CnsHalfDegs.append(int(ceil(tot_deg / 2.)))
            elif CnsTyp in [self.LEQ, self.LT]:
                non_red_exp = cnst.rhs - cnst.lhs
                expr = self.ReduceExp(non_red_exp)
                self.Constraints.append(expr)
                tot_deg = Poly(expr).total_degree()
                self.CnsDegs.append(tot_deg)
                self.CnsHalfDegs.append(int(ceil(tot_deg / 2.)))
            elif CnsTyp is self.EQ:
                non_red_exp = cnst.lhs - cnst.rhs
                expr = self.ReduceExp(non_red_exp)
                self.Constraints.append(self.ErrorTolerance + expr)
                self.Constraints.append(self.ErrorTolerance - expr)
                tot_deg = Poly(expr).total_degree()
                # add twice
                self.CnsDegs.append(tot_deg)
                self.CnsDegs.append(tot_deg)
                self.CnsHalfDegs.append(int(ceil(tot_deg / 2.)))
                self.CnsHalfDegs.append(int(ceil(tot_deg / 2.)))

    def MomentConstraint(self, cnst):
        r"""
        Takes constraints on the moments. The input must be an instance of
        `Mom` class.
        """
        from sympy import Poly
        from math import ceil
        assert isinstance(
            cnst, Mom), "The argument must be of moment type 'Mom'"
        CnsTyp = cnst.TYPE
        if CnsTyp in ['ge', 'gt']:
            expr = self.ReduceExp(cnst.Content)
            self.MomConst.append([expr, cnst.rhs])
        elif CnsTyp in ['le', 'lt']:
            expr = self.ReduceExp(-cnst.Content)
            self.MomConst.append([expr, -cnst.rhs])
        elif CnsTyp == 'eq':
            non_red_exp = cnst.Content - cnst.rhs
            expr = self.ReduceExp(cnst.Content)
            self.MomConst.append([expr, cnst.rhs - self.ErrorTolerance])
            self.MomConst.append([-expr, -cnst.rhs - self.ErrorTolerance])

    def ReducedMonomialBase(self, deg):
        r"""
        Returns a reduce monomial basis up to degree `d`.
        """
        if deg in self.ReducedBases:
            return self.ReducedBases[deg]
        from itertools import product
        from operator import mul
        from sympy import Poly
        all_monos = product(range(deg + 1), repeat=self.NumGenerators)
        req_monos = filter(lambda x: sum(x) <= deg, all_monos)
        monos = [reduce(mul, [self.AuxSyms[i]**expn[i]
                              for i in range(self.NumGenerators)], 1) for expn in req_monos]
        RBase = []
        for expr in monos:
            rexpr = self.ReduceExp(expr)
            expr_monos = Poly(rexpr, *self.AuxSyms).as_dict()
            for mono_exp in expr_monos:
                t_mono = reduce(mul, [self.AuxSyms[i]**mono_exp[i]
                                      for i in range(self.NumGenerators)], 1)
                if t_mono not in RBase:
                    RBase.append(t_mono)
        self.ReducedBases[deg] = RBase
        return RBase

    def ExponentsVec(self, deg):
        r"""
        Returns all the exponents that appear in the reduced basis of all
        monomials of the auxiliary symbols of degree at most `deg`.
        """
        from sympy import Poly
        basis = self.ReducedMonomialBase(deg)
        exponents = []
        for elmnt in basis:
            rbp = Poly(elmnt, *self.AuxSyms).as_dict()
            for expnt in rbp:
                if expnt not in exponents:
                    exponents.append(expnt)
        return exponents

    def MomentsOrd(self, ord):
        r"""
        Sets the order of moments to be considered.
        """
        from types import IntType
        assert (type(ord) is IntType) and (ord > 0), self.MmntOrdError
        self.MmntOrd = ord

    def RelaxationDeg(self):
        r"""
        Finds the minimum required order of moments according to user's
        request, objective function and constraints.
        """
        if self.CnsHalfDegs == []:
            CHD = 0
        else:
            CHD = max(self.CnsHalfDegs)
        RlxDeg = max([CHD, self.ObjHalfDeg, self.MmntOrd])
        self.MmntOrd = RlxDeg
        return RlxDeg

    def PolyCoefFullVec(self):
        r"""
        return the vector of coefficient of the reduced objective function
        as an element of the vector space of elements of degree up to the
        order of moments.
        """
        from sympy import Poly
        c = []
        fmono = Poly(self.RedObjective, *self.AuxSyms).as_dict()
        exponents = self.ExponentsVec(2 * self.MmntOrd)
        for expn in exponents:
            if expn in fmono:
                c.append(fmono[expn])
            else:
                c.append(0)
        return c

    def LocalizedMoment(self, p):
        r"""
        Computes the reduced symbolic moment generating matrix localized
        at `p`.
        """
        from sympy import Poly, Matrix, expand, zeros
        from math import ceil
        try:
            tot_deg = Poly(p).total_degree()
        except:
            tot_deg = 0
        half_deg = int(ceil(tot_deg / 2.))
        mmntord = self.MmntOrd - half_deg
        m = Matrix(self.ReducedMonomialBase(mmntord))
        LMmnt = expand(p * m * m.T)
        LrMmnt = zeros(*LMmnt.shape)
        for i in range(LMmnt.shape[0]):
            for j in range(i, LMmnt.shape[1]):
                LrMmnt[i, j] = self.ReduceExp(LMmnt[i, j])
                LrMmnt[j, i] = LrMmnt[i, j]
        return LrMmnt

    def MomentMat(self):
        r"""
        Returns the numerical moment matrix resulted from solving the SDP.
        """
        assert 'moments' in self.Info, "The sdp has not been (successfully) solved (yet)."
        from numpy import array, float64
        from sympy import Poly
        from operator import mul
        Mmnt = self.LocalizedMoment(1.)
        for i in range(Mmnt.shape[0]):
            for j in range(Mmnt.shape[1]):
                t_monos = Poly(Mmnt[i, j], *self.AuxSyms).as_dict()
                t_mmnt = 0
                for expn in t_monos:
                    mono = reduce(mul, [self.AuxSyms[k]**expn[k]
                                        for k in range(self.NumGenerators)], 1)
                    t_mmnt += t_monos[expn] * self.Info['moments'][mono]
                Mmnt[i, j] = t_mmnt
                Mmnt[j, i] = Mmnt[i, j]
        return array(Mmnt.tolist()).astype(float64)

    def Calpha(self, expn, Mmnt):
        r"""
        Given an exponent `expn`, this method finds the corresponding
        :math:`C_{expn}` matrix.
        """
        from numpy import array, float64
        from sympy import zeros, Poly
        r = Mmnt.shape[0]
        C = zeros(r, r)
        for i in range(r):
            for j in range(i, r):
                entity = Mmnt[i, j]
                entity_monos = Poly(entity, *self.AuxSyms).as_dict()
                if expn in entity_monos:
                    C[i, j] = entity_monos[expn]
                    C[j, i] = C[i, j]
        return array(C.tolist()).astype(float64)

    def InitSDP(self):
        r"""
        Initializes the semidefinite program (SDP) whose solution is a lower 
        bound for the minimum of the program.
        """
        from numpy import array, float64
        from sympy import zeros, Matrix
        from time import time
        start = time()
        self.SDP = sdp(self.SDPSolver)
        self.RelaxationDeg()
        N = len(self.ReducedMonomialBase(2 * self.MmntOrd))
        self.MatSize = [len(self.ReducedMonomialBase(self.MmntOrd)), N]
        Blck = [[] for _ in range(N)]
        C = []
        # Number of constraints
        NumCns = len(self.CnsDegs)
        # Number of moment constraints
        NumMomCns = len(self.MomConst)
        # Reduced vector of monomials of the given order
        ExpVec = self.ExponentsVec(2 * self.MmntOrd)
        ## The localized moment matrices should be psd ##
        for idx in range(NumCns):
            d = len(self.ReducedMonomialBase(
                self.MmntOrd - self.CnsHalfDegs[idx]))
            # Corresponding C block is 0
            h = zeros(d, d)
            C.append(array(h.tolist()).astype(float64))
            Mmnt = self.LocalizedMoment(self.Constraints[idx])
            for i in range(N):
                Blck[i].append(self.Calpha(ExpVec[i], Mmnt))
        ## Moment matrix should be psd ##
        d = len(self.ReducedMonomialBase(self.MmntOrd))
        # Corresponding C block is 0
        h = zeros(d, d)
        C.append(array(h.tolist()).astype(float64))
        Mmnt = self.LocalizedMoment(1.)
        for i in range(N):
            Blck[i].append(self.Calpha(ExpVec[i], Mmnt))
        ## L(1) = 1 ##
        for i in range(N):
            Blck[i].append(array(
                zeros(1, 1).tolist()).astype(float64))
            Blck[i].append(array(
                zeros(1, 1).tolist()).astype(float64))
        Blck[0][NumCns + 1][0] = 1
        Blck[0][NumCns + 2][0] = -1
        C.append(array(Matrix([1]).tolist()).astype(float64))
        C.append(array(Matrix([-1]).tolist()).astype(float64))
        # Moment constraints
        for idx in range(NumMomCns):
            MomCns = Matrix([self.MomConst[idx][0]])
            for i in range(N):
                Blck[i].append(self.Calpha(ExpVec[i], MomCns))
            C.append(array(
                Matrix([self.MomConst[idx][1]]).tolist()).astype(float64))
        self.SDP.C = C
        self.SDP.b = self.PolyCoefFullVec()
        self.SDP.A = Blck
        elapsed = (time() - start)
        self.InitTime = elapsed

    def Minimize(self):
        r"""
        Finds the minimum of the truncated moment problem which provides
        a lower bound for the actual minimum.
        """
        self.SDP.solve()
        self.Solution = SDRelaxSol()
        self.Info = {}
        self.Solution.Status = self.SDP.Info['Status']
        if self.SDP.Info['Status'] == 'Optimal':
            self.f_min = min(self.SDP.Info['PObj'], self.SDP.Info['DObj'])
            self.Solution.Primal = self.SDP.Info['PObj']
            self.Solution.Dual = self.SDP.Info['DObj']
            self.Info = {"min": self.f_min, "CPU": self.SDP.Info[
                'CPU'], 'InitTime': self.InitTime}
            self.Solution.RunTime = self.SDP.Info['CPU']
            self.Solution.InitTime = self.InitTime
            self.Info['status'] = 'Optimal'
            self.Info[
                'Message'] = 'Feasible solution for moments of order ' + str(self.MmntOrd)
            self.Solution.Message = self.Info['Message']
            self.Info['tms'] = self.SDP.Info['y']
            FullMonVec = self.ReducedMonomialBase(2 * self.MmntOrd)
            self.Info['moments'] = {FullMonVec[i]: self.Info[
                'tms'][i] for i in range(len(FullMonVec))}
            self.Info['solver'] = self.SDP.solver
            for idx in self.Info['moments']:
                self.Solution.TruncatedMmntSeq[idx.subs(self.RevSymDict)] = self.Info[
                    'moments'][idx]
            self.Solution.MomentMatrix = self.MomentMat()
            self.Solution.Solver = self.SDP.solver
        else:
            self.f_min = None
            self.Info['min'] = self.f_min
            self.Info['status'] = 'Infeasible'
            self.Info['Message'] = 'No feasible solution for moments of order ' + \
                str(self.MmntOrd) + ' were found'
            self.Solution.Status = 'Infeasible'
            self.Solution.Message = self.Info['Message']
            self.Solution.Solver = self.SDP.solver
        self.Info["Size"] = self.MatSize
        return self.f_min

#######################################################################
# Solution of the Semidefinite Relaxation


class SDRelaxSol(object):
    r"""
    Instances of this class carry information on the solution of the
    semidefinite relaxation associated to a optimization problem.
    It include various pieces of information:
        - ``SDRelaxSol.TruncatedMmntSeq`` a dictionary of resulted moments
        - ``SDRelaxSol.MomentMatrix`` the resulted moment matrix
        - ``SDRelaxSol.Primal`` the value of the SDP in primal form
        - ``SDRelaxSol.Dual`` the value of the SDP in dual form
        - ``SDRelaxSol.RunTime`` the run time of the sdp solver
        - ``SDRelaxSol.InitTime`` the total time consumed for initialization of the sdp
        - ``SDRelaxSol.Solver`` the name of sdp solver
        - ``SDRelaxSol.Status`` final status of the sdp solver
        - ``SDRelaxSol.RelaxationOrd`` order of relaxation
        - ``SDRelaxSol.Message`` the message that maybe returned by the sdp solver
    """

    def __init__(self):
        self.TruncatedMmntSeq = {}
        self.MomentMatrix = None
        self.Primal = None
        self.Dual = None
        self.RunTime = None
        self.InitTime = None
        self.Solver = None
        self.Status = None
        self.RelaxationOrd = None
        self.Message = None

    def __str__(self):
        out_str = "Solution of a Semidefinite Program:\n"
        out_str += "                Solver: " + self.Solver + "\n"
        out_str += "                Status: " + self.Status + "\n"
        out_str += "   Initialization Time: " + \
            str(self.InitTime) + " seconds\n"
        out_str += "              Run Time: " + \
            str(self.RunTime) + " seconds\n"
        out_str += "Primal Objective Value: " + str(self.Primal) + "\n"
        out_str += "  Dual Objective Value: " + str(self.Dual) + "\n"
        out_str += self.Message + "\n"
        return out_str

#######################################################################
# A Symbolic object to handle moment constraints


class Mom(object):
    r"""
    This is a simple interface to define moment constraints to be 
    used via `SDPRelaxations.MomentConstraint`.
    It takes a sympy expression as input and initiates an object 
    which can be used to force particular constraints on the moment
    sequence.

    **Example:** Force the moment of :math:`x^2f(x) + f(x)^2` to be at least `.5`::

        Mom(x**2 * f + f**2) >= .5
        # OR
        Mom(x**2 * f) + Mom(f**2) >= .5
    """

    def __init__(self, expr):
        from types import IntType, LongType, FloatType
        self.NumericTypes = [IntType, LongType, FloatType]
        self.Content = expr
        self.rhs = 0
        self.TYPE = None

    def __add__(self, x):
        if isinstance(x, Mom):
            self.Content += x.Content
        else:
            self.Content += x
        return self

    def __sub__(self, x):
        if isinstance(x, Mom):
            self.Content -= x.Content
        else:
            self.Content -= x
        return self

    def __neg__(self):
        self.Content = -self.Content
        return self

    def __mul__(self, x):
        if type(x) in self.NumericTypes:
            self.Content = x * self.Content
        else:
            raise Exception("Operation not supported")
        return self

    def __rmul__(self, x):

        if type(x) in self.NumericTypes:
            self.Content = x * self.Content
        else:
            raise Exception("Operation not supported")
        return self

    def __ge__(self, x):
        if isinstance(x, Mom):
            self.rhs = 0
            self.Content -= x.Content
        elif type(x) in self.NumericTypes:
            self.rhs = x
        self.TYPE = 'ge'
        return self

    def __gt__(self, x):
        if isinstance(x, Mom):
            self.rhs = 0
            self.Content -= x.Content
        elif type(x) in self.NumericTypes:
            self.rhs = x
        self.TYPE = 'gt'
        return self

    def __le__(self, x):
        if isinstance(x, Mom):
            self.rhs = 0
            self.Content -= x.Content
        elif type(x) in self.NumericTypes:
            self.rhs = x
        self.TYPE = 'le'
        return self

    def __lt__(self, x):
        if isinstance(x, Mom):
            self.rhs = 0
            self.Content -= x.Content
        elif type(x) in self.NumericTypes:
            self.rhs = x
        self.TYPE = 'lt'
        return self

    def __eq__(self, x):
        if isinstance(x, Mom):
            self.rhs = 0
            self.Content -= x.Content
        elif type(x) in self.NumericTypes:
            self.rhs = x
        self.TYPE = 'eq'
        return self

    def __str__(self):
        symbs = {'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'eq': '=='}
        strng = str(self.Content)
        if self.TYPE is not None:
            strng += " " + symbs[self.TYPE]
            strng += " " + str(self.rhs)
        return strng