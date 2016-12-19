from base import base
from sdp import sdp


def Calpha_(expn, Mmnt):
    r"""
    Given an exponent `expn`, this function finds the corresponding
    :math:`C_{expn}` matrix which can be used for parallel processing.
    """
    from numpy import array, float64
    from sympy import zeros, Poly
    r = Mmnt.shape[0]
    C = zeros(r, r)
    for i in range(r):
        for j in range(i, r):
            entity = Mmnt[i, j]
            if expn in entity:
                C[i, j] = entity[expn]
                C[j, i] = C[i, j]
    return array(C.tolist()).astype(float64)


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
    ErrorTolerance = 10**-6
    AvailableSolvers = []
    PSDMoment = True
    Probability = True

    def __init__(self, gens, relations=[]):
        assert type(gens) is list, self.GensError
        assert type(gens) is list, self.RelsError
        from sympy import Function, Symbol, QQ, RR, groebner
        from sympy.core.relational import Equality, GreaterThan, LessThan, StrictGreaterThan, StrictLessThan
        import multiprocessing
        try:
            from joblib import Parallel
            self.Parallel = True
        except Exception as e:
            self.Parallel = False
        self.NumCores = multiprocessing.cpu_count()
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

    def SetNumCores(self, num):
        r"""
        Sets the maximum number of workers which cannot be bigger than 
        number of available cores.
        """
        assert (num > 0) and type(
            num) is int, "Number of cores must be a positive integer."
        self.NumCores = min(self.NumCores, num)

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
        except Exception as e:
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

    def LocalizedMoment_(self, p):
        r"""
        Computes the reduced symbolic moment generating matrix localized
        at `p`.
        """
        from sympy import Poly, Matrix, expand, zeros
        from math import ceil
        try:
            tot_deg = Poly(p).total_degree()
        except Exception as e:
            tot_deg = 0
        half_deg = int(ceil(tot_deg / 2.))
        mmntord = self.MmntOrd - half_deg
        m = Matrix(self.ReducedMonomialBase(mmntord))
        LMmnt = expand(p * m * m.T)
        LrMmnt = zeros(*LMmnt.shape)
        for i in range(LMmnt.shape[0]):
            for j in range(i, LMmnt.shape[1]):
                LrMmnt[i, j] = Poly(self.ReduceExp(
                    LMmnt[i, j]), *self.AuxSyms).as_dict()
                LrMmnt[j, i] = LrMmnt[i, j]
        return LrMmnt

    def MomentMat(self):
        r"""
        Returns the numerical moment matrix resulted from solving the SDP.
        """
        assert 'moments' in self.Info, "The sdp has not been (successfully) solved (yet)."
        from numpy import array, float64, matrix
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

    def sInitSDP(self):
        r"""
        Initializes the semidefinite program (SDP), in serial mode, whose 
        solution is a lower bound for the minimum of the program.
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
        if self.PSDMoment:
            d = len(self.ReducedMonomialBase(self.MmntOrd))
            # Corresponding C block is 0
            h = zeros(d, d)
            C.append(array(h.tolist()).astype(float64))
            Mmnt = self.LocalizedMoment(1.)
            for i in range(N):
                Blck[i].append(self.Calpha(ExpVec[i], Mmnt))
        ## L(1) = 1 ##
        if self.Probability:
            for i in range(N):
                Blck[i].append(array(
                    zeros(1, 1).tolist()).astype(float64))
                Blck[i].append(array(
                    zeros(1, 1).tolist()).astype(float64))
            #Blck[0][NumCns + 1][0] = 1
            #Blck[0][NumCns + 2][0] = -1
            Blck[0][-2][0] = 1
            Blck[0][-1][0] = -1
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

    def pInitSDP(self):
        r"""
        Initializes the semidefinite program (SDP), in parallel, whose 
        solution is a lower bound for the minimum of the program.
        """
        from numpy import array, float64
        from sympy import zeros, Matrix
        from time import time
        from joblib import Parallel, delayed
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
            Mmnt = self.LocalizedMoment_(self.Constraints[idx])
            results = Parallel(n_jobs=self.NumCores)(
                delayed(Calpha_)(ExpVec[i], Mmnt) for i in range(N))
            for i in range(N):
                Blck[i].append(results[i])
        ## Moment matrix should be psd ##
        if self.PSDMoment:
            d = len(self.ReducedMonomialBase(self.MmntOrd))
            # Corresponding C block is 0
            h = zeros(d, d)
            C.append(array(h.tolist()).astype(float64))
            Mmnt = self.LocalizedMoment_(1.)
            results = Parallel(n_jobs=self.NumCores)(
                delayed(Calpha_)(ExpVec[i], Mmnt) for i in range(N))
            for i in range(N):
                Blck[i].append(results[i])
        ## L(1) = 1 ##
        if self.Probability:
            for i in range(N):
                Blck[i].append(array(
                    zeros(1, 1).tolist()).astype(float64))
                Blck[i].append(array(
                    zeros(1, 1).tolist()).astype(float64))
            #Blck[0][NumCns + 1][0] = 1
            #Blck[0][NumCns + 2][0] = -1
            Blck[0][-2][0] = 1
            Blck[0][-1][0] = -1
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

    def InitSDP(self):
        r"""
        Initializes the SDP based on availability of ``joblib``.
        If it is available, it runs in parallel mode, otherwise
        in serial.
        """
        if self.Parallel:
            self.pInitSDP()
        else:
            self.sInitSDP()

    def Minimize(self):
        r"""
        Finds the minimum of the truncated moment problem which provides
        a lower bound for the actual minimum.
        """
        self.SDP.solve()
        self.Solution = SDRelaxSol(
            self.AuxSyms, symdict=self.SymDict, err_tol=self.ErrorTolerance)
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
            self.Solution.NumGenerators = self.NumGenerators
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

    def Decompose(self):
        r"""
        Returns a dictionary that associates a list to every constraint,
        :math:`g_i\ge0` for :math:`i=0,\dots,m`, where :math:`g_0=1`.
        Each list consists of elements of algebra whose sums of squares
        is equal to :math:`\sigma_i` and :math:`f-f_*=\sum_{i=0}^m\sigma_ig_i`.
        Here, :math:`f_*` is the output of the ``SDPRelaxation.Minimize()``.
        """
        from numpy.linalg import cholesky
        from sympy import Matrix
        SOSCoefs = {}
        blks = []
        NumCns = len(self.CnsDegs)
        for M in self.SDP.Info['X']:
            blks.append(Matrix(cholesky(M)))
        for idx in range(NumCns):
            SOSCoefs[idx + 1] = []
            v = Matrix(self.ReducedMonomialBase(
                self.MmntOrd - self.CnsHalfDegs[idx])).T
            decomp = v * blks[idx]
            for p in decomp:
                SOSCoefs[idx + 1].append(p.subs(self.RevSymDict))
        v = Matrix(self.ReducedMonomialBase(self.MmntOrd)).T
        SOSCoefs[0] = []
        decomp = v * blks[NumCns]
        for p in decomp:
            SOSCoefs[0].append(p.subs(self.RevSymDict))
        return SOSCoefs

    def getObjective(self):
        r"""
        Returns the objective function of the problem after reduction modulo the relations, if given.
        """
        return self.RedObjective.subs(self.RevSymDict)

    def getConstraint(self, idx):
        r"""
        Returns the constraint number `idx` of the problem after reduction modulo the relations, if given.
        """
        assert idx < len(self.Constraints), "Index out of range."
        return self.Constraints[idx].subs(self.RevSymDict) >= 0

    def getMomentConstraint(self, idx):
        r"""
        Returns the moment constraint number `idx` of the problem after reduction modulo the relations, if given.
        """
        assert idx < len(self.MomConst), "Index out of range."
        from sympy import sympify
        return self.MomConst[idx][0].subs(self.RevSymDict) >= sympify(self.MomConst[idx][1]).subs(self.RevSymDict)

    def __str__(self):
        from sympy import sympify
        out_txt = "="*70+"\n"
        out_txt += "Minimize\t"
        out_txt += str(self.RedObjective.subs(self.RevSymDict)) + "\n"
        out_txt += "Subject to\n"
        for cns in self.Constraints:
            out_txt += "\t\t" + str(cns.subs(self.RevSymDict) >= 0)+"\n"
        out_txt += "And\n"
        for cns in self.MomConst:
            out_txt += "\t\tMoment " + str(cns[0].subs(self.RevSymDict) >= sympify(cns[1]).subs(self.RevSymDict))+"\n"
        out_txt += "="*70+"\n"
        return out_txt

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
        - ``SDRelaxSol.ScipySolver`` the scipy solver to extract solutions
        - ``SDRelaxSol.err_tol`` the minimum value which is considered to be nonzero
        - ``SDRelaxSol.Support`` the support of discrete measure resulted from ``SDPRelaxation.Minimize()``
        - ``SDRelaxSol.Weights`` corresponding weights for the Dirac measures
    """

    def __init__(self, X, symdict={}, err_tol=10e-6):
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
        self.Xij = None
        self.NumGenerators = None
        # SDPRelaxations auxiliary symbols
        self.X = X
        self.SymDict = symdict
        self.err_tol = err_tol
        self.ScipySolver = 'lm'
        self.Support = None
        self.Weights = None

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
        if self.Support is not None:
            out_str += "               Support:\n"
            for p in self.Support:
                out_str += "\t\t" + str(p) + "\n"
            out_str += "          Scipy solver: " + self.ScipySolver + "\n"
        out_str += self.Message + "\n"
        return out_str

    def SetScipySolver(self, solver):
        r"""
        Sets the ``scipy.optimize.root`` solver to `solver`.
        """
        assert solver.lower() in ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov',
                                  'df-sane'], "Unrecognized solver. The solver must be among 'hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane'"
        self.ScipySolver = solver.lower()

    def NumericalRank(self):
        r"""
        Finds the rank of the moment matrix based on the size of its
        eigenvalues. It considers those with absolute value less than 
        ``self.err_tol`` to be zero.
        """
        from scipy.linalg import eigvals
        from numpy import isreal, real, abs
        num_rnk = 0
        eignvls = eigvals(self.MomentMatrix)
        for ev in eignvls:
            if abs(ev) >= self.err_tol:
                num_rnk += 1
        return num_rnk

    def Term2Mmnt(self, trm, rnk, X):
        r"""
        Converts a moment object into an algebraic equation.
        """
        num_vars = len(X)
        expr = 0
        for i in range(rnk):
            expr += self.weight[i] * \
                trm.subs({X[j]: self.Xij[i][j] for j in range(num_vars)})
        return expr

    def ExtractSolution(self):
        r"""
        This method tries to extract the corresponding values for 
        generators of the ``SDPRelaxation`` class.
        Number of points is the rank of the moment matrix which is 
        computed numerically according to the size of its eigenvalues.
        Then the points are extracted as solutions of a system of 
        polynomial equations using a `scipy` solver.
        The followin solvers are currently acceptable by ``scipy``:
            - ``hybr``, 
            - ``lm`` (default),
            - ``broyden1``, 
            - ``broyden2``,
            - ``anderson``,
            - ``linearmixing``,
            - ``diagbroyden``, 
            - ``excitingmixing``,
            - ``krylov``,
            - ``df-sane``.
        """
        from scipy import optimize as opt
        from sympy import Symbol, lambdify, Abs
        rnk = self.NumericalRank()
        self.weight = [Symbol('w%d' % i, real=True) for i in range(1, rnk + 1)]
        self.Xij = [[Symbol('X%d%d' % (i, j), real=True) for i in range(1, self.NumGenerators + 1)]
                    for j in range(1, rnk + 1)]
        syms = [s for row in self.Xij for s in row]
        for ri in self.weight:
            syms.append(ri)
        req = sum(self.weight) - 1
        algeqs = {idx.subs(self.SymDict): self.TruncatedMmntSeq[
            idx] for idx in self.TruncatedMmntSeq}
        indices = []
        included_sysms = set(self.weight)
        EQS = [req]
        hold = []
        for i in range(len(algeqs)):
            trm = algeqs.keys()[i]
            if trm != 1:
                strm = self.Term2Mmnt(trm, rnk, self.X) - algeqs[trm]
                strm_syms = strm.free_symbols
                if not strm_syms.issubset(included_sysms):
                    # EQS.append(strm)
                    EQS.append(strm.subs({ri: Abs(ri) for ri in self.weight}))
                    included_sysms = included_sysms.union(strm_syms)
                else:
                    # hold.append(strm)
                    hold.append(strm.subs({ri: Abs(ri) for ri in self.weight}))
        idx = 0
        while (len(EQS) < len(syms)):
            if len(hold) > idx:
                EQS.append(hold[idx])
                idx += 1
            else:
                break
        if (included_sysms != set(syms)) or (len(EQS) != len(syms)):
            raise Exception("Unable to find the support.")
        f_ = [lambdify(syms, eq, 'numpy') for eq in EQS]

        def f(x):
            z = tuple(float(x.item(i)) for i in range(len(syms)))
            return [fn(*z) for fn in f_]
        init_point = tuple(0.  # uniform(-self.err_tol, self.err_tol)
                           for _ in range(len(syms)))
        sol = opt.root(f, init_point, method=self.ScipySolver)
        if sol['success']:
            self.Support = []
            self.Weights = []
            idx = 0
            while idx < len(syms) - rnk:
                minimizer = []
                for i in range(self.NumGenerators):
                    minimizer.append(sol['x'][idx])
                    idx += 1
                self.Support.append(tuple(minimizer))
            while idx < len(syms):
                self.Weights.append(sol['x'][idx])
                idx += 1

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
        from sympy import sympify
        self.NumericTypes = [IntType, LongType, FloatType]
        self.Content = sympify(expr)
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
