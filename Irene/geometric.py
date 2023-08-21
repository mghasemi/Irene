# from sage.matrix.constructor import Matrix
import numpy as np
from sympy import Symbol, Function, total_degree


class GPTools(object):
    """
    A class to find lower bounds of an even degree polynomial, using 
    Geometric Programs.
    """

    number_of_variables = 1
    total_degree = 2
    polynomial = 0
    constant_term = 0
    constant_terms = []
    fgp = 0
    Info = {'status': ''}
    MarginalCoeffs = []
    MaxIdx = 0
    Ord = 0
    Prog = []

    def __init__(self, Prg,  # Rng,
                 H=None, Settings={'Iterations': 150, 'detail': True, 'tryKKT': 0, 'AutoMatrix': True, 'precision': 7}):

        self.UDlt = None
        f = Prg[0].as_poly()
        self.Prog = []
        # self.PolyRng = Rng
        self.VARS = [_ for _ in f.atoms() if type(_) in [Symbol, Function]]
        self.polynomial = f
        # self.Field = self.polynomial.base_ring()
        self.number_of_variables = len(self.VARS)
        self.total_degree = total_degree(f)
        self.constant_term = self.const_trm(f)
        self.prg_size = 1
        if len(Prg) > 1:
            self.prg_size = len(Prg[1]) + 1
        if H is None:
            self.H = np.identity(self.prg_size)
        else:
            self.H = H
        self.Prog.append(f)
        self.constant_terms.append(self.constant_term)
        for i in range(self.prg_size - 1):
            self.Prog.append(Prg[1][i].as_poly(self.VARS))
            self.constant_terms.append(self.const_trm(Prg[1][i]))
            tmp_deg = total_degree(Prg[1][i])
            if self.total_degree < tmp_deg:
                self.total_degree = tmp_deg
        self.Settings = Settings
        if 'AutoMatrix' in Settings:
            self.AutoMatrix = Settings['AutoMatrix']
        else:
            self.AutoMatrix = False
        if 'Order' in Settings:
            self.Ord = max(Settings['Order'], self.total_degree)
        else:
            self.Ord = self.total_degree
        if (self.Ord % 2) == 1:
            self.Ord = self.Ord + 1
            self.MarginalCoeffs = []
        if 'precision' in Settings:
            self.error_bound = 10 ** (-Settings['precision'])
        else:
            self.error_bound = 10 ** (-7)
        self.Info = {'gp': 0, 'CPU': 0, 'Wall': 0, 'status': 'Unknown', 'Message': ''}

    def const_trm(self, ply):
        cnt_exp = tuple([0 for _ in self.VARS])
        dct = ply.as_dict()
        if cnt_exp in dct:
            return dct[cnt_exp]
        else:
            return 0.

    @staticmethod
    def is_square_mono(expn, coef):
        """
        This functions gets the coefficient and the exponent of a term and returns 
        True if it is a square term and False otherwise.
        """

        flag = True
        if coef < 0:
            return False
        for ex in expn:
            flag = flag & (ex % 2 == 0)
        return flag

    def UDelta(self):
        """
        This method returns a list consists of 0:coefficients, 1: monomials, 
        2: total degree of monomials, 3: number of non-zero exponents, 
        4: coefficients of diagonal monomials 
        for those terms that are not square in the program.
        """

        Dlt = [[], [], [], [], []]
        # Dlt[0]:Coefs     Dlt[1]:Monos     Dlt[2]:Degree     Delt[3]:n_alpha    Delt[4]:Diagonal monos coefs
        idx = 0
        # diagonal_monos = [p ** self.Ord for p in self.VARS]
        diagonal_monos = [tuple([0 for _ in range(self.number_of_variables)])]
        for i in range(self.number_of_variables):
            dflt = [0 for _ in range(self.number_of_variables)]
            dflt[i] = self.Ord
            diagonal_monos.append(tuple(dflt))
        # diagonal_monos.append(1)
        for tf in self.Prog:
            if idx == 0:
                f = tf
            else:
                f = -tf
            dct = f.as_dict()
            coefs = []
            expos = []
            for _ in dct:
                expos.append(_)
                coefs.append(dct[_])
            num_terms = len(coefs)
            for i in range(num_terms):
                if (expos[i] not in diagonal_monos) and (expos[i] not in Dlt[1]) and (
                        not self.is_square_mono(expos[i], coefs[i])):
                    Dlt[0].append(coefs[i])
                    Dlt[1].append(expos[i])
                    Dlt[2].append(sum(expos[i]))
                    Dlt[3].append(self.n_alpha(expos[i]))
                elif expos[i] in diagonal_monos:
                    Dlt[4].append(coefs[i])
            idx += 1
        self.UDlt = Dlt

    @staticmethod
    def tuple_to_exp(t1, t2):
        """
        This function takes two tuple of real numbers, raise each one in the fist one to the power of the corresponding
        entity in the second and then multiply them together.
        """

        mlt = 1
        n = len(t1)
        for i in range(0, n):
            if (t1[i] == 0) and (t2[i] == 0):
                continue
            mlt *= t1[i] ** t2[i]
        return mlt

    @staticmethod
    def n_alpha(alpha):
        """
        This function, counts the number of non-zero entities in the given exponent.
        """

        num = 0
        for i in alpha:
            if i != 0:
                num += 1
        return num

    @staticmethod
    def non_zero_before(mon, idx):
        """
        Counts the number of auxiliary lifting variables before a variable
        in a certain monomial.
        """

        cnt = 0
        for i in range(idx):
            if mon[i] != 0:
                cnt += 1
        return cnt

    @staticmethod
    def Matrix2CVXOPT(M):
        """
        Converts a Sage matrix into a matrix acceptable for the CvxOpt package.
        """

        from cvxopt.base import matrix as Mtx
        from array import array

        # n = M.ncols()
        # m = M.nrows()
        m, n = M.shape
        CM = []
        print(M)
        print(M.shape)
        for j in range(n):
            for i in range(m):
                CM.append(M[i, j])
        CC = Mtx(array('d', CM), (m, n))
        return CC

    def auto_matrix(self):
        """
        Returns a candidate for matrix H
        """

        self.H = np.identity(self.prg_size)
        diagonal_terms = [list((p ** self.Ord).as_poly(self.VARS).as_dict().keys())[0] for p in self.VARS]

        thm42 = True
        rmk43 = True

        for i in range(self.number_of_variables):
            nonzero = False
            k = self.prg_size
            gkdi = 0
            while not nonzero:
                k -= 1
                if k < 0:
                    break
                # if diagonal_terms[i] in self.Prog[k].monomials():
                prog_dict = self.Prog[k].as_dict()
                if diagonal_terms[i] in prog_dict:
                    # gkdi = self.Prog[k].coefficients()[self.Prog[k].monomials().index(diagonal_terms[i])]
                    gkdi = prog_dict[diagonal_terms[i]]
                if gkdi != 0:
                    nonzero = True
            if gkdi > 0:
                thm42 = False
                rmk43 = False
                break
            for j in range(k):
                gjdi = 0
                prog_dict = self.Prog[j].as_dict()
                # if diagonal_terms[i] in self.Prog[j].monomials():
                if diagonal_terms[i] in prog_dict:
                    # gjdi = self.Prog[j].coefficients()[self.Prog[j].monomials().index(diagonal_terms[i])]
                    gjdi = prog_dict[diagonal_terms[i]]
                if gjdi < 0:
                    rmk43 = False
                    break
        if rmk43:
            prog_dict0 = self.Prog[0].as_dict()
            for j in range(1, self.prg_size):
                sums = [0]
                for i in range(self.number_of_variables):
                    g0di = 0
                    # if diagonal_terms[i] in self.Prog[0].monomials():
                    if diagonal_terms[i] in prog_dict0:
                        # g0di = -self.Prog[0].coefficients()[self.Prog[0].monomials().index(diagonal_terms[i])]
                        g0di = -prog_dict0[diagonal_terms[i]]
                    tmp = g0di
                    gjdi = 0
                    prog_dict_j = self.Prog[j].as_dict()
                    # if diagonal_terms[i] in self.Prog[j].monomials():
                    if diagonal_terms[i] in prog_dict_j:
                        # gjdi = self.Prog[j].coefficients()[self.Prog[j].monomials().index(diagonal_terms[i])]
                        gjdi = prog_dict_j[diagonal_terms[i]]
                    if gjdi != 0:
                        for jp in range(1, j):
                            gjpdi = 0
                            prog_dict_jp = self.Prog[jp].as_dict()
                            if diagonal_terms[i] in prog_dict_jp:
                                # gjpdi = self.Prog[jp].coefficients()[self.Prog[jp].monomials().index(diagonal_terms[i])]
                                gjpdi = prog_dict_jp[diagonal_terms[i]]
                            tmp += self.H[jp, 0] * gjpdi
                        sums.append(tmp / gjdi)
                self.H[j, 0] = -max(sums)
            return
        if thm42:
            for j in range(self.prg_size):
                prog_dict_j = self.Prog[j].as_dict()
                for k in range(j):
                    sums = [0]
                    for i in range(self.number_of_variables):
                        gjdi = 0
                        # if diagonal_terms[i] in self.Prog[j].monomials():
                        if diagonal_terms[i] in prog_dict_j:
                            # gjdi = self.Prog[j].coefficients()[self.Prog[j].monomials().index(diagonal_terms[i])]
                            gjdi = prog_dict_j[diagonal_terms[i]]
                        rest_are_zero = True
                        for jp in range(j + 1, self.prg_size):
                            prog_dict_jp = self.Prog[jp].as_dict()
                            gjpdi = 0
                            # if diagonal_terms[i] in self.Prog[jp].monomials():
                            if diagonal_terms[i] in prog_dict_jp:
                                # gjpdi = self.Prog[jp].coefficients()[self.Prog[jp].monomials().index(diagonal_terms[i])]
                                gjpdi = prog_dict_jp[diagonal_terms[i]]
                            if gjpdi != 0:
                                rest_are_zero = False
                        if (gjdi < 0) and rest_are_zero:
                            tmp = gjdi
                            for jp in range(k + 1, j):
                                prog_dict_jp = self.Prog[jp].as_dict()
                                gjpdi = 0
                                # if diagonal_terms[i] in self.Prog[jp].monomials():
                                if diagonal_terms[i] in prog_dict_jp:
                                    # gjpdi = self.Prog[jp].coefficients()[
                                    #    self.Prog[jp].monomials().index(diagonal_terms[i])]
                                    gjpdi = prog_dict_jp[diagonal_terms[i]]
                                tmp += self.H[jp, k] * gjpdi
                            sums.append(-tmp / gjdi)
                        self.H[j, k] = min(sums)

    def init_geometric_program(self):
        """
        This function initializes the geometric program associated to
        the input a polynomial.
        """
        from math import log
        self.UDelta()
        num_z_alpha = sum(self.UDlt[3])
        num_w_alpha = len(self.UDlt[0])
        num_lambda = len(self.Prog) - 1
        big_dim = num_z_alpha + num_w_alpha + num_lambda
        zero_row = [0 for i in range(big_dim)]
        d = self.Ord
        diagonal_terms = [p ** d for p in self.VARS]
        Kt = []
        Gt = []
        Ft = []
        if self.AutoMatrix:
            self.auto_matrix()
        H2 = self.H
        H2[0, 0] = -self.H[0, 0]
        # POLYS = Matrix(self.Prog) * H2
        POLYS = np.matrix(self.Prog) * H2
        # self.constant_term = POLYS[0][0].constant_coefficient()
        self.constant_term = self.const_trm(POLYS[0, 0])

        ##########  Objective function: ##########
        ### sum over |alpha|<2d for objective: ###
        Ftr = [0 for l in range(big_dim)]
        var_before = 0
        term_cnt = 0
        for i in range(num_w_alpha):
            tmp_exp = self.UDlt[1][i]  # .exponents()[0]
            absalpha = self.UDlt[2][i]
            if absalpha < d:
                Gt.append(log((d - absalpha) * (self.tuple_to_exp(tmp_exp, tmp_exp) * (1.0 / d) ** d) ** (
                        1.0 / (d - absalpha))))
                for j in range(self.number_of_variables):
                    if tmp_exp[j] != 0:
                        Ftr[var_before] = -tmp_exp[j] * (1.0 / (d - absalpha))
                        var_before += 1
                Ftr[num_z_alpha + i] = d * (1.0 / (d - absalpha))
                Ft.append(Ftr)
                Ftr = [0 for l in range(big_dim)]
                term_cnt += 1
            else:
                var_before += self.UDlt[3][i]
        ### Linear part: ###
        for j in range(1, self.prg_size):
            # if POLYS[0, j].constant_coefficient() > 0:
            if self.const_trm(POLYS[0, j]) > 0:
                # Gt.append(log(POLYS[0, j].constant_coefficient()))
                Gt.append(log(self.const_trm(POLYS[0, j])))
                Ftr[num_z_alpha + num_w_alpha + j - 1] = 1
                Ft.append(Ftr)
                Ftr = [0 for l in range(big_dim)]
                term_cnt += 1
        if term_cnt > 0:
            Kt.append(term_cnt)
        ### End ###

        ### constraint set 1 for i=1,...,n ###
        self.geometric_program = True
        for j in range(self.number_of_variables):
            Ftr = [0 for l in range(big_dim)]
            var_before = 0
            term_cnt = 0
            lmbd = []
            pos_idx = -1
            positive_term_counter = 0
            for k in range(self.prg_size):
                sgn = -1
                tmp_poly = POLYS[0, k].as_dict()
                # if diagonal_terms[j] in POLYS[0][k].monomials():
                if diagonal_terms[j] in tmp_poly:
                    # Glambda = sgn * POLYS[0][k].coefficients()[POLYS[0][k].monomials().index(diagonal_terms[j])]
                    Glambda = sgn * tmp_poly[diagonal_terms[j]]
                    if Glambda > 0:
                        pos_idx = k
                        positive_term_counter += 1
                    lmbd.append(Glambda)
                else:
                    lmbd.append(0)
            ### Check for consistency: ###
            if positive_term_counter > 1:
                self.geometric_program = False
            ### ###
            if pos_idx >= 0:
                for i in range(num_w_alpha):
                    tmp_exp = self.UDlt[1][i]  # .exponents()[0]
                    if (tmp_exp[j] != 0):
                        ### ###
                        Gt.append(-log(lmbd[pos_idx]))
                        Ftr[var_before + self.non_zero_before(tmp_exp, j)] = 1
                        if pos_idx > 0:
                            Ftr[num_z_alpha + num_w_alpha + pos_idx - 1] = -1
                        Ft.append(Ftr)
                        Ftr = [0 for l in range(big_dim)]
                        term_cnt += 1
                    var_before += self.UDlt[3][i]
                for k in range(self.prg_size):
                    if (k != pos_idx) and (lmbd[k] != 0):
                        Gt.append(log(-lmbd[k]) - log(lmbd[pos_idx]))
                        if k > 0:
                            Ftr[num_z_alpha + num_w_alpha + k - 1] = 1
                        if pos_idx > 0:
                            Ftr[num_z_alpha + num_w_alpha + pos_idx - 1] = -1
                        Ft.append(Ftr)
                        Ftr = [0 for l in range(big_dim)]
                        term_cnt += 1
                if term_cnt > 0:
                    Kt.append(term_cnt)
        # End #

        # Constraints for |alpha|=2d: #
        var_before = 0
        for i in range(num_w_alpha):
            if self.UDlt[2][i] == d:
                tmp_exp = self.UDlt[1][i]
                Gt.append(log(self.tuple_to_exp(tmp_exp, tmp_exp) * (1.0 / d) ** d))
                Ftr[num_z_alpha + i] = d
                for j in range(self.number_of_variables):
                    if tmp_exp[j] != 0:
                        Ftr[var_before + self.non_zero_before(tmp_exp, j)] = -tmp_exp[j]
                Ft.append(Ftr)
                Ftr = [0 for _ in range(big_dim)]
                Kt.append(1)
            var_before += self.UDlt[3][i]
        ### End ###

        ### Constraints for H^+ & H^-: ###
        for i in range(num_w_alpha):
            Gp = []
            Gm = []
            sgn = 1
            for k in range(self.prg_size):
                if k > 0:
                    sgn = 1
                Glambda = 0
                p_k_dict = POLYS[0, k].as_dict()
                if self.UDlt[1][i] in p_k_dict:  # monomials():
                    # Glambda = sgn * POLYS[0][k].coefficients()[POLYS[0][k].monomials().index(self.UDlt[1][i])]
                    Glambda = sgn * p_k_dict[self.UDlt[1][i]]
                if Glambda >= 0:
                    Gm.append(Glambda)
                    Gp.append(0)
                else:
                    Gp.append(-Glambda)
                    Gm.append(0)
            term_cnt = 0
            for k in range(self.prg_size):
                if Gm[k] > 0:
                    Gt.append(log(Gm[k]))
                    if k > 0:
                        Ftr[num_z_alpha + num_w_alpha + k - 1] = 1
                    Ftr[num_z_alpha + i] = -1
                    term_cnt += 1
                    Ft.append(Ftr)
                    Ftr = [0 for l in range(big_dim)]
            if term_cnt > 0:
                Kt.append(term_cnt)
            term_cnt = 0
            for k in range(self.prg_size):
                if Gp[k] > 0:
                    Gt.append(log(Gp[k]))
                    if k > 0:
                        Ftr[num_z_alpha + num_w_alpha + k - 1] = 1
                    Ftr[num_z_alpha + i] = -1
                    term_cnt += 1
                    Ft.append(Ftr)
                    Ftr = [0 for l in range(big_dim)]
            if term_cnt > 0:
                Kt.append(term_cnt)
        ### Constraints for rows of H: ###
        for j in range(self.prg_size):
            num_nzr = 0
            num_pos = 0
            pos_idx = 0
            term_cnt = 0
            idx = 0
            for a in self.H[j]:
                if a != 0:
                    num_nzr += 1
                if a > 0:
                    num_pos += 1
                    pos_idx = idx
                idx += 1
            if (num_pos > 1) and (num_pos != num_nzr):
                self.geometric_program = False
            elif (num_pos != num_nzr) and (num_pos == 1):
                for k in range(self.prg_size):
                    if (k != pos_idx) and (self.H[j, k] != 0):
                        Gt.append(log(-self.H[j, k] / self.H[j, pos_idx]))
                        if k > 0:
                            Ftr[num_z_alpha + num_w_alpha + k - 1] = 1
                        if pos_idx > 0:
                            Ftr[num_z_alpha + num_w_alpha + pos_idx - 1] = -1
                        term_cnt += 1
                        Ft.append(Ftr)
                        Ftr = [0 for l in range(big_dim)]
                if term_cnt > 0:
                    Kt.append(term_cnt)
        # return [Matrix(Gt).transpose(), Matrix(Ft), Kt]
        return [np.matrix(Gt).transpose(), np.matrix(Ft), Kt]

    def minimize(self):
        """
        The main function to compute the lower bound for an
        even degree polynomial, using Geometric Program.
        """

        from cvxopt import solvers
        # from time import time, clock
        from math import exp
        RealNumber = float  # Required for CvxOpt
        Integer = int  # Required for CvxOpt

        if 'Iterations' in self.Settings:
            solvers.options['maxiters'] = self.Settings['Iterations']
        else:
            solvers.options['maxiters'] = 100
        if 'Details' in self.Settings:
            solvers.options['show_progress'] = self.Settings['Details']
        else:
            solvers.options['show_progress'] = False
        if 'tryKKT' in self.Settings:
            solvers.options['refinement'] = self.Settings['tryKKT']
        else:
            solvers.options['refinement'] = 1

        n = self.number_of_variables
        d = self.Ord
        f = self.polynomial
        GP = self.init_geometric_program()
        f0 = self.constant_term
        if not self.geometric_program:
            self.Info['status'] = 'Inapplicable'
            self.Info['Message'] = 'The input data does not result in a geometric program'
            self.Info['gp'] = None
            self.fgp = None
            return self.fgp
        F = self.Matrix2CVXOPT(GP[1])
        g = self.Matrix2CVXOPT(GP[0])
        K = GP[2]
        # start = time()
        # start2 = clock()
        # if True:
        try:
            sol = solvers.gp(K=K, F=F, g=g)
            # elapsed = (time() - start)
            # elapsed2 = (clock() - start2)
            self.fgp = min(-f0 - exp(sol['primal objective']), -f0 - exp(sol['dual objective']))
            self.Info = {"gp": self.fgp, }  # "Wall": elapsed, "CPU": elapsed2}
            if (sol['status'] == 'unknown'):
                if (abs(sol['relative gap']) < self.error_bound) or (abs(sol['gap']) < self.error_bound):
                    self.Info['status'] = 'The solver did not converge to a solution'
                    self.Info['Message'] = 'A possible optimum value were found.'
                else:
                    self.Info['status'] = 'Singular KKT or non-convergent IP may occurd'
                    self.Info['Message'] = 'Maximum iteration has reached by solver or a singular KKT matrix occurred.'
            else:
                self.Info['status'] = 'Optimal'
                self.Info['Message'] = 'Optimal solution found by solver.'
        # else:
        except ValueError as msg:
            self.Info['Message'] = "An error has occurred on CvxOpt.GP solver: `%s`" % msg
            self.Info['status'] = 'Infeasible'
            self.Info['gp'] = None
            self.fgp = None
        return self.fgp

########################################################################
