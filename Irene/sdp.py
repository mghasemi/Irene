from base import base


class sdp(base):
    r"""
    This is the class which intends to solve semidefinite programs in 
    primal format: 

    .. math::
        \left\lbrace
        \begin{array}{lll}
            \min & \sum_{i=1}^m b_i x_i & \\
            \textrm{subject to} & & \\
                & \sum_{i=1}^m A_{ij}x_i - C_j \succeq 0 & j=1,\dots,k.
        \end{array}\right.

    For the argument `solver` following sdp solvers are supported (if they are installed):
        + `CVXOPT`,
        + `CSDP`,
        + `SDPA`,
        + `DSDP`.
    """
    Solvers = ['CVXOPT', 'SDPA', 'CSDP', 'DSDP']
    SolverOptions = {}
    Info = {}

    def __init__(self, solver='cvxopt'):
        assert solver.upper() in self.Solvers, "Currently the\
        following solvers are supported: 'CVXOPT', 'SDPA', 'CSDP', 'DSDP'"
        self.solver = solver.upper()
        self.BlockStruct = []
        self.b = None
        self.A = []
        self.C = []
        # checks the availability of solver
        if solver.upper() not in self.AvailableSDPSolvers():
            raise ImportError("The solver '%s' is not available" % solver)

    def SetObjective(self, b):
        r"""
        Takes the coefficients of the objective function.
        """
        self.b = b

    def AddConstraintBlock(self, A):
        r"""
        This takes a list of square matrices which corresponds to coefficient
        of :math:`x_i`. Simply, :math:`A_i=[A_{i1},\dots,A_{ik}]`.
        Note that the :math:`i^{th}` call of ``AddConstraintBlock`` fills the 
        blocks associated with :math:`i^{th}` variable :math:`x_i`.
        """
        BlkStc = []
        for blk in A:
            BlkStc.append(blk.shape[0])
        if (self.BlockStruct != []) and (self.BlockStruct == BlkStc):
            self.A.append(A)
        elif (self.BlockStruct == []):
            self.BlockStruct = BlkStc
            self.A.append(A)
        else:
            raise TypeError("The block structure is inconsistent.")

    def AddConstantBlock(self, C):
        r"""
        `C` must be a list of ``numpy`` matrices that represent :math:`C_j`
        for each `j`.
        This method sets the value for :math:`C=[C_1,\dots,C_k]`.
        """
        BlkStc = []
        for blk in C:
            BlkStc.append(blk.shape[0])
        if (self.BlockStruct != []) and (self.BlockStruct == BlkStc):
            self.C = C
        elif (self.BlockStruct == []):
            self.BlockStruct = BlkStc
            self.C = C
        else:
            raise TypeError("The block structure is inconsistent.")

    def Option(self, param, val):
        r"""
        Sets the `param` option of the solver to `val` if the solver accepts
        such an option. The following options are supported by solvers:

            + ``CVXOPT``:

                + ``show_progress``: ``True`` or ``False``, turns the output to the screen on or off (default: ``True``);

                + ``maxiters``: maximum number of iterations (default: 100);

                + ``abstol``: absolute accuracy (default: 1e-7);

                + ``reltol``: relative accuracy (default: 1e-6);

                + ``feastol``: tolerance for feasibility conditions (default: 1e-7);

                + ``refinement``: number of iterative refinement steps when solving KKT equations (default: 0 if the problem has no second-order cone or matrix inequality constraints; 1 otherwise).

            + ``SDPA``:

                + ``maxIteration``: Maximum number of iterations. The SDPA stops when the iteration exceeds ``maxIteration``;

                + ``epsilonStar``, ``epsilonDash``: The accuracy of an approximate optimal solution of the SDP;

                + ``lambdaStar``: This parameter determines an initial point;

                + ``omegaStar``: This parameter determines the region in which the SDPA searches an optimal solution;

                + ``lowerBound``: Lower bound of the minimum objective value of the primal problem;

                + ``upperBound``: Upper bound of the maximum objective value of the dual problem;

                + ``betaStar``: Parameter controlling the search direction when current state is feasible;

                + ``betaBar``: Parameter controlling the search direction when current state is infeasible;

                + ``gammaStar``: Reduction factor for the primal and dual step lengths; 0.0 < ``gammaStar`` < 1.0.
        """
        self.SolverOptions[param] = val

    def write_sdpa_dat(self, filename):
        r"""
        Writes the semidefinite program in the file `filename` with dense SDPA format.
        """
        f = open(filename, 'w')
        f.write("%d=mDIM\n" % len(self.b))
        f.write("%d=nBLOCK\n" % len(self.C))
        f.write(str(self.BlockStruct).replace(
            '[', '{').replace(']', '}') + "=bLOCKsTRUCT\n")
        f.write(str(self.b).replace('[', '{').replace(']', '}') + "\n")
        f.write('{\n')
        for B in self.C:
            f.write(str(B).replace('[', '{').replace(']', '}') + '\n')
        f.write('}\n')
        for B in self.A:
            f.write('{\n')
            for Bl in B:
                f.write(str(Bl).replace('[', '{').replace(']', '}') + '\n')
            f.write('}\n')
        f.close()

    def write_sdpa_dat_sparse(self, filename):
        r"""
        Writes the semidefinite program in the file `filename` with sparse SDPA format.
        """
        f = open(filename, 'w')
        f.write("%d = mDIM\n" % len(self.b))
        f.write("%d = nBLOCK\n" % len(self.C))
        f.write(str(self.BlockStruct).replace('[', '').replace(
            ']', '').replace(',', ' ') + " = bLOCKsTRUCT\n")
        f.write(str(self.b).replace(
            '[', '').replace(']', '').replace(',', ' ') + "\n")
        mat_no = 0
        blk_no = 1
        for B in self.C:
            for i in range(1, B.shape[0] + 1):
                for j in range(i, B.shape[1] + 1):
                    if B[i - 1][j - 1] != 0.:
                        f.write("%d %d %d %d %f\n" %
                                (mat_no, blk_no, i, j, B[i - 1][j - 1]))
            blk_no += 1
        for B in self.A:
            mat_no += 1
            blk_no = 1
            for Bl in B:
                for i in range(1, Bl.shape[0] + 1):
                    for j in range(i, Bl.shape[1] + 1):
                        if Bl[i - 1][j - 1] != 0.:
                            f.write("%d %d %d %d %f\n" %
                                    (mat_no, blk_no, i, j, Bl[i - 1][j - 1]))
                blk_no += 1
        f.close()

    def parse_solution_matrix(self, iterator):
        r"""
        Parses and returns the matrices and vectors found by `SDPA` solver.
        This was taken from `ncpol2sdpa` and customized for `Irene`.
        """
        import numpy as np
        solution_matrix = []
        while True:
            sol_mat = None
            in_matrix = False
            i = 0
            for row in iterator:
                if row.find('}') < 0:
                    continue
                if row.startswith('}'):
                    break
                if row.find('{') != row.rfind('{'):
                    in_matrix = True
                numbers = row[
                    row.rfind('{') + 1:row.find('}')].strip().split(',')
                if sol_mat is None:
                    sol_mat = np.empty((len(numbers), len(numbers)))
                for j, number in enumerate(numbers):
                    sol_mat[i, j] = float(number)
                if row.find('}') != row.rfind('}') or not in_matrix:
                    break
                i += 1
            solution_matrix.append(sol_mat)
            if row.startswith('}'):
                break
        if len(solution_matrix) > 0 and solution_matrix[-1] is None:
            solution_matrix = solution_matrix[:-1]
        return solution_matrix

    def read_sdpa_out(self, filename):
        r"""
        Extracts information from `SDPA`'s output file `filename`.
        This was taken from `ncpol2sdpa` and customized for `Irene`.
        """
        from numpy import array
        primal = None
        dual = None
        x_mat = None
        y_mat = None
        xVec = None
        status_string = None

        with open(filename, 'r') as file_:
            for line in file_:
                if line.find("objValPrimal") > -1:
                    primal = float((line.split())[2])
                if line.find("objValDual") > -1:
                    dual = float((line.split())[2])
                if line.find("total time") > -1:
                    total_time = float((line.split('='))[1])
                if line.find("xMat =") > -1:
                    x_mat = self.parse_solution_matrix(file_)
                if line.find("yMat =") > -1:
                    y_mat = self.parse_solution_matrix(file_)
                if line.find("xVec =") > -1:
                    line = next(file_)
                    xVec = array([float(m) for m in line.replace(
                        '{', '').replace('}', '').split(',')])
                if line.find("phase.value") > -1:
                    if (line.find("pdOPT") > -1) or line.find("pdFEAS") > -1:
                        status_string = 'Optimal'
                    elif line.find("INF") > -1:
                        status_string = 'Infeasible'
                    elif line.find("UNBD") > -1:
                        status_string = 'Unbounded'
                    else:
                        status_string = 'Unknown'

        for var in [primal, dual, status_string]:
            if var is None:
                status_string = 'invalid'
                break
        for var in [x_mat, y_mat]:
            if var is None:
                status_string = 'invalid'
                break
        self.Info['PObj'] = primal
        self.Info['DObj'] = dual
        self.Info['X'] = y_mat
        self.Info['Z'] = x_mat
        self.Info['y'] = xVec
        self.Info['Status'] = status_string
        self.Info['CPU'] = total_time

    def sdpa_param(self):
        r"""
        Produces sdpa.param file from ``SolverOptions``.
        """
        f = open("param.sdpa", 'w')
        if 'maxIteration' in self.SolverOptions:
            f.write("%d unsigned int maxIteration;\n" %
                    self.SolverOptions['maxIteration'])
        else:
            f.write("40  unsigned int maxIteration;\n")
        if 'epsilonStar' in self.SolverOptions:
            f.write("%f  double 0.0 < epsilonStar;\n" %
                    self.SolverOptions['epsilonStar'])
        else:
            f.write("1.0E-7  double 0.0 < epsilonStar;\n")
        if 'lambdaStar' in self.SolverOptions:
            f.write("%f  double 0.0 < lambdaStar;\n" %
                    self.SolverOptions['lambdaStar'])
        else:
            f.write("1.0E2  double 0.0 < lambdaStar;\n")
        if 'omegaStar' in self.SolverOptions:
            f.write("%f  double 1.0 < omegaStar;\n" %
                    self.SolverOptions['omegaStar'])
        else:
            f.write("2.0  double 1.0 < omegaStar;\n")
        if 'lowerBound' in self.SolverOptions:
            f.write("%f double lowerBound;\n" %
                    self.SolverOptions['lowerBound'])
        else:
            f.write("-1.0E5  double lowerBound;\n")
        if 'upperBound' in self.SolverOptions:
            f.write("%f  double upperBound;\n" %
                    self.SolverOptions['upperBound'])
        else:
            f.write("1.0E5  double upperBound;\n")
        if 'betaStar' in self.SolverOptions:
            f.write("%f  double 0.0 <= betaStar < 1.0;\n" %
                    self.SolverOptions['betaStar'])
        else:
            f.write("0.1  double 0.0 <= betaStar < 1.0;\n")
        if 'betaBar' in self.SolverOptions:
            f.write("%f  double 0.0 <= betaBar < 1.0, betaStar <= betaBar;\n" %
                    self.SolverOptions['betaBar'])
        else:
            f.write("0.2  double 0.0 <= betaBar < 1.0, betaStar <= betaBar;\n")
        if 'gammaStar' in self.SolverOptions:
            f.write("%f  double 0.0 < gammaStar < 1.0;\n" %
                    self.SolverOptions['gammaStar'])
        else:
            f.write("0.9  double 0.0 < gammaStar < 1.0;\n")
        if 'epsilonDash' in self.SolverOptions:
            f.write("%f  double 0.0 < epsilonDash;\n" %
                    self.SolverOptions['epsilonDash'])
        else:
            f.write("1.0E-7  double 0.0 < epsilonDash;\n")
        f.close()

    def read_csdp_out(self, filename, txt):
        r"""
        Takes a file name and a string that are the outputs of `CSDP` as
        a file and command line outputs of the solver and extracts the
        required information.
        """
        from numpy import array, zeros
        Status = 'Unknown'
        progress = txt.split('\n')
        for line in progress:
            if line.find("Success") > -1:
                Status = 'Optimal'
            elif line.find("Primal objective value") > -1:
                primal = float(line.split(':')[1])
            elif line.find("Dual objective value") > -1:
                dual = float(line.split(':')[1])
            elif line.find("Total time") > -1:
                total_time = float(line.split(':')[1])
        file_ = open(filename, 'r')
        line = file_.readline()
        xVec = array([float(m) for m in line.split(' ')[:-1]])
        X = [zeros((d, d)) for d in self.BlockStruct]
        Z = [zeros((d, d)) for d in self.BlockStruct]
        for line in file_:
            entity = line.split(' ')
            if int(entity[0]) == 1:
                Z[int(entity[1]) - 1][int(entity[2]) -
                                      1][int(entity[3]) - 1] = float(entity[4])
            elif int(entity[0]) == 2:
                X[int(entity[1]) - 1][int(entity[2]) -
                                      1][int(entity[3]) - 1] = float(entity[4])
        # self.BlockStruct
        self.Info['PObj'] = primal
        self.Info['DObj'] = dual
        self.Info['X'] = X
        self.Info['Z'] = Z
        self.Info['y'] = xVec
        self.Info['Status'] = Status
        self.Info['CPU'] = total_time

    def VEC(self, M):
        """
        Converts the matrix M into a column vector acceptable by `CVXOPT`.
        """

        V = []
        n, m = M.shape
        for j in range(m):
            for i in range(n):
                V.append(M[i, j])
        return V

    def CvxOpt(self):
        r"""
        This calls `CVXOPT` and `DSDP` to solve the initiated semidefinite program.
        """
        from time import time, clock
        from numpy import matrix, array, float64
        try:
            from cvxopt import solvers
            from cvxopt.base import matrix as Mtx
            RealNumber = float  # Required for CvxOpt
            Integer = int       # Required for CvxOpt
            self.CvxOpt_Available = True
        except Exception as e:
            self.CvxOpt_Available = False
            self.ErrorString = "CVXOPT is not available."
            raise Exception(self.ErrorString)
        self.solver_options = {}
        self.Info = {}

        self.num_constraints = len(self.A)
        self.num_blocks = len(self.C)

        Cns = []
        for idx in range(self.num_constraints):
            Cns.append([])
        Acvxopt = []
        Ccvxopt = []
        acvxopt = []
        for M in self.C:
            Ccvxopt.append(-Mtx(M, tc='d'))
        for blk_no in range(self.num_blocks):
            Ablock = []
            for Cns in self.A:
                Ablock.append(self.VEC(Cns[blk_no]))
            Acvxopt.append(-Mtx(matrix(Ablock).transpose(), tc='d'))
        aTranspose = []
        for elmnt in self.b:
            aTranspose.append([elmnt])
        n1 = len(aTranspose[0])
        m1 = len(aTranspose)
        acvxopt = Mtx(array(aTranspose).reshape(
            m1 * n1, order='F').astype(float64), size=(m1, n1), tc='d')
        # CvxOpt options
        for param in self.SolverOptions:
            solvers.options[param] = self.SolverOptions[param]
        start1 = time()
        start2 = clock()

        try:
            # if True:
            sol = solvers.sdp(acvxopt, Gs=Acvxopt, hs=Ccvxopt,
                              solver=self.solver.lower())
            elapsed1 = (time() - start1)
            elapsed2 = (clock() - start2)
            if sol['status'] != 'optimal':
                self.Info = {'Status': 'Infeasible'}
            else:
                self.Info = {'Status': 'Optimal', 'DObj': sol['dual objective'],
                             'PObj': sol['primal objective'], 'Wall': elapsed1, 'CPU': elapsed2}
                self.Info['y'] = array(
                    list(sol['x']))  # .reshape(*sol['x'].size)
                self.Info['Z'] = []
                for ds in sol['ss']:
                    self.Info['Z'].append(
                        array(list(ds)).reshape(*ds.size))
                self.Info['X'] = []
                for ds in sol['zs']:
                    self.Info['X'].append(
                        array(list(ds)).reshape(*ds.size))
        except Exception as e:
            self.Info = {'Status': 'Infeasible'}

        self.Info['solver'] = self.solver

    def sdpa(self):
        r"""
        Calls `SDPA` to solve the initiated semidefinite program.
        """
        from subprocess import call
        prg_file = "prg.dat"
        out_file = "out.res"
        self.sdpa_param()
        par_file = "param.sdpa"
        if self.BlockStruct == []:
            self.BlockStruct = [len(B) for B in self.C]
        self.write_sdpa_dat(prg_file)
        call(["sdpa", "-dd", prg_file, "-o", out_file, "-p", par_file])
        self.read_sdpa_out(out_file)

    def csdp(self):
        r"""
        Calls `SDPA` to solve the initiated semidefinite program.
        """
        from subprocess import check_output
        prg_file = "prg.dat-s"
        out_file = "out.res"
        if self.BlockStruct == []:
            self.BlockStruct = [len(B) for B in self.C]
        self.write_sdpa_dat_sparse(prg_file)
        try:
            out = check_output(["csdp", prg_file, out_file])
        except Exception as e:
            pass
        self.read_csdp_out(out_file, out)

    def solve(self):
        r"""
        Solves the initiated semidefinite program according to the requested solver.
        """
        if self.solver in ['CVXOPT', 'DSDP']:
            self.CvxOpt()
        elif self.solver == 'SDPA':
            self.sdpa()
        elif self.solver == 'CSDP':
            self.csdp()

    def __str__(self):
        out_text = "Semidefinite program with\n"
        out_text += "             # variables:" + str(len(self.C)) + "\n"
        out_text += "           # constraints:" + str(len(self.A)) + "\n"
        out_text += "             with solver:" + self.solver

    def __latex__(self):
        return "SDP(%d, %d, %s)"%(len(self.C), len(self.A), self.solver)