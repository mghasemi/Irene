from .base import base

from numpy import array, zeros, matrix, float64
from time import time


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

    def __init__(self, solver='cvxopt', solver_path=None):
        solver_upper = solver.upper() if isinstance(solver, str) else None
        if solver_upper not in self.Solvers:
            raise ValueError("Currently the following solvers are supported: 'CVXOPT', 'SDPA', 'CSDP', 'DSDP'")
        super(sdp, self).__init__()
        if solver_path:
            self.Path = dict(solver_path)
        self.solver = solver_upper
        self.BlockStruct = []
        self.b = None
        self.A = []
        self.C = []
        self.CvxOpt_Available = False
        self.ErrorString = ""
        self.solver_options = {}
        self.Info = {}
        self.num_constraints = 0
        self.num_blocks = 0

        # checks the availability of solver
        if self.solver not in self.AvailableSDPSolvers():
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
        elif not self.BlockStruct:
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
        elif not self.BlockStruct:
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

    @staticmethod
    def _coerce_float(value):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("SDP data must be numeric") from exc

    def _objective_values_as_floats(self):
        return [self._coerce_float(value) for value in self.b]

    def write_sdpa_dat(self, filename):
        r"""
        Writes the semidefinite program in the file `filename` with dense SDPA format.
        """
        f = open(filename, 'w')
        f.write("%d=mDIM\n" % len(self.b))
        f.write("%d=nBLOCK\n" % len(self.C))
        f.write(str(self.BlockStruct).replace(
            '[', '{').replace(']', '}') + "=bLOCKsTRUCT\n")
        objective = ', '.join('{:.16g}'.format(value)
                              for value in self._objective_values_as_floats())
        f.write('{' + objective + "}\n")
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
        Uses sparse iteration over non-zero entries to avoid dense nested loops.
        """
        import numpy as np
        sparse_zero_tol = 1e-12
        with open(filename, 'w') as f:
            f.write("%d = mDIM\n" % len(self.b))
            f.write("%d = nBLOCK\n" % len(self.C))
            f.write(str(self.BlockStruct).replace('[', '').replace(
                ']', '').replace(',', ' ') + " = bLOCKsTRUCT\n")
            objective = ' '.join('{:.16g}'.format(value)
                                 for value in self._objective_values_as_floats())
            f.write(objective + "\n")
            mat_no = 0
            blk_no = 1
            for B in self.C:
                # Use argwhere to iterate only over non-zero entries
                nonzero_idx = np.argwhere(np.abs(B) > sparse_zero_tol)
                for idx in nonzero_idx:
                    i, j = idx[0], idx[1]
                    # Only output upper triangular part (i <= j)
                    if i <= j:
                        val = B[i, j]
                        f.write("%d %d %d %d %f\n" %
                                (mat_no, blk_no, i + 1, j + 1, val))
                blk_no += 1
            for B in self.A:
                mat_no += 1
                blk_no = 1
                for Bl in B:
                    # Use argwhere to iterate only over non-zero entries
                    nonzero_idx = np.argwhere(np.abs(Bl) > sparse_zero_tol)
                    for idx in nonzero_idx:
                        i, j = idx[0], idx[1]
                        # Only output upper triangular part (i <= j)
                        if i <= j:
                            val = Bl[i, j]
                            f.write("%d %d %d %d %f\n" %
                                    (mat_no, blk_no, i + 1, j + 1, val))
                    blk_no += 1

    @staticmethod
    def parse_solution_matrix(iterator):
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
            row = None
            for row in iterator:
                stripped = row.strip()
                if stripped.find('}') < 0:
                    continue
                if stripped.startswith('}'):
                    if sol_mat is None:
                        return solution_matrix
                    break
                if stripped.find('{') < 0:
                    raise ValueError("Malformed solution matrix row")
                if stripped.find('{') != stripped.rfind('{'):
                    in_matrix = True
                numbers = stripped[
                          stripped.rfind('{') + 1:stripped.find('}')].strip().split(',')
                if len(numbers) == 1 and numbers[0] == '':
                    raise ValueError("Empty solution matrix row")
                if sol_mat is None:
                    sol_mat = np.empty((len(numbers), len(numbers)))
                elif len(numbers) != sol_mat.shape[1]:
                    raise ValueError("Inconsistent solution matrix row width")
                if i >= sol_mat.shape[0]:
                    raise ValueError("Too many rows in solution matrix")
                for j, number in enumerate(numbers):
                    sol_mat[i, j] = float(number)
                i += 1
                if stripped.find('}') != stripped.rfind('}') or not in_matrix:
                    break
            if sol_mat is None:
                return solution_matrix
            if i != sol_mat.shape[0]:
                raise ValueError("Incomplete solution matrix")
            solution_matrix.append(sol_mat)
            if row is not None and row.strip().startswith('}'):
                break
        return solution_matrix

    def read_sdpa_out(self, filename):
        r"""
        Extracts information from `SDPA`'s output file `filename`.
        This was taken from `ncpol2sdpa` and customized for `Irene`.
        """
        primal = None
        dual = None
        x_mat = None
        y_mat = None
        xVec = None
        status_string = None
        total_time = None

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
                    elif line.find("noINFO") > -1:
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
        primal = None
        dual = None
        total_time = None
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
        with open(filename, 'r') as file_:
            line = file_.readline()
            x_tokens = line.split()
            if not x_tokens:
                raise ValueError("Malformed CSDP solution vector")
            xVec = array([float(token) for token in x_tokens])
            X = [zeros((d, d)) for d in self.BlockStruct]
            Z = [zeros((d, d)) for d in self.BlockStruct]
            for line in file_:
                entity = line.split()
                if not entity:
                    continue
                if len(entity) != 5:
                    raise ValueError("Malformed CSDP solution row")
                mat_type, block_idx, row_idx, col_idx, value = entity
                block_no = int(block_idx) - 1
                row_no = int(row_idx) - 1
                col_no = int(col_idx) - 1
                if int(mat_type) == 1:
                    Z[block_no][row_no][col_no] = float(value)
                elif int(mat_type) == 2:
                    X[block_no][row_no][col_no] = float(value)
        # self.BlockStruct
        self.Info['PObj'] = primal
        self.Info['DObj'] = dual
        self.Info['X'] = X
        self.Info['Z'] = Z
        self.Info['y'] = xVec
        self.Info['Status'] = Status
        self.Info['CPU'] = total_time

    @staticmethod
    def VEC(M):
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
        try:
            from cvxopt import solvers
            from cvxopt.base import matrix as Mtx
            RealNumber = float  # Required for CvxOpt
            Integer = int  # Required for CvxOpt
            self.CvxOpt_Available = True
        except Exception as e:
            self.CvxOpt_Available = False
            self.ErrorString = "CVXOPT is not available."
            raise Exception(self.ErrorString)
        self.solver_options = {}
        self.Info = {}

        self.num_constraints = len(self.A)
        self.num_blocks = len(self.C)

        # Build Ccvxopt: objective matrix blocks
        Ccvxopt = [-Mtx(M, tc='d') for M in self.C]
        
        # Build Acvxopt: constraint matrix blocks
        Acvxopt = []
        for blk_no in range(self.num_blocks):
            Ablock = [self.VEC(constraint[blk_no]) for constraint in self.A]
            Acvxopt.append(-Mtx(matrix(Ablock).transpose(), tc='d'))
        
        # Build acvxopt: objective vector using direct numpy approach
        b_coerced = [self._coerce_float(elmnt) for elmnt in self.b]
        acvxopt = Mtx(array(b_coerced).reshape(-1, 1), tc='d')
        
        # CvxOpt options
        for param in self.SolverOptions:
            solvers.options[param] = self.SolverOptions[param]
        start1 = time()

        try:
            # if True:
            sol = solvers.sdp(acvxopt, Gs=Acvxopt, hs=Ccvxopt,
                              solver=self.solver.lower())
            elapsed1 = (time() - start1)
            if sol['status'] != 'optimal':
                self.Info = {'Status': 'Infeasible'}
            else:
                self.Info = {'Status': 'Optimal', 'DObj': sol['dual objective'], 'PObj': sol['primal objective'],
                             'Wall': elapsed1, 'CPU': None, 'y': array(
                        list(sol['x'])), 'Z': []}
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
        import subprocess
        prg_file = "prg.dat"
        out_file = "out.res"
        self.sdpa_param()
        par_file = "param.sdpa"
        if not self.BlockStruct:
            self.BlockStruct = [len(B) for B in self.C]
        self.write_sdpa_dat(prg_file)
        try:
            subprocess.run(
                [self.Path['sdpa'], "-dd", prg_file, "-o", out_file, "-p", par_file],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("SDPA execution failed") from exc
        self.read_sdpa_out(out_file)

    def csdp(self):
        r"""
        Calls `SDPA` to solve the initiated semidefinite program.
        """
        import subprocess
        prg_file = "prg.dat-s"
        out_file = "out.res"
        if not self.BlockStruct:
            self.BlockStruct = [len(B) for B in self.C]
        self.write_sdpa_dat_sparse(prg_file)
        try:
            completed = subprocess.run(
                [self.Path['csdp'], prg_file, out_file],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("CSDP execution failed") from exc
        out = completed.stdout
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
        return out_text

    def __latex__(self):
        return "SDP(%d, %d, %s)" % (len(self.C), len(self.A), self.solver)
