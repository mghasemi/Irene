r"""
This is the base module for all other objects of the package.

    + `LaTeX` returns a LaTeX string out of an `Irene` object.
    + `base` is the parent of all `Irene` objects.
"""


def LaTeX(obj):
    r"""
    Returns LaTeX representation of Irene's objects.
    """
    has_latex = hasattr(obj, '__latex__') and callable(getattr(obj, '__latex__'))
    if has_latex:
        return obj.__latex__()
    else:
        from sympy import Basic, latex
    if isinstance(obj, Basic):
        return latex(obj)


class base(object):
    r"""
    All the modules in `Irene` extend this class which perform some common
    tasks such as checking existence of certain software.
    """

    def __init__(self):
        from sys import platform
        self.os = platform
        if self.os == 'win32':
            import os
            BASE = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)) + os.sep
            self.Path = dict(csdp=BASE + "csdp.exe", sdpa=BASE + "sdpa.exe")
        else:
            self.Path = dict(csdp="csdp", sdpa="sdpa")

    @staticmethod
    def which(program):
        r"""
        Check the availability of the `program` system-wide.
        Returns the path of the program if exists and returns 
        'None' otherwise.
        """
        import os

        def is_exe(filepath):
            return os.path.isfile(filepath) and os.access(filepath, os.X_OK)

        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
        return None

    def AvailableSDPSolvers(self):
        r"""
        find the existing sdp solvers.
        """
        existing = []
        # CVXOPT
        try:
            import cvxopt
            existing.append('CVXOPT')
        except ImportError:
            pass
        for solver_name in ('DSDP', 'SDPA', 'CSDP'):
            if self._solver_is_available(solver_name):
                existing.append(solver_name)
        return existing

    def _solver_is_available(self, solver_name):
        r"""
        Return True if ``solver_name`` is available in the current platform setup.
        """
        solver_map = {
            'DSDP': ('dsdp', 'dsdp5'),
            'SDPA': ('sdpa', 'sdpa'),
            'CSDP': ('csdp', 'csdp'),
        }
        if solver_name not in solver_map:
            return False

        path_key, binary_name = solver_map[solver_name]
        if self.os == 'win32':
            from os.path import isfile
            solver_path = self.Path.get(path_key)
            return bool(solver_path) and isfile(solver_path)

        return self.which(binary_name) is not None
