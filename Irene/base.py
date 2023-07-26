r"""
This is the base module for all other objects of the package.

    + `LaTeX` returns a LaTeX string out of an `Irene` object.
    + `base` is the parent of all `Irene` objects.
"""


def LaTeX(obj):
    r"""
    Returns LaTeX representation of Irene's objects.
    """
    from sympy.core.core import ordering_of_classes
    from Irene import SDPRelaxations, SDRelaxSol, Mom
    inst = isinstance(obj, SDPRelaxations) or isinstance(
        obj, SDRelaxSol) or isinstance(obj, Mom)
    if inst:
        return obj.__latex__()
    elif isinstance(obj, tuple(ordering_of_classes)):
        from sympy import latex
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
        existsing = []
        # CVXOPT
        try:
            import cvxopt
            existsing.append('CVXOPT')
        except ImportError:
            pass
        if self.os == 'win32':
            from os.path import isfile
            # DSDP
            if 'dsdp' in self.Path:
                if isfile(self.Path['dsdp']):
                    existsing.append('DSDP')
            # SDPA
            if 'sdpa' in self.Path:
                if isfile(self.Path['sdpa']):
                    existsing.append('SDPA')
            if 'csdp' in self.Path:
                if isfile(self.Path['csdp']):
                    existsing.append('CSDP')
        else:
            # DSDP
            if self.which('dsdp5') is not None:
                existsing.append('DSDP')
            # SDPA
            if self.which('sdpa') is not None:
                existsing.append('SDPA')
            # CSDP
            if self.which('csdp') is not None:
                existsing.append('CSDP')
        return existsing
