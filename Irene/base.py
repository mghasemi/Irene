class base(object):
    r"""
    All the modules in `Irene` extend this class which perform some common
    tasks such as checking existence of certain softwares.
    """

    def __init__(self):
        pass

    def which(self, program):
        r"""
        Check the availability of the `program` system-wide.
        Returns the path of the program if exists and returns 
        'None' otherwise.
        """
        import os

        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

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
