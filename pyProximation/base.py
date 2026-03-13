class Foundation:
    """
    This class contains common features of all modules.
    """

    def __init__(self):
        pass

    def DetSymEnv(self):
        """
        Returns ``['sympy']`` when sympy is available.
        """
        try:
            import sympy  # noqa: F401
            return ['sympy']
        except ImportError:
            return []

    def CommonSymFuncs(self, env):
        if env != 'sympy':
            raise ValueError("Only 'sympy' is supported as symbolic environment.")
        from sympy import expand, sqrt, sin, cos, pi, diff, Symbol
        self.expand = expand
        self.sqrt = sqrt
        self.sin = sin
        self.cos = cos
        self.pi = pi
        self.diff = diff
        self.Symbol = Symbol
