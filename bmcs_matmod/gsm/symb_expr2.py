import os
import dill
import sympy as sp
import hashlib


class SymbExpr:
    symb_variables = []
    symb_model_params = []
    symb_expressions = []
    cache_dir = '_symbolic_cache'

    def __init__(self, *args, **kw):
        # Check or create cache directory
        self.cache_path = os.path.join(self.cache_dir, f'{self.__class__.__name__}_symbolic.pkl')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize symbolic expressions
        if not self._load_symbolic_expressions():
            self._derive()  # Derive only if not already cached
            self._cache_symbolic_expressions()

        # Define the lambdified functions using cached or fresh expressions
        self._define_callable_functions()

    def _derive(self):
        """Override this method in subclasses to define symbolically derived expressions once."""
        raise NotImplementedError("Symbolic derivation needs to be defined in subclass.")

    def _load_symbolic_expressions(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as file:
                data = dill.load(file)
                for attr, value in data.items():
                    setattr(self, attr, value)
            return True
        return False

    def _cache_symbolic_expressions(self):
        # Serialize symbolic expressions
        data = {name: getattr(self, name) for name in self.symb_expressions}
        with open(self.cache_path, 'wb') as file:
            dill.dump(data, file)

    def _define_callable_functions(self):
        # Assuming symbolic expressions are already loaded
        for expr in self.symb_expressions:
            symbolic_expr = getattr(self, expr)
            # Assumption: default_symbols contains all necessary symbols
            callable_function = sp.lambdify(self.symb_variables + self.symb_model_params, symbolic_expr, 'numpy')
            setattr(self, f'get_{expr}', callable_function)

