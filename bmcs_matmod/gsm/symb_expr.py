
import os
import dill
import functools
import sympy as sp
import traits.api as tr
from collections.abc import Iterable

def lambdify_and_cache(cache_dir='_lambdified_cache'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, expr_name, symbols, param_symbols):
            # Generate the filename based on class name and expression name
            class_name = self.__class__.__name__
            filename = os.path.join(cache_dir, f"{class_name}_{expr_name}.pkl")

            # Check if the cache directory exists, if not, create it
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            # Check if the file already exists
            if os.path.exists(filename):
                # Load the lambdified function from the file
                with open(filename, 'rb') as f:
                    lambdified_func = dill.load(f)
            else:
                # Call the original function to get the lambdified expression
                lambdified_func = func(self, expr_name, symbols, param_symbols)
                # Save the lambdified function to a file
                with open(filename, 'wb') as f:
                    dill.dump(lambdified_func, f)

            return lambdified_func
        return wrapper
    return decorator

class SymbExpr(tr.HasStrictTraits):
    symb_variables = []
    symb_model_params = []
    symb_expressions = []
    model = tr.WeakRef
    cse = tr.Bool(False)

    def get_model_params(self):
        return tuple(
            getattr(self.model, param_name)
            for param_name in self.symb_model_params
        )

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        default_symbols = tuple(
            getattr(self, sym_name) for sym_name in self.symb_variables
        )
        for expression in self.symb_expressions:
            if isinstance(expression, Iterable):
                expr_name, sym_names = expression
                symbols = tuple(getattr(self, sym_name) for sym_name in sym_names)
            elif isinstance(expression, str):
                expr_name = expression
                symbols = default_symbols
            else:
                raise TypeError(
                    'Expected name of expression attribute with a list of variables'
                )
            param_symbols = tuple(
                getattr(self, model_param)
                for model_param in self.symb_model_params
            )
            expr = getattr(self, expr_name)
            self._define_lambdified_callable(expr_name, symbols, param_symbols)

    @lambdify_and_cache()
    def _lambdify_expression(self, expr_name, symbols, param_symbols):
        expr = getattr(self, expr_name)
        return sp.lambdify(symbols + param_symbols, expr, 'numpy', cse=self.cse)

    def _define_lambdified_callable(self, expr_name, symbols, param_symbols):
        callable = self._lambdify_expression(expr_name, symbols, param_symbols)

        def define_callable(callable):
            def on_the_fly(*args):
                all_args = args + self.get_model_params()
                return callable(*all_args)
            return on_the_fly

        self.add_trait(f'get_{expr_name}', tr.Callable(define_callable(callable)))


class InjectSymbExpr(tr.HasStrictTraits):
    '''
    Inject expressions into a model class
    '''

    symb_class = tr.Type

    symb = tr.Instance(SymbExpr)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.symb = self.symb_class(model = self)
