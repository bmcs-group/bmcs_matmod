import sympy as sp
import re

class Cymbol(sp.Symbol):
    '''Add a codename attribute to the sympy symbol to
    enable
    1) latex names with upper index or lower index
    2) transformation of symbol arrays to function attributes
    (e.g. state variable arrays passed as dictionaries to
     the corrector-predictor functions)
    3) generate lambdified function that can be inspected using
    import inspect
    inspect.getsource(lambdified_function)
    '''
    def __init__(self, name, codename='', **assumptions):
        super(Cymbol, self).__init__()
        if codename:
            self.codename = codename
        else:
            self.codename = name

def _print_Symbol(self, expr):
    CodePrinter = sp.printing.codeprinter.CodePrinter
    if hasattr(expr, 'codename'):
        name = expr.codename
    else:
        name = super(CodePrinter, self)._print_Symbol(expr)
    return re.sub(r'[\\\{\}]', '', name)

sp.printing.codeprinter.CodePrinter._print_Symbol = _print_Symbol

from sympy.utilities.codegen import codegen

def ccode(cfun_name, sp_expr, cfile):
    '''Generate c function cfun_name for expr and directive name cfile
    '''
    return codegen((cfun_name, sp_expr), 'C89', cfile + '_' + cfun_name)


