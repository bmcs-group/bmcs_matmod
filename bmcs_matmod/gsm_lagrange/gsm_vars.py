import sympy as sp

class Scalar(sp.Symbol):
    """
    Scalar variable for GSM models.
    Inherits from sympy.Symbol and adds optional codename attribute.
    """
    def __new__(cls, name, codename=None, **assumptions):
        obj = sp.Symbol.__new__(cls, name, **assumptions)
        obj.codename = codename if codename is not None else name
        return obj

class Vector(sp.Matrix):
    """
    Vector variable for GSM models.
    Inherits from sympy.Matrix and adds name and codename attributes.
    """
    def __new__(cls, name, elements, codename=None):
        mat = sp.Matrix.__new__(cls, len(elements), 1, elements)
        mat.name = name
        mat.codename = codename if codename is not None else name
        return mat

class Tensor(sp.Matrix):
    """
    Tensor variable for GSM models.
    Inherits from sympy.Matrix and adds name and codename attributes.
    """
    def __new__(cls, name, shape, elements, codename=None):
        mat = sp.Matrix.__new__(cls, *shape, elements)
        mat.name = name
        mat.codename = codename if codename is not None else name
        return mat
