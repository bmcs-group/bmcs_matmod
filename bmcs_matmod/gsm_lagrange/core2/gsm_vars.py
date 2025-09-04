import sympy as sp
from typing import List, Optional, Any, Union

class Scalar(sp.Symbol):
    """
    Scalar variable for GSM models.
    Inherits from sympy.Symbol and adds optional codename attribute.
    """
    def __new__(cls, name: str, codename: Optional[str] = None, **assumptions: Any) -> 'Scalar':
        obj = sp.Symbol.__new__(cls, name, **assumptions)
        obj.codename = codename if codename is not None else name  # type: ignore[attr-defined]
        return obj

class Vector(sp.Matrix):
    """
    Vector variable for GSM models.
    Inherits from sympy.Matrix and adds name and codename attributes.
    """
    def __new__(cls, name: str, elements: List[Union[sp.Symbol, sp.Expr]], codename: Optional[str] = None) -> 'Vector':
        mat = sp.Matrix.__new__(cls, len(elements), 1, elements)
        mat.name = name  # type: ignore[attr-defined]
        mat.codename = codename if codename is not None else name  # type: ignore[attr-defined]
        return mat

class Tensor(sp.Matrix):
    """
    Tensor variable for GSM models.
    Inherits from sympy.Matrix and adds name and codename attributes.
    """
    def __new__(cls, name: str, shape: tuple, elements: List[Union[sp.Symbol, sp.Expr]], codename: Optional[str] = None) -> 'Tensor':
        mat = sp.Matrix.__new__(cls, *shape, elements)
        mat.name = name  # type: ignore[attr-defined]
        mat.codename = codename if codename is not None else name  # type: ignore[attr-defined]
        return mat
