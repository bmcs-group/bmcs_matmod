import sympy as sp
from typing import List, Optional, Any, Union, Literal
from .basic_unit_db import get_base_unit

class GSMVariableBase:
    """
    Base class for GSM variables providing shared functionality.
    
    Provides common attributes and functionality shared by Scalar, Vector, and Tensor:
    - name: Primary symbol name used for SymPy operations and LaTeX rendering
    - codename: Alternative name for code generation, lambdification, and variable mapping
    - desc: Human-readable description of the variable
    - base_unit: Base SI unit from unit database lookup
    - unit: User-specified unit (overrides base_unit if provided)
    - rank: Tensor rank (0=scalar, 1=vector, 2=tensor)
    - dimensionality: Spatial dimensionality (1d, 2d, 3d)
    
    LaTeX Rendering:
    The 'name' parameter supports LaTeX syntax directly. Use raw strings for LaTeX symbols:
    - name='eps', codename='strain' -> symbol 'eps' for both SymPy and code
    - name=r'\\varepsilon', codename='eps' -> LaTeX '\\varepsilon' for display, 'eps' for code
    - name=r'\\sigma', codename='sig' -> LaTeX '\\sigma' for display, 'sig' for code
    
    The LaTeX rendering is handled automatically by SymPy's latex() function on the name.
    The codename is used for:
    - Variable mapping in numerical computations
    - Code generation and lambdification
    - Dictionary keys in state vectors and parameter sets
    """
    
    def _set_common_attributes(self, name: str, codename: Optional[str] = None, 
                             desc: Optional[str] = None, unit: Optional[str] = None,
                             rank: int = 0, dimensionality: Literal[1, 2, 3] = 1) -> None:
        """Set common attributes for GSM variables."""
        self.codename = codename if codename is not None else name  # type: ignore[attr-defined]
        self.desc = desc  # type: ignore[attr-defined]
        
        # Set units - base_unit from database, unit from user (overrides base_unit)
        self.base_unit = get_base_unit(self.codename)  # type: ignore[attr-defined]
        self.unit = unit if unit is not None else self.base_unit  # type: ignore[attr-defined]
        
        # Set rank and dimensionality
        self.rank = rank  # type: ignore[attr-defined]
        self.dimensionality = dimensionality  # type: ignore[attr-defined]


class Scalar(sp.Symbol, GSMVariableBase):
    """
    Scalar variable for GSM models.
    
    Inherits from sympy.Symbol and GSMVariableBase, adding codename, desc, unit,
    and dimension attributes for code generation and documentation.
    
    Automatically sets:
    - rank = 0 (scalar)
    - dimensionality = 1 (always 1d for scalars)
    
    Usage Examples:
    >>> eps = Scalar('eps', desc='strain')  
    >>> strain = Scalar(r'\\varepsilon', 'eps', 'strain variable')  
    >>> stress = Scalar(r'\\sigma', 'sig', 'stress variable', unit='MPa')
    """
    def __new__(cls, name: str, codename: Optional[str] = None, 
                desc: Optional[str] = None, unit: Optional[str] = None,
                **assumptions: Any) -> 'Scalar':
        obj = sp.Symbol.__new__(cls, name, **assumptions)
        obj._set_common_attributes(name, codename, desc, unit, rank=0, dimensionality=1)
        return obj


class Vector(sp.Matrix, GSMVariableBase):
    """
    Vector variable for GSM models.
    
    Inherits from sympy.Matrix and GSMVariableBase, adding name, codename, desc, unit,
    and dimension attributes.
    
    Automatically sets:
    - rank = 1 (vector)
    - dimensionality = user-specified (1d, 2d, or 3d)
    
    Usage Examples:
    >>> u = Vector('u', [u_x, u_y], dimensionality=2, desc='displacement vector')
    >>> force = Vector('F', [F_x, F_y, F_z], dimensionality=3, unit='kN')
    """
    def __new__(cls, name: str, elements: List[Union[sp.Symbol, sp.Expr]], 
                codename: Optional[str] = None, desc: Optional[str] = None,
                unit: Optional[str] = None, dimensionality: Literal[1, 2, 3] = 1) -> 'Vector':
        mat = sp.Matrix.__new__(cls, len(elements), 1, elements)
        mat.name = name  # type: ignore[attr-defined]
        mat._set_common_attributes(name, codename, desc, unit, rank=1, dimensionality=dimensionality)
        return mat


class Tensor(sp.Matrix, GSMVariableBase):
    """
    Tensor variable for GSM models.
    
    Inherits from sympy.Matrix and GSMVariableBase, adding name, codename, desc, unit,
    and dimension attributes.
    
    Automatically sets:
    - rank = 2 (tensor)
    - dimensionality = user-specified (1d, 2d, or 3d)
    
    Usage Examples:
    >>> stress = Tensor('sigma', (3,3), elements, dimensionality=3, desc='stress tensor')
    >>> strain = Tensor('eps', (2,2), elements, dimensionality=2, unit='1')
    """
    def __new__(cls, name: str, shape: tuple, elements: List[Union[sp.Symbol, sp.Expr]], 
                codename: Optional[str] = None, desc: Optional[str] = None,
                unit: Optional[str] = None, dimensionality: Literal[1, 2, 3] = 1) -> 'Tensor':
        mat = sp.Matrix.__new__(cls, *shape, elements)
        mat.name = name  # type: ignore[attr-defined]
        mat._set_common_attributes(name, codename, desc, unit, rank=2, dimensionality=dimensionality)
        return mat


def markdown_vars_table(vars_list: List[Union[Scalar, Vector, Tensor]]) -> str:
    """
    Create a markdown table representation of variable symbols and descriptions.
    Includes units, rank, dimensionality, and key assumptions.

    Args:
        vars_list: List of Scalar, Vector, or Tensor variable instances

    Returns:
        str: Markdown table with all variable attributes

    Example:
        >>> variables = [temp, entropy, stress_tensor]
        >>> table_md = create_md_vars_table(variables)
    """
    # Build markdown table header with new columns
    markdown_content = "| Description | LaTeX Symbol | Codename | Base Unit | Unit | Rank | Dim | Key Assumptions |\n"
    markdown_content += "|-------------|--------------|----------|-----------|------|------|-----|------------------|\n"
    
    # Define assumption priority mapping
    assumption_priority = ['positive', 'nonnegative', 'real', 'finite']
    
    for var in vars_list:
        # Get description
        desc = getattr(var, 'desc', 'No description')
        
        # Get LaTeX representation - handle different types
        try:
            if isinstance(var, Scalar):
                latex_symbol = f"${sp.latex(var)}$"
            else:
                # For Vector and Tensor, use the name attribute
                name = getattr(var, 'name', str(var))
                latex_symbol = f"${name}$"
        except Exception:
            # Fallback to name or string representation
            latex_symbol = f"${getattr(var, 'name', str(var))}$"
        
        # Get codename - wrapped in backticks for code formatting
        codename = f"`{getattr(var, 'codename', 'N/A')}`"
        
        # Get units
        base_unit = getattr(var, 'base_unit', 'N/A')
        unit = getattr(var, 'unit', 'N/A')
        
        # Get rank and dimensionality
        rank = getattr(var, 'rank', 'N/A')
        dimensionality = getattr(var, 'dimensionality', 'N/A')
        
        # Get key assumptions using mapping approach
        if hasattr(var, 'assumptions0'):
            key_assumptions = [
                assumption for assumption in assumption_priority
                if var.assumptions0.get(assumption, False)
            ]
        else:
            key_assumptions = []
        
        assumptions_str = ", ".join(key_assumptions) if key_assumptions else "none"
        
        markdown_content += f"| {desc} | {latex_symbol} | {codename} | {base_unit} | {unit} | {rank} | {dimensionality}d | {assumptions_str} |\n"
    
    return markdown_content
