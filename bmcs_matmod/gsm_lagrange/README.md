# Generalized Standard Material Framework

This package implements a symbolic-to-executable (symb-exec) framework for thermodynamically consistent material modeling. It enables the formulation of complex material models based on thermodynamic potentials and constraints, which are automatically transformed into efficient numerical implementations.

## Architectural Design

The framework follows a clear separation of concerns between symbolic model definitions and their executable numerical implementations:

### 1. Symbolic Layer (Class Level)
- **Definition**: Class-level symbolic expressions and relationships
- **Implementation**: Using `__init_subclass__` to perform symbolic calculations once at class definition time
- **Advantages**: Clear, mathematical formulation; automatic derivation of constitutive equations

### 2. Executable Layer (Instance Level)
- **Definition**: Instance-specific parameter values and numerical implementations
- **Implementation**: Lambdified functions for efficient computation
- **Advantages**: High-performance numerical evaluation; caching of computational results

### 3. Bridging Components
- **Definition**: Components that connect symbolic models with concrete parameters
- **Implementation**: `GSMMaterialModel` class (forthcoming)
- **Advantages**: Trait-based interface; automatic parameter mapping

## Core Components

### GSMBase

A class that provides a front-end for both Helmholtz and Gibbs free energy formulations:

```python
class MyModel(GSMBase):
    # Symbolic variables and parameters
    eps = sp.Symbol('\\varepsilon', real=True)
    E = sp.Symbol('E', positive=True)
    
    # Helmholtz free energy
    F_expr = 0.5 * E * eps**2
    
    # Define engine
    F_engine = GSMMPDP(...)
```

Key features:
- Class-level symbolic initialization via `__init_subclass__`
- Consistent method naming for Helmholtz (`get_F_*`) and Gibbs (`get_G_*`) formulations
- Automatic Legendre transform between Helmholtz and Gibbs potentials

### GSMMPDP

The symbolic engine responsible for:

- Managing thermodynamic potentials and state variables
- Deriving evolution equations via the minimum principle of dissipation potential
- Implementing time-stepping algorithms and return-mapping procedures

## Workflow: From Symbolic to Executable

1. **Define a Material Model**:
   - Subclass `GSMBase`
   - Define symbolic variables, parameters, and potentials
   - Class-level symbolic computations happen automatically

2. **Use the Material Model**:
   - Create an instance with specific parameter values
   - Call methods like `get_F_response` or `get_G_response`
   - Analyze and visualize results

The framework handles the complex transition from mathematical formulations to numerical computations, ensuring thermodynamic consistency throughout.

## Implementation Approach

### Symbolic Computations at Class Level

The recent implementation uses Python's `__init_subclass__` hook to perform all symbolic derivations when a subclass of `GSMBase` is defined:

```python
@classmethod
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    
    if hasattr(cls, 'F_engine') and cls.F_engine is not None:
        # Build parameter codenames
        cls.param_codenames = cls._build_param_codenames()
        
        # Calculate symbolic expressions
        cls._calculate_symbolic_expressions()
        
        # Initialize the Gibbs engine
        cls._initialize_gibbs_engine()
```

This approach offers several advantages:
- Symbolic computation happens once at class definition time
- Clear separation between symbolic expressions and numerical computations
- Improved performance as symbolic derivations aren't repeated for each instance

### Consistent Method Naming

The framework now uses consistent method naming for Helmholtz and Gibbs formulations:
- `get_F_sig`, `get_F_response`, `get_F_Sig` for Helmholtz-based methods
- `get_G_eps`, `get_G_response`, `get_G_Sig` for Gibbs-based methods

Legacy methods (`get_sig`, `get_response`, `get_Sig`) are maintained for backward compatibility but will be deprecated in future versions.

## Future Developments

### GSMMaterialModel

A forthcoming addition to the framework is the `GSMMaterialModel` class, which will:
- Create trait-based interfaces for material parameters
- Enable parameter studies and visualization
- Simplify the bridge between symbolic models and concrete parameter values

### Extended Visualization

Future versions will include enhanced visualization capabilities:
- Interactive exploration of material behavior
- Comparison of different material models
- Parameter sensitivity analysis

### Wider Integration

Integration with other components of the BMCS framework:
- Time functions for loading scenarios
- Structural models for component-level analysis
- Finite element models for spatial discretization
