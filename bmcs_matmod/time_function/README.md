# Concept: Symbolic-to-Executable Time Function Modeling

This module demonstrates a concept for bridging the gap between algebraic (symbolic) model definitions and their executable, numerical counterparts. The approach is motivated by the need for flexible, analytically defined model components—such as time functions—for use in material fatigue modeling and screening of dissipative mechanisms.

## General Workflow

1. **Algebraic Model Definition**  
   Models are first described analytically using symbolic variables and parameters (via SymPy). This enables clear, mathematical specification of model behavior, independent of implementation details.

2. **Parameter and Variable Management**  
   Each model component (e.g., a time function) defines its parameters in a `params` attribute. These parameters are mapped to both symbolic variables (for algebraic manipulation) and traits (for instance-level configuration).

3. **Symbolic Expression Specification**  
   Subclasses of `TimeFnBase` implement the `_get_symb_expr` method, which returns the symbolic expression representing the model's behavior (e.g., a piecewise function for step loading).

4. **Automatic Lambdification**  
   The symbolic expression, together with the symbolic variables and parameters, is automatically "lambdified"—converted into a fast, executable NumPy function. This enables efficient numerical evaluation of the model for given parameter values and input arrays.

5. **Unified Interface**  
   All time function classes provide a common interface:  
   - The `__call__` method maps an input time array to the evaluated time function, using the lambdified symbolic expression.
   - Model parameters are accessible as traits, allowing for easy configuration and introspection.

6. **Extensibility**  
   New types of time functions (e.g., step, monotonic, cyclic) are implemented as subclasses of `TimeFnBase`, specifying their unique parameters and symbolic expressions. This supports a wide range of model behaviors within a unified framework.

## Application Context

The `TimeFnBase` and its subclasses are designed as components of a broader modeling framework for fatigue characterization of materials. They enable:
- Systematic screening of material models to identify covered and missing dissipative mechanisms.
- Rapid prototyping and comparison of different time-dependent loading scenarios.
- A clear, maintainable workflow from mathematical model specification to executable code.

## Implementation Highlights

- **SymbolRegistry**: Ensures consistent reuse of symbolic variables across the framework.
- **Traits Integration**: Parameters are defined as traits, supporting validation, documentation, and UI integration.
- **Dynamic Symbolic Mapping**: The `__new__` method in `TimeFnBase` dynamically creates symbolic variables for all parameters, ensuring seamless algebraic-to-numeric transition.
- **Lambdification Pipeline**: Symbolic expressions are automatically converted to efficient numerical functions, minimizing manual coding and errors.
- **Subclass Customization**: Each time function subclass specifies its parameters and symbolic expression, enabling a wide variety of behaviors with minimal boilerplate.

## Example: Time Function Subclasses

- **StepLoading**: Piecewise constant function with a step at a specified time.
- **MonotonicAscending**: Linear ramp function.
- **TimeFnCycleSinus / Linear / WithRamps**: Various cyclic loading scenarios, each with their own symbolic parameterization.

## Implementation Approaches: Symbolic vs Numerical

The framework supports two primary approaches for implementing time functions, with a hybrid option combining the best of both:

### 1. Symbolic Approach
- **Implementation**: Override `symb_expr`, `collect_symb_params`, and `get_args` methods
- **Benefits**:
  - Mathematical clarity and direct correspondence to algebraic formulations
  - Support for symbolic manipulations (derivatives, integration)
  - Automatic code generation via lambdification
- **Use when**: Expressions are relatively simple and algebraically representable

### 2. Numerical Approach
- **Implementation**: Override `__call__` method directly
- **Benefits**:
  - Can handle complex operations not easily expressible symbolically
  - Works well with array inputs and complex control flow
  - Often more performant for complex operations
- **Use when**: Expressions involve arrays, complex control flow, or are difficult to represent symbolically
- **Example**: `TimeFnStepping` uses this approach due to its array-based parameters

### 3. Hybrid Approach
- **Implementation**: Implement both symbolic and numerical methods with fallback mechanisms
- **Benefits**:
  - Attempts symbolic representation when possible
  - Falls back to direct numerical computation when needed
  - Maintains mathematical clarity while ensuring robust execution
- **Example**: `TimeFnOverlay` combines functions symbolically when possible, falls back to direct computation when needed

### Design Considerations

When implementing new time functions, consider:

1. **Complexity**: Is the algebraic expression manageable symbolically?
2. **Parameters**: Are you using arrays or other complex parameters?
3. **Composition**: Does your function combine multiple other functions?
4. **Derivatives**: Do you need symbolic derivatives of your function?

For complex composite models, the hybrid approach offers the best balance between algebraic clarity and practical computation, allowing for symbolic derivations while ensuring computational robustness.

---

This concept provides a robust foundation for generalizing analytically defined model components, supporting both symbolic manipulation and high-performance numerical evaluation within a unified, extensible framework.
