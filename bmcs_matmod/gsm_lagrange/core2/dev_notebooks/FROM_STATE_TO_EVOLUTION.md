# From State Functions to Evolutionary Material Models

## Current Framework Status

### What We Have: Static State Function Description
The current `gsm_state_fn.py` implementation provides:
- **Static thermodynamic state representation** with four fundamental potentials (U, F, H, G)
- **Variable organization** distinguishing thermal (T, S) and mechanical (σ, ε) natural/conjugate pairs
- **Intensive vs extensive variable classification** for thermodynamic consistency
- **Legendre transformation capability** for state function interconversion
- **Markdown rendering** for constitutive relations and variable tables

**Key Limitation**: This framework only describes the **current state** of the system - it cannot evolve in time or handle irreversible processes.

### What We Want: Time-Evolving Material Models
The next development phase aims to extend the framework to handle:
- **Multi-physics processes** (thermal + mechanical + dissipative)
- **Time evolution** of internal variables
- **Irreversible state transitions** governed by thermodynamic principles
- **Executable material models** for integration into nonlinear simulators

## Proposed Framework Extension

### 1. From Static to Dynamic: Additional Expressions

The extension from static state functions to evolutionary models requires four additional symbolic expressions, all defined in terms of the natural and conjugate variables from `gsm_state_fn.py`:

#### 1.1 Threshold Function (`f_expr`)
```
f_expr: sp.Expr
```
- **Purpose**: Delineates reversible vs irreversible states
- **Based on**: Classical plasticity theory naming (yield function)
- **Constraint**: Single inequality constraint `f ≤ 0`
- **Usage**: Determines whether system remains elastic or enters inelastic regime

#### 1.2 External Flow Potential (`phi_ext_expr`)  
```
phi_ext_expr: sp.Expr
```
- **Purpose**: Extends threshold function to define evolution direction
- **Combined potential**: `φ = f_expr + phi_ext_expr`
- **Usage**: Provides flow rule for irreversible evolution (∂φ/∂Sig gives evolution direction)

#### 1.3 Equality Constraints (`h_k`)
```
h_k: List[sp.Expr]
```
- **Purpose**: Sequence of equality constraints
- **Index k**: Indicates constraint dimensionality/multiplicity
- **Usage**: Additional kinematic or thermodynamic constraints

#### 1.4 Rate Bounds (`dot_var_bound_expr`)
```
dot_var_bound_expr: sp.Expr
```
- **Purpose**: Direct constraints on variable evolution rates
- **Example**: Viscous strain rate cannot exceed total strain rate
- **Usage**: Physical bounds on evolution velocities

### 2. Integration with Existing Framework

#### 2.1 Variable Alignment Strategy
The integration will align the comprehensive `gsm_symb_def.py` machinery with the generalized `gsm_state_fn.py` variable system:

**Current `gsm_symb_def.py` approach**:
- Helmholtz-based with `eps_vars` (strain) as natural variables
- `sig_vars` (stress) as conjugate variables
- Framework prepared for thermal variables but not yet integrated

**Proposed integration**:
- Use SymPy variables from `gsm_state_fn.py` for strain/stress definitions
- Map to `th_x/th_y` (thermal) and `mc_x/mc_y` (mechanical) variables
- Leverage existing derivation machinery for discrete time stepping

#### 2.2 Thermodynamic Consistency
The framework will maintain thermodynamic rigor through:
- **First Law**: Energy balance equations
- **Second Law**: Maximum dissipation principle for irreversible evolution
- **Automatic derivation**: Discrete time stepping scheme from thermodynamic principles

### 3. Existing Implementation Reference

#### 3.1 GSMSymbDef Structure (Target Pattern)
The `gsm_symb_def.py` already provides the target structure:
- Pure symbolic definitions without embedded engines
- Comprehensive derivation machinery for time discretization
- Cached properties for all derived expressions
- Validation and rendering capabilities

#### 3.2 GSMEngine Integration (Execution Target)  
The `gsm_engine.py` and `gsm_def.py` demonstrate:
- Numerical execution of symbolic expressions
- Lambdification for computational efficiency
- State evolution algorithms
- Integration interfaces for simulators

### 4. Development Roadmap

#### Phase 1: Framework Extension

- Create thermodynamic potential subclasses (`GSMHelmholtzFn`, `GSMGibbsFn`, `GSMEnthalpyFn`, `GSMInternalEnergyFn`)
- Implement explicit variable attributes (T, S, eps, sig) with correct mapping to general framework (th_x, th_y, mc_x, mc_y)
- Extend base `GSMStateFn` to include evolutionary expressions (`f_expr`, `phi_ext_expr`, `h_k`, `dot_var_bound_expr`)
- Ensure constructors establish proper linking between specific and general variable representations
- Maintain backward compatibility with static state function usage
- Add validation for evolutionary expression consistency across all thermodynamic potentials

#### Phase 2: Derivation Engine Integration
- Integrate `gsm_symb_def.py` derivation machinery with extended `GSMStateFn`
- Map generalized thermal/mechanical variables to specific symbolic definitions
- Ensure thermodynamic consistency across all four state function types

#### Phase 3: Execution Engine
- Create executable material models from symbolic definitions
- Generate discrete time stepping algorithms
- Provide interfaces for FE and discrete simulators

#### Phase 4: Thermodynamic Box Enhancement
- Extend round-trip transformation capability to evolutionary models
- Validate conservation laws and thermodynamic consistency
- Demonstrate multi-physics coupling capabilities

## Target Architecture

### Desired Integration Pattern

#### Core Class Hierarchy
```
GSMStateFn (abstract base class)
    ├── GSMHelmholtzFn (F potential: T, ε as natural variables)
    ├── GSMGibbsFn (G potential: T, σ as natural variables) 
    ├── GSMEnthalpyFn (H potential: S, σ as natural variables)
    └── GSMInternalEnergyFn (U potential: S, ε as natural variables)
        ↓ (explicit variable mapping to general framework)
GSMEvolutionDef (symbolic evolutionary expressions - universal)
    ↓ (derivation machinery from gsm_symb_def.py - universal)
GSMExecutionEngine (numerical time stepping - universal)
    ↓ (simulator interface)
Material Model Integration (FE/discrete frameworks)
```

#### Thermodynamic State Function Subclasses
Each thermodynamic potential will have its own specialized subclass that explicitly defines the natural and conjugate variables:

**GSMHelmholtzFn** (Free Energy: F(T, ε)):
- **Natural variables**: `T` (temperature), `eps` (strain) 
- **Conjugate variables**: `S` (entropy), `sig` (stress)
- **Mapping**: `th_x=T, th_y=S, mc_x=eps, mc_y=sig`

**GSMGibbsFn** (Gibbs Free Energy: G(T, σ)):
- **Natural variables**: `T` (temperature), `sig` (stress)
- **Conjugate variables**: `S` (entropy), `eps` (strain)  
- **Mapping**: `th_x=T, th_y=S, mc_x=sig, mc_y=eps`

**GSMEnthalpyFn** (Enthalpy: H(S, σ)):
- **Natural variables**: `S` (entropy), `sig` (stress)
- **Conjugate variables**: `T` (temperature), `eps` (strain)
- **Mapping**: `th_x=S, th_y=T, mc_x=sig, mc_y=eps`

**GSMInternalEnergyFn** (Internal Energy: U(S, ε)):
- **Natural variables**: `S` (entropy), `eps` (strain) 
- **Conjugate variables**: `T` (temperature), `sig` (stress)
- **Mapping**: `th_x=S, th_y=T, mc_x=eps, mc_y=sig`

#### Universal Derivation and Execution
The key advantage of this approach is that the **derivation machinery in GSMEvolutionDef becomes universal** across all four thermodynamic potentials:

- **Symbolic expressions** (`f_expr`, `phi_ext_expr`, `h_k`, `dot_var_bound_expr`) use the generalized variable names
- **Derivation algorithms** operate on abstract `th_x/th_y` and `mc_x/mc_y` variables
- **Execution engines** work with any thermodynamic potential type
- **Material models** inherit the correct thermodynamic framework automatically

### Key Design Principles

#### Class Roles and Responsibilities

**GSMStateFn Subclasses (Thermodynamic Specialization)**:
- **Purpose**: Provide thermodynamic potential-specific variable naming and mapping
- **Constructor role**: Establish correct linking between explicit variables (T, S, eps, sig) and general framework (th_x, th_y, mc_x, mc_y)
- **Example**: `GSMHelmholtzFn.__init__()` sets `self.th_x = self.T` and `self.mc_x = self.eps`
- **Inheritance**: All inherit evolutionary expression capability (`f_expr`, `phi_ext_expr`, etc.)
- **Benefit**: Type safety and thermodynamic correctness without code duplication

**GSMEvolutionDef (Universal Symbolic Engine)**:
- **Purpose**: Generate all symbolic expressions for time evolution from thermodynamic principles
- **Input**: Any GSMStateFn subclass with its variable mappings
- **Processing**: Uses general variables (th_x, th_y, mc_x, mc_y) for derivations
- **Output**: Complete symbolic framework (Lagrangian, optimality conditions, time discretization)
- **Universality**: Works identically for all four thermodynamic potentials

**GSMExecutionEngine (Universal Numerical Engine)**:
- **Purpose**: Execute symbolic expressions numerically with optimal performance
- **Lambdification**: Convert symbolic expressions to compiled numerical functions
- **State evolution**: Implement Newton-Raphson and other numerical algorithms
- **Interface**: Provide standardized material model API for simulators
- **Sharing**: Same engine works for all thermodynamic potentials and material models

#### Integration Benefits
This architecture provides several key advantages:

1. **Code Reusability**: Derivation and execution engines are shared across all thermodynamic potentials
2. **Type Safety**: Explicit variable naming (T, S, eps, sig) prevents thermodynamic errors
3. **Maintainability**: Changes to derivation algorithms automatically benefit all material models
4. **Extensibility**: New thermodynamic potentials require only variable mapping, not new engines
5. **Consistency**: Guaranteed thermodynamic consistency across all implementations

#### 1. Generalization
- Support all four thermodynamic potentials (U, F, H, G)
- Handle arbitrary combinations of thermal/mechanical loading
- Accommodate various internal variable types

#### 2. Thermodynamic Consistency
- Automatic enforcement of thermodynamic laws
- Maximum dissipation principle for irreversible evolution
- Conservation law validation

#### 3. Computational Efficiency
- Symbolic-to-numerical translation
- Optimized execution engines
- Minimal computational overhead

#### 4. Simulator Integration
- Clean interfaces for various simulator types
- Standardized material model APIs
- Robust error handling and validation

## Implementation Strategy

### 1. Incremental Development
- Start with Helmholtz potential (most common in mechanics)
- Gradually extend to other potentials
- Maintain working examples at each stage

### 2. Validation Through Examples
- Use existing material models for verification
- Compare with analytical solutions where available
- Demonstrate thermodynamic consistency through round-trip tests

### 3. Documentation and Testing
- Comprehensive notebooks demonstrating capabilities
- Unit tests for all symbolic derivations
- Integration tests with sample simulators

## Success Metrics

The completed framework should enable:
- **Rapid material model development** from thermodynamic potentials
- **Automatic time discretization** with guaranteed stability
- **Multi-physics coupling** (thermal-mechanical-chemical)
- **Simulator integration** with minimal coding overhead
- **Thermodynamic validation** through built-in consistency checks

## Connection to Existing Work

This development builds directly on:
- **GSMStateFn framework**: Variable organization and state function handling
- **GSMSymbDef machinery**: Symbolic derivation and time discretization
- **GSMEngine execution**: Numerical implementation and lambdification
- **Thermodynamic box**: Round-trip validation and transformation testing

The goal is to unify these components into a coherent framework that simplifies material model development while maintaining full thermodynamic rigor and computational efficiency.