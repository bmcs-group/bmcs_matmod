# Variable Naming Conventions and Architecture

## Overview

This document describes the naming conventions and architectural relationships between the core classes in the BMCS Material Modeling framework, specifically for the GSM (Generalized Standard Material) module.

## Core Architecture Components

### 1. GSMSymbDef (Symbolic Definitions)
- **Purpose**: Defines the symbolic mathematical expressions for material behavior
- **Variable Convention**: `gsm_def` or `<model_type>_gsm_def` (e.g., `ed_gsm_def`, `epd_gsm_def`)
- **Responsibilities**:
  - Define free energy functions (F)
  - Define dissipation criteria (f)
  - Provide symbolic derivatives and expressions
  - Validate symbolic consistency
- **Example Usage**:
  ```python
  ed_gsm_def = GSM1D_ED()  # Elastic damage symbolic definition
  epd_gsm_def = GSM1D_EPD()  # Elastoplastic damage symbolic definition
  ```

### 2. GSMEngine (Numerical Engine)
- **Purpose**: Handles numerical computation for single time-step integration
- **Variable Convention**: `engine` or `<model_type>_engine`
- **Responsibilities**:
  - Execute single time-step state updates
  - Numerical evaluation of symbolic expressions
  - State vector management
- **Example Usage**:
  ```python
  engine = GSMEngine(ed_gsm_def)
  new_state = engine.get_state_n1(current_state)
  ```

### 3. GSMModel (High-Level Model Interface)
- **Purpose**: Provides complete material model interface combining symbolic definitions with time integration
- **Variable Convention**: `model` or `<model_type>_model`
- **Responsibilities**:
  - Time series integration (get_response method)
  - Model validation and parameter management
  - Interface between symbolic definitions and practical usage
- **Example Usage**:
  ```python
  model = GSMModel(ed_gsm_def, material_params)
  response = model.get_response(time_series, initial_state)
  ```

### 4. MaterialParams (Parameter Repository)
- **Purpose**: Manages material parameters and their validation
- **Variable Convention**: `params` or `material_params`
- **Responsibilities**:
  - Parameter storage and validation
  - Bounds checking
  - Model compatibility verification
- **Example Usage**:
  ```python
  params = MaterialParams(E=30000, gamma_T=100, K=10, S=0.005)
  model = GSMModel(ed_gsm_def, params)
  ```

## Naming Convention Rationale

### Why `gsm_def` instead of `model`?

The distinction between `gsm_def` and `model` is crucial for architectural clarity:

- **`gsm_def`** (GSMSymbDef instances): Contains only symbolic mathematical definitions
  - Pure symbolic expressions
  - No numerical computation capabilities
  - Mathematical consistency validation
  - Foundation for numerical implementations

- **`model`** (GSMModel instances): Complete functional material models
  - Combines symbolic definitions with numerical engines
  - Handles time integration and response calculation
  - Manages material parameters
  - Provides user-facing interface

### Collection Naming

When working with multiple definitions or models:
- **`gsm_defs`**: Dictionary or list of symbolic definitions
- **`models`**: Dictionary or list of complete material models

Example:
```python
# Collection of symbolic definitions
gsm_defs = {
    'ed': GSM1D_ED(),
    'epd': GSM1D_EPD(),
    'viscoelastic': GSM1D_VE()
}

# Collection of complete models
models = {}
for name, gsm_def in gsm_defs.items():
    models[name] = GSMModel(gsm_def, material_params)
```

## Architectural Flow

The typical workflow follows this pattern:

1. **Define Symbolic Behavior**: Create `GSMSymbDef` instances (`gsm_def`)
2. **Set Parameters**: Create `MaterialParams` instances (`params`)
3. **Build Complete Model**: Combine into `GSMModel` instances (`model`)
4. **Execute Simulations**: Use model to calculate material response

```python
# Step 1: Define symbolic behavior
ed_gsm_def = GSM1D_ED()

# Step 2: Set material parameters
params = MaterialParams(E=30000, gamma_T=100, K=10, S=0.005)

# Step 3: Build complete model
model = GSMModel(ed_gsm_def, params)

# Step 4: Execute simulation
strain_history = np.linspace(0, 0.01, 100)
response = model.get_response(strain_history)
```

## Benefits of This Architecture

1. **Separation of Concerns**: Clear distinction between mathematical definitions and computational implementation
2. **Reusability**: Symbolic definitions can be reused with different parameters and engines
3. **Testability**: Each component can be tested independently
4. **Maintainability**: Changes to symbolic definitions don't require changes to numerical engines
5. **Extensibility**: New material models can be added by extending symbolic definitions

## Validation and Consistency

Each component includes validation methods:
- **GSMSymbDef**: `validate_symbolic_expressions()` for mathematical consistency
- **MaterialParams**: Parameter bounds and compatibility checking
- **GSMModel**: Overall model validation combining symbolic and parameter validation

This naming convention and architecture ensure clear communication about the role and capabilities of each component in the material modeling framework.
