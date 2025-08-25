# 11. Phase 1: GSMDef Purification - Development Plan

**Date:** August 24, 2025  
**Status:** Planning Phase  
**Priority:** High - Core Architecture Implementation  
**Phase:** 1 of 5 (GSMDef Purification)

## Overview

This document outlines the detailed plan for Phase 1 of the GSM architecture refinement, focusing on creating a pure GSMDef implementation that separates symbolic thermodynamic definitions from execution engines.

## Development Strategy

### Incremental Core Module Replacement

The approach involves creating a new `gsm_lagrange/core2` package with clean implementations of core classes, developed and tested incrementally:

1. **GSMDef** - Pure symbolic thermodynamic definition
2. **GSMEngine** - Numerical execution engine (accepting GSMDef)
3. **MaterialDBRecord** - Database record for parameter management
4. **GSMModel** - Integration layer combining definition and parameters

### Development Location

- **Implementation**: `/bmcs_matmod/gsm_lagrange/core2/`
- **Documentation**: `/bmcs_matmod/gsm_lagrange/dev-reports/`
- **Testing**: Jupyter notebooks co-located with implementations

## Phase 1, Step 1: Pure GSMDef Implementation

### Objective

Create a pure symbolic GSMDef class that:
- Contains only thermodynamic definitions (no embedded engines)
- Provides symbolic expressions for potentials and constraints
- Supports specialization to concrete models (e.g., GSM1D_ED)
- Includes basic rendering capabilities for symbolic expressions

### Target File Structure

```
gsm_lagrange/core2/
├── __init__.py
├── gsm_def.py          # Pure GSMDef implementation
└── gsm_def.ipynb       # Testing and validation notebook
```

### GSMDef Class Design (Pure Symbolic)

#### Core Responsibilities
- **Symbolic Variable Definitions**: eps_vars, Eps_vars, m_params, etc.
- **Thermodynamic Potentials**: F_expr (state potential), f_expr (inequality constraint)
- **Constraint Definitions**: Equality constraints (h_k)
- **Codename Mappings**: User-friendly parameter names
- **Expression Validation**: Symbolic consistency checks

#### Key Attributes (Symbolic Only)
```python
class GSMDef:
    """Pure symbolic thermodynamic definition"""
    
    # Primary variables (always Helmholtz-based)
    eps_vars: Tuple[sp.Symbol, ...]      # External strain variables
    Eps_vars: Tuple[sp.Symbol, ...]      # Internal strain variables  
    m_params: Tuple[sp.Symbol, ...]      # Material parameters
    
    # Thermodynamic expressions
    F_expr: sp.Expr                      # Free energy (state potential)
    f_expr: sp.Expr                      # Dissipation potential (inequality constraint)
    phi_ext_expr: sp.Expr                # External potential
    h_k: List[sp.Expr]                   # Equality constraints
    
    # Variable properties
    Sig_signs: Tuple[int, ...]           # Signs for internal stress variables
    
    # Display mappings
    eps_codenames: Dict[sp.Symbol, str]  # Strain variable codenames
    Eps_codenames: Dict[sp.Symbol, str]  # Internal strain codenames
    param_codenames: Dict[sp.Symbol, str] # Parameter codenames
```

#### Removed Elements (Engine-Related)
- ❌ `F_engine` - No embedded engines
- ❌ `G_engine` - No embedded engines  
- ❌ `subs_eps_sig` - Computed expressions (move to engine)
- ❌ `subs_dot_eps_sig` - Computed expressions (move to engine)
- ❌ Numerical methods (`get_sig`, `get_response`, etc.)

#### New Methods to Implement
```python
@classmethod
def get_required_parameters(cls) -> List[str]:
    """Return list of required parameter codenames"""
    
@classmethod  
def get_parameter_descriptions(cls) -> Dict[str, str]:
    """Return parameter descriptions for documentation"""
    
@classmethod
def validate_symbolic_expressions(cls) -> ValidationResult:
    """Validate consistency of symbolic expressions"""
    
def render_potentials(self) -> str:
    """Render thermodynamic potentials in LaTeX format"""
    
def get_expression_summary(self) -> Dict[str, Any]:
    """Get summary of all symbolic expressions"""
```

### Test Model: GSM1D_ED Specialization

#### Target Model Definition
Create a specialization of pure GSMDef for elastic-damage model:

```python
class GSM1D_ED(GSMDef):
    """One-dimensional elastic-damage GSM definition"""
    
    # Material parameters (symbolic)
    E = sp.Symbol('E', real=True, positive=True)      # Young's modulus
    omega_0 = sp.Symbol('omega_0', real=True, positive=True)  # Damage threshold
    r = sp.Symbol('r', real=True, positive=True)      # Damage evolution rate
    
    # External variables
    eps = sp.Symbol('epsilon', real=True)             # Strain
    
    # Internal variables  
    omega = sp.Symbol('omega', real=True, nonnegative=True)  # Damage
    
    # Variable collections
    eps_vars = (eps,)
    Eps_vars = (omega,)
    m_params = (E, omega_0, r)
    
    # State potential (Helmholtz free energy)
    F_expr = sp.Rational(1,2) * E * (1 - omega) * eps**2
    
    # Dissipation potential (damage evolution constraint)
    f_expr = sp.sqrt((E * eps**2 / 2)**2) - (omega_0 + r * omega)
    
    # External potential (typically zero for this model)
    phi_ext_expr = sp.S.Zero
    
    # No equality constraints for this simple model
    h_k = []
    
    # Internal stress signs
    Sig_signs = (1,)  # Positive for damage driving force
    
    # Codename mappings
    eps_codenames = {eps: 'strain'}
    Eps_codenames = {omega: 'damage'}
    param_codenames = {
        E: 'youngs_modulus',
        omega_0: 'damage_threshold', 
        r: 'damage_rate'
    }
```

### Testing Notebook: gsm_def.ipynb

#### Notebook Structure and Content

**Cell 1: Introduction and Setup**
```markdown
# GSMDef Pure Implementation Testing

This notebook tests the pure GSMDef implementation that contains only symbolic 
thermodynamic definitions without embedded engines.

## Testing Objectives
- Validate GSM1D_ED model definition
- Render governing potentials
- Verify symbolic expression consistency
- Test codename mappings and parameter extraction
```

**Cell 2: Import and Setup**
```python
import sympy as sp
import sys
import os
from pathlib import Path

# Add core2 to path for testing
sys.path.insert(0, str(Path.cwd()))

from gsm_def import GSMDef, GSM1D_ED
from IPython.display import display, Math, Markdown

# Enable pretty printing
sp.init_printing()
```

**Cell 3: Model Instantiation**
```python
# Create GSM1D_ED instance
ed_model = GSM1D_ED()

print("✅ GSM1D_ED model created successfully")
print(f"📊 Model name: {ed_model.__class__.__name__}")
print(f"🔢 External variables: {len(ed_model.eps_vars)}")
print(f"🔢 Internal variables: {len(ed_model.Eps_vars)}")
print(f"🔢 Material parameters: {len(ed_model.m_params)}")
```

**Cell 4: Parameter Information**
```python
# Test parameter extraction methods
print("🎯 Required Parameters:")
required_params = ed_model.get_required_parameters()
for param in required_params:
    print(f"  - {param}")

print("\n📝 Parameter Descriptions:")
param_descriptions = ed_model.get_parameter_descriptions()
for param, desc in param_descriptions.items():
    print(f"  - {param}: {desc}")
```

**Cell 5: Symbolic Expression Display**
```python
# Display governing potentials
print("🧮 Governing Potentials:")
print("=" * 50)

print("\n📈 State Potential (Helmholtz Free Energy):")
display(Math(r'F = ' + sp.latex(ed_model.F_expr)))

print("\n⚠️ Inequality Constraint (Dissipation Potential):")
display(Math(r'f = ' + sp.latex(ed_model.f_expr) + r' \leq 0'))

print("\n🌐 External Potential:")
display(Math(r'\phi_{ext} = ' + sp.latex(ed_model.phi_ext_expr)))

if ed_model.h_k:
    print("\n🔗 Equality Constraints:")
    for i, constraint in enumerate(ed_model.h_k):
        display(Math(r'h_{' + str(i) + r'} = ' + sp.latex(constraint) + r' = 0'))
else:
    print("\n🔗 Equality Constraints: None")
```

**Cell 6: Expression Analysis**
```python
# Analyze expressions
print("🔍 Expression Analysis:")
print("=" * 30)

# Check expression dependencies
F_symbols = ed_model.F_expr.free_symbols
f_symbols = ed_model.f_expr.free_symbols

print(f"\n📈 F_expr symbols: {F_symbols}")
print(f"⚠️ f_expr symbols: {f_symbols}")

# Verify all symbols are defined
all_defined_symbols = set(ed_model.eps_vars + ed_model.Eps_vars + ed_model.m_params)
undefined_in_F = F_symbols - all_defined_symbols
undefined_in_f = f_symbols - all_defined_symbols

if undefined_in_F:
    print(f"❌ Undefined symbols in F_expr: {undefined_in_F}")
else:
    print("✅ All symbols in F_expr are properly defined")
    
if undefined_in_f:
    print(f"❌ Undefined symbols in f_expr: {undefined_in_f}")
else:
    print("✅ All symbols in f_expr are properly defined")
```

**Cell 7: Codename Mapping Validation**
```python
# Test codename mappings
print("🏷️ Codename Mapping Validation:")
print("=" * 35)

def test_codename_mapping(symbol_dict, codename_dict, category_name):
    print(f"\n{category_name}:")
    for symbol in symbol_dict:
        codename = codename_dict.get(symbol, "❌ MISSING")
        print(f"  {symbol} → {codename}")
        
test_codename_mapping(ed_model.eps_vars, ed_model.eps_codenames, "🔵 Strain Variables")
test_codename_mapping(ed_model.Eps_vars, ed_model.Eps_codenames, "🟢 Internal Variables") 
test_codename_mapping(ed_model.m_params, ed_model.param_codenames, "⚙️ Material Parameters")
```

**Cell 8: Symbolic Differentiation Tests**
```python
# Test symbolic differentiation (preview of engine operations)
print("∂ Symbolic Differentiation Tests:")
print("=" * 35)

eps, omega = ed_model.eps_vars[0], ed_model.Eps_vars[0]

# Stress from free energy
sigma = sp.diff(ed_model.F_expr, eps)
print("🔵 Stress (∂F/∂ε):")
display(Math(r'\sigma = \frac{\partial F}{\partial \varepsilon} = ' + sp.latex(sigma)))

# Thermodynamic force from free energy  
Omega = -sp.diff(ed_model.F_expr, omega)
print("\n🟢 Damage driving force (-∂F/∂ω):")
display(Math(r'\Omega = -\frac{\partial F}{\partial \omega} = ' + sp.latex(Omega)))

# Damage evolution check
print("\n⚠️ Damage evolution condition:")
display(Math(r'f = ' + sp.latex(ed_model.f_expr) + r' \leq 0'))
```

**Cell 9: Model Summary and Validation**
```python
# Generate model summary
print("📋 GSM1D_ED Model Summary:")
print("=" * 30)

summary = ed_model.get_expression_summary()
for key, value in summary.items():
    print(f"{key}: {value}")

# Validate model
print("\n✅ Model Validation:")
validation_result = ed_model.validate_symbolic_expressions()
if validation_result.is_valid:
    print("✅ Model is symbolically consistent")
else:
    print("❌ Model validation issues:")
    for issue in validation_result.issues:
        print(f"  - {issue}")
```

### Success Criteria for Step 1

#### Functional Requirements
- [ ] GSMDef class contains only symbolic definitions
- [ ] GSM1D_ED specialization works correctly
- [ ] All thermodynamic potentials render properly in LaTeX
- [ ] Parameter extraction methods work as expected
- [ ] Codename mappings are complete and consistent
- [ ] Symbolic differentiation produces expected results

#### Implementation Quality  
- [ ] No embedded engines in GSMDef
- [ ] Clean separation of symbolic vs numerical operations
- [ ] Comprehensive test coverage in notebook
- [ ] Clear documentation and examples
- [ ] Proper error handling and validation

#### Output Validation
- [ ] Notebook runs without errors
- [ ] All symbolic expressions display correctly
- [ ] Parameter mappings are accurate
- [ ] Model validation passes
- [ ] Expression analysis shows proper symbol definitions

### Next Steps After Step 1

Once GSMDef purification is complete and validated:

1. **Step 2**: Implement GSMEngine to accept GSMDef as input
2. **Step 3**: Implement MaterialDBRecord for parameter management  
3. **Step 4**: Create GSMModel integration layer
4. **Step 5**: Update GSMDefWidget to work with new architecture

### File Dependencies

#### New Files to Create
- `gsm_lagrange/core2/gsm_def.py` - Pure GSMDef implementation
- `gsm_lagrange/core2/gsm_def.ipynb` - Testing notebook
- `gsm_lagrange/core2/__init__.py` - Package initialization

#### Reference Files (Read-Only)
- `gsm_lagrange/core/gsm_def.py` - Current implementation for reference
- `gsm_lagrange/models/gsm1d_ed.py` - Current ED model for comparison

This plan provides a clear, focused approach to implementing the first step of GSMDef purification with comprehensive testing and validation through the co-located Jupyter notebook.
