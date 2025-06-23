# GSM Definition vs GSM Model - Terminology Clarification

## Key Distinction

### GSM Definitions (GSMDef)
- **What they are**: Classes that define constitutive behavior and symbolic formulations
- **Location**: Files like `gsm1d_ed.py`, `gsm1d_ve.py`, etc.
- **Content**: Mathematical formulations, symbolic expressions, parameter definitions
- **Example**: `GSM1D_ED` class defining elasto-damage behavior
- **Purpose**: Reusable templates for constitutive models

### GSM Models  
- **What they are**: Instantiated GSM Definitions with specific material parameters
- **Creation**: `gsm_def_class(material_parameters)` 
- **Content**: Numerical values, ready for simulation
- **Example**: `GSM1D_ED(E=30000, S=0.005, c=2.8, r=1.2)` 
- **Purpose**: Concrete models for specific loading scenarios

## Registry Role

The **GSM Definition Registry** manages:
- ✅ **GSMDef Classes**: `GSM1D_ED`, `GSM1D_VE`, etc.
- ✅ **Discovery**: Finding definition files and classes
- ✅ **Access**: Providing GSMDef classes for instantiation

The registry does **NOT** manage:
- ❌ **Model Instances**: Specific parameter combinations
- ❌ **Loading Scenarios**: Applied loads, boundary conditions  
- ❌ **Simulation Results**: Stress-strain responses

## Usage Workflow

```python
# 1. Get GSM Definition from registry
from gsm_def_registry_new import get_model_class
GSM_ED_Def = get_model_class('ED')  # This is a GSMDef class

# 2. Create GSM Model with specific parameters  
gsm_model = GSM_ED_Def(
    E=30000,     # Young's modulus
    S=0.005,     # Damage threshold
    c=2.8,       # Damage evolution parameter
    r=1.2        # Damage evolution exponent
)

# 3. Use GSM Model in loading scenario
response = gsm_model.get_response(strain_history)
```

## Registry Functions - Clarified

### CLI-Safe Functions (No GSMDef Import)
```python
# Get available GSM definition names
defs, paths = get_available_models()  # Returns: ['GSM1D_ED', 'GSM1D_VE', ...]

# Check if a GSM definition exists  
exists = check_model_exists('ED')     # Returns: True/False

# Get module path for a GSM definition
path = get_model_module_path('ED')    # Returns: 'bmcs_matmod.gsm_lagrange.gsm1d_ed'
```

### Class-Loading Functions (Imports GSMDef)
```python
# Get actual GSMDef class (for instantiation)
GSM_ED_Def = get_model_class('ED')    # Returns: <class 'GSM1D_ED'>

# Get all GSMDef classes
all_defs = get_gsm_defs()             # Returns: {'ED': <class 'GSM1D_ED'>, ...}

# List GSMDef classes  
def_list = list_gsm_defs()            # Returns: [('GSM1D_ED', <class>), ...]
```

## Refactored Code Benefits

1. **Clear Terminology**: 
   - `LazyGSMDefRegistry` (not `LazyGSMRegistry`)
   - `get_def_names()` (not `get_model_names()`)
   - `get_def_class()` (not `get_class()`)

2. **Conceptual Clarity**:
   - Registry handles **definitions** (templates)
   - Users create **models** (instances) from definitions
   - Clear separation of concerns

3. **Backward Compatibility**:
   - API function names preserved for existing code
   - Function documentation clarifies the distinction
   - No breaking changes to existing workflows

## Example: Creating Models from Definitions

```python
# Registry provides the definition templates
from gsm_def_registry_new import get_model_class

# Get GSM definitions (templates)
ED_Def = get_model_class('ED')
VE_Def = get_model_class('VE') 
EP_Def = get_model_class('EP')

# Create specific GSM models (instances) for different scenarios
concrete_model = ED_Def(E=35000, S=0.004, c=3.0, r=1.5)
steel_model = EP_Def(E=200000, sig_y=400, H=2000)
polymer_model = VE_Def(E_inf=1000, E=5000, tau=100)

# Each model can now be used in specific loading scenarios
concrete_response = concrete_model.get_response(loading_scenario_1)
steel_response = steel_model.get_response(loading_scenario_2)
polymer_response = polymer_model.get_response(loading_scenario_3)
```

This distinction clarifies that the registry is a **template provider**, not a **model manager**.
