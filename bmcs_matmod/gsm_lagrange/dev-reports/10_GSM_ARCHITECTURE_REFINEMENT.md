# 10. GSM Framework Architecture Refinement

**Date:** August 23, 2025  
**Status:** Proposed Architecture Improvements  
**Priority:** High - Core Framework Design

## Current Architecture Issues

### 1. Dual Engine Embedding Problem

The current `GSMDef` class embeds both Helmholtz (`F_engine`) and Gibbs (`G_engine`) engines directly as class attributes. This creates several issues:

- **Redundant Definitions**: Both engines are maintained separately despite being thermodynamically related
- **Inconsistent Primary Definition**: Unclear which formulation (Helmholtz vs Gibbs) is the primary source
- **Tight Coupling**: GSMDef is tightly coupled to GSMEngine implementation details
- **Computational Overhead**: Both engines are always computed, even when only one is needed

### 2. Unclear Separation of Concerns

Current responsibilities are mixed:
- `GSMDef`: Should define thermodynamic relationships (symbolic)
- `GSMEngine`: Should provide time-stepping execution (numerical)
- Current: GSMDef contains execution engines instead of pure definitions

### 3. Material Parameter Management Confusion

The relationship between `GSMDef`, `GSMEngine`, and `GSMMaterial` is unclear:
- GSMDef defines available parameters
- GSMEngine embeds parameter usage
- GSMMaterial should manage concrete parameter values

## Proposed Architecture Refinement

### 1. Pure GSMDef as Thermodynamic Definition

**GSMDef** should become a pure symbolic definition containing:
- Primary thermodynamic potential (Helmholtz free energy)
- Variable definitions (strain, stress, internal variables)
- Material parameter specifications
- Constraint definitions
- **No embedded engines**

```python
class GSMDef:
    """Pure thermodynamic definition - symbolic only"""
    
    # Primary variables (always Helmholtz-based)
    eps_vars: Tuple[sp.Symbol, ...]     # External strain variables
    Eps_vars: Tuple[sp.Symbol, ...]     # Internal strain variables
    m_params: Tuple[sp.Symbol, ...]     # Material parameters
    
    # Primary potential (Helmholtz free energy)
    F_expr: sp.Expr                     # Free energy expression
    f_expr: sp.Expr                     # Dissipation potential
    phi_ext_expr: sp.Expr               # External potential
    h_k: List[sp.Expr]                  # Constraints
    
    # Codename mappings for UI/display
    eps_codenames: Dict[sp.Symbol, str]
    param_codenames: Dict[sp.Symbol, str]
    # ... other codename mappings
    
    # Methods
    def legendre_transform(self) -> 'GSMDef':
        """Return Gibbs-transformed GSMDef"""
        pass
        
    def create_engine(self) -> 'GSMEngine':
        """Create execution engine from this definition"""
        pass
```

### 2. Legendre Transform as GSMDef Method

**Core Principle**: Helmholtz formulation is always the primary definition.

```python
def legendre_transform(self) -> 'GSMDef':
    """
    Transform Helmholtz-based GSMDef to Gibbs-based GSMDef.
    
    Process:
    1. Calculate strain substitutions: ε = ∂F/∂σ
    2. Compute Gibbs free energy: G = σ·ε - F
    3. Substitute strains in dissipative terms
    4. Return new GSMDef with transformed expressions
    
    Returns:
        GSMDef: New definition with Gibbs formulation
    """
    # Calculate stress-strain relationships
    sig_a = self.get_stress_variables()
    eps_subs = self._calculate_strain_substitutions()
    
    # Transform free energy
    G_expr = self._compute_gibbs_energy(eps_subs)
    
    # Transform dissipative terms
    f_gibbs = self.f_expr.subs(eps_subs)
    phi_gibbs = self.phi_ext_expr.subs(eps_subs)
    h_gibbs = [h.subs(eps_subs) for h in self.h_k]
    
    # Create new GSMDef with transformed expressions
    return GSMDef(
        eps_vars=sig_a,           # Stress becomes primary variable
        sig_vars=eps_subs,        # Strain expressions become derived
        F_expr=G_expr,
        f_expr=f_gibbs,
        phi_ext_expr=phi_gibbs,
        h_k=h_gibbs,
        # ... other transformed attributes
    )
```

### 3. Separated GSMEngine Creation

**GSMEngine** becomes a pure execution engine created from GSMDef:

```python
class GSMEngine:
    """Pure execution engine for time-stepping"""
    
    def __init__(self, gsm_def: GSMDef):
        """Create engine from GSM definition"""
        self.gsm_def = gsm_def
        self._lambdify_expressions()
        
    def get_sig(self, eps, *args) -> np.ndarray:
        """Calculate stress response"""
        pass
        
    def get_response(self, eps_ta, t_t, *args) -> Tuple:
        """Time-stepping solution"""
        pass
```

### 4. Material Parameter Management

The material parameter handling involves three distinct concepts:

#### 4.1 MaterialParams (Database Record)

**MaterialParams** represents a database record storing material parameter values:

```python
class MaterialParams:
    """Database record for material parameter values"""
    
    def __init__(self, material_id: str):
        self.material_id = material_id
        self.parameters = {}  # Dynamic parameter storage
        self.calibrated_models = set()  # GSMDef classes used with this material
        
    def add_parameters(self, gsm_def_class: Type[GSMDef], **param_values):
        """Add parameters for a specific GSM model"""
        # Validate parameters against GSMDef requirements
        required_params = gsm_def_class.get_required_parameters()
        self._validate_parameters(required_params, param_values)
        
        # Store parameters and track model usage
        self.parameters.update(param_values)
        self.calibrated_models.add(gsm_def_class.__name__)
        
    def get_parameters_for_model(self, gsm_def_class: Type[GSMDef]) -> Dict[str, float]:
        """Extract parameters relevant for a specific GSM model"""
        required_params = gsm_def_class.get_required_parameters()
        return {param: self.parameters[param] 
                for param in required_params if param in self.parameters}
                
    def has_complete_parameters(self, gsm_def_class: Type[GSMDef]) -> bool:
        """Check if all required parameters are available for a GSM model"""
        required_params = set(gsm_def_class.get_required_parameters())
        available_params = set(self.parameters.keys())
        return required_params.issubset(available_params)
```

**Key Features of MaterialParams:**
- **Dynamic Parameter Union**: Accumulates parameters from all GSM models ever used
- **Model Tracking**: Records which GSMDef classes were calibrated with this material
- **Parameter Validation**: Ensures compatibility with specific GSM model requirements
- **Database Persistence**: Designed for storage and retrieval from material database

#### 4.2 GSMModel (Executable Material Model)

**GSMModel** combines a GSMDef with MaterialParams to create an executable model:

```python
class GSMModel:
    """Executable material model combining GSMDef and material parameters"""
    
    def __init__(self, gsm_def_class: Type[GSMDef], material_params: MaterialParams):
        self.gsm_def_class = gsm_def_class
        self.material_params = material_params
        
        # Validate parameter completeness
        if not material_params.has_complete_parameters(gsm_def_class):
            missing = self._get_missing_parameters()
            raise ValueError(f"Missing parameters for {gsm_def_class.__name__}: {missing}")
            
        # Extract relevant parameters
        self.param_values = material_params.get_parameters_for_model(gsm_def_class)
        
    def create_helmholtz_engine(self) -> GSMEngine:
        """Create Helmholtz-based engine with concrete parameters"""
        gsm_def = self.gsm_def_class()
        engine = gsm_def.create_engine()
        engine.set_parameters(**self.param_values)
        return engine
        
    def create_gibbs_engine(self) -> GSMEngine:
        """Create Gibbs-based engine with concrete parameters"""
        gsm_def = self.gsm_def_class()
        gibbs_def = gsm_def.legendre_transform()
        engine = gibbs_def.create_engine()
        engine.set_parameters(**self.param_values)
        return engine
        
    def get_parameter_value(self, param_name: str) -> float:
        """Get specific parameter value"""
        return self.param_values[param_name]
        
    def update_parameter(self, param_name: str, value: float):
        """Update parameter value (affects both local and database record)"""
        self.param_values[param_name] = value
        self.material_params.parameters[param_name] = value
```

## Implementation Benefits

### 1. Clear Separation of Concerns
- **GSMDef**: Pure symbolic thermodynamic definition
- **GSMEngine**: Numerical time-stepping execution
- **MaterialParams**: Database record for parameter storage and management
- **GSMModel**: Executable model combining definition and parameters

### 2. Computational Efficiency
- Only create engines when needed (Helmholtz OR Gibbs)
- Lazy evaluation of Legendre transforms
- No redundant engine storage
- Parameter validation at model creation time

### 3. Enhanced Flexibility
- Easy switching between Helmholtz and Gibbs formulations
- Clear extension path for other thermodynamic potentials
- Simplified testing and validation
- Dynamic parameter accumulation across multiple GSM models

### 4. Better Database Integration
- MaterialParams designed for persistence
- Parameter union across multiple model types
- Tracking of model-material relationships
- Support for manual parameter addition and calibration workflows

### 5. Better Code Organization
- Symbolic operations in GSMDef
- Numerical operations in GSMEngine
- Parameter management in MaterialParams
- Model instantiation in GSMModel

## Usage Scenarios

### 1. Material Database Workflow
```python
# Create or load material parameter record
steel_params = MaterialParams("Steel_S355")

# Add parameters from experimental characterization
steel_params.add_parameters(GSM1D_EP, E=210e3, K=1000, f_c=355)
steel_params.add_parameters(GSM1D_VE, E=210e3, eta=1e6, tau=3600)

# Later, create specific model instances
ep_model = GSMModel(GSM1D_EP, steel_params)
ve_model = GSMModel(GSM1D_VE, steel_params)  # Reuses E parameter
```

### 2. Finite Element Simulation (Strain Control)
```python
# Load material from database
steel_params = load_material_params("Steel_S355")
model = GSMModel(GSM1D_EP, steel_params)

# Use Helmholtz formulation (strain-controlled)
engine = model.create_helmholtz_engine()
sig_response = engine.get_response(eps_history, time_steps)
```

### 3. Fatigue Analysis (Stress Control)
```python
# Same material, different formulation
steel_params = load_material_params("Steel_S355")
model = GSMModel(GSM1D_EP, steel_params)

# Use Gibbs formulation (stress-controlled)
engine = model.create_gibbs_engine()
eps_response = engine.get_response(sig_history, time_steps)
```

### 4. Model Development and Parameter Sensitivity
```python
# Create material with baseline parameters
baseline_params = MaterialParams("Development_Material")
baseline_params.add_parameters(GSM1D_EP, E=210e3, K=1000, f_c=20)

# Study parameter sensitivity
for f_c_value in [15, 20, 25, 30]:
    # Create model variant
    variant_params = baseline_params.copy()
    variant_params.update_parameter('f_c', f_c_value)
    
    model = GSMModel(GSM1D_EP, variant_params)
    engine = model.create_helmholtz_engine()
    
    # Run analysis
    response = engine.get_response(load_history, time_steps)
    analyze_sensitivity(f_c_value, response)
```

### 5. Multi-Model Material Characterization
```python
# Progressive material characterization
concrete_params = MaterialParams("Concrete_C30")

# Start with elastic characterization
concrete_params.add_parameters(LinearElastic, E=30e3, nu=0.2)

# Add damage parameters after damage tests
concrete_params.add_parameters(GSM1D_ED, E=30e3, omega_0=0.1, r=2.0)

# Add rate-dependent parameters after dynamic tests
concrete_params.add_parameters(GSM1D_VE, eta=1e8, tau=100)

# Now concrete_params contains union of all parameters:
# {E: 30e3, nu: 0.2, omega_0: 0.1, r: 2.0, eta: 1e8, tau: 100}

# Can create any compatible model
ed_model = GSMModel(GSM1D_ED, concrete_params)  # Uses E, omega_0, r
ve_model = GSMModel(GSM1D_VE, concrete_params)  # Uses E, eta, tau
```

## Migration Strategy

### Phase 1: GSMDef Purification

1. Remove embedded F_engine and G_engine from GSMDef
2. Move engine creation to separate factory methods
3. Implement legendre_transform() method
4. Add get_required_parameters() class method to GSMDef

### Phase 2: Engine Separation

1. Refactor GSMEngine to accept GSMDef in constructor
2. Remove direct symbolic attribute access from engines
3. Implement lazy engine creation
4. Add parameter injection methods to GSMEngine

### Phase 3: Parameter Management System

1. Create MaterialParams class for database records
2. Implement parameter validation and union logic
3. Add model tracking and calibration history
4. Design persistence layer for material database

### Phase 4: Model Integration

1. Create GSMModel class combining GSMDef and MaterialParams
2. Implement engine factory methods with parameter injection
3. Add parameter sensitivity and updating capabilities
4. Update existing model definitions to use new architecture

### Phase 5: Database Integration

1. Design material parameter database schema
2. Implement load/save operations for MaterialParams
3. Create migration tools for existing parameter data
4. Add material management utilities and CLI tools

## Expected Outcomes

1. **Cleaner Architecture**: Clear separation between definition and execution
2. **Better Performance**: Only compute required engines
3. **Enhanced Maintainability**: Simpler testing and debugging
4. **Improved Extensibility**: Easy addition of new thermodynamic formulations
5. **Better Documentation**: Clear conceptual boundaries
6. **Database Integration**: Seamless material parameter management
7. **Multi-Model Support**: Natural parameter sharing across GSM models

## Next Steps - Detailed Implementation Plan

### Immediate Actions (Week 1-2)

#### 1. MaterialParams Class Implementation

```python
# File: bmcs_matmod/gsm_lagrange/core/material_params.py
class MaterialParams:
    """Database record for material parameter values with model tracking"""
    
    def __init__(self, material_id: str, description: str = ""):
        self.material_id = material_id
        self.description = description
        self.parameters = {}  # {param_name: value}
        self.calibrated_models = {}  # {model_name: calibration_info}
        self.metadata = {
            'created_date': datetime.now(),
            'last_modified': datetime.now(),
            'version': '1.0'
        }
    
    def add_parameters_from_calibration(self, gsm_def_class, calibration_data, **param_values):
        """Add parameters from experimental calibration"""
        pass
        
    def add_parameters_manual(self, **param_values):
        """Manually add parameters (for theoretical studies)"""
        pass
        
    def validate_for_model(self, gsm_def_class) -> ValidationResult:
        """Check parameter completeness and consistency"""
        pass
        
    def export_to_dict(self) -> Dict:
        """Export for database storage"""
        pass
        
    @classmethod
    def import_from_dict(cls, data: Dict) -> 'MaterialParams':
        """Import from database"""
        pass
```

#### 2. GSMModel Class Implementation

```python
# File: bmcs_matmod/gsm_lagrange/core/gsm_model.py
class GSMModel:
    """Executable material model combining GSMDef and MaterialParams"""
    
    def __init__(self, gsm_def_class: Type[GSMDef], material_params: MaterialParams):
        self.gsm_def_class = gsm_def_class
        self.material_params = material_params
        self._validate_compatibility()
        self._extract_parameters()
        
    def create_helmholtz_engine(self) -> GSMEngine:
        """Create strain-controlled engine"""
        pass
        
    def create_gibbs_engine(self) -> GSMEngine:
        """Create stress-controlled engine"""
        pass
        
    def get_parameter_info(self) -> Dict:
        """Get detailed parameter information"""
        pass
        
    def update_parameters(self, **new_values):
        """Update parameters with validation"""
        pass
```

#### 3. GSMDef Refactoring - Add Required Methods

```python
# Add to existing GSMDef class
@classmethod
def get_required_parameters(cls) -> List[str]:
    """Return list of required parameter names (codenames)"""
    if not hasattr(cls, 'F_engine') or cls.F_engine is None:
        return []
    return [cls.param_codenames.get(param, param.name) 
            for param in cls.F_engine.m_params]

@classmethod  
def get_parameter_descriptions(cls) -> Dict[str, str]:
    """Return parameter descriptions for UI/documentation"""
    pass
    
@classmethod
def validate_parameter_values(cls, **param_values) -> ValidationResult:
    """Validate parameter values against physical constraints"""
    pass

def legendre_transform(self) -> 'GSMDef':
    """Transform to Gibbs formulation - PRIORITY IMPLEMENTATION"""
    pass
```

### Short-term Goals (Week 3-4)

#### 4. Database Schema Design

```sql
-- Material parameter storage schema
CREATE TABLE material_params (
    id SERIAL PRIMARY KEY,
    material_id VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    parameters JSONB, -- {param_name: value}
    calibrated_models JSONB, -- {model_name: calibration_info}
    metadata JSONB,
    created_date TIMESTAMP,
    last_modified TIMESTAMP
);

CREATE INDEX idx_material_id ON material_params(material_id);
CREATE INDEX idx_parameters ON material_params USING GIN(parameters);
CREATE INDEX idx_models ON material_params USING GIN(calibrated_models);
```

#### 5. Widget Integration Update

Update GSMDefWidget to work with new architecture:

```python
# Update gsm_def_widget.py to handle MaterialParams display
class GSMDefWidget:
    def __init__(self, gsm_model: GSMModel):  # Changed from gsm_def_class
        self.gsm_model = gsm_model
        self.gsm_def_class = gsm_model.gsm_def_class
        self.material_params = gsm_model.material_params
        # ... rest of implementation
        
    def _add_material_parameters_section(self):
        """Add section showing actual parameter values from MaterialParams"""
        pass
        
    def _add_model_compatibility_info(self):
        """Show which other models are compatible with this material"""
        pass
```

### Medium-term Goals (Month 2)

#### 6. Migration Tools

```python
# File: bmcs_matmod/gsm_lagrange/migration/convert_existing_models.py
def migrate_embedded_engines_to_pure_defs():
    """Convert existing GSMDef classes to pure definitions"""
    pass
    
def extract_parameter_defaults():
    """Extract default parameters from existing model definitions"""
    pass
    
def create_material_params_from_examples():
    """Create MaterialParams records from example usages"""
    pass
```

#### 7. CLI Tools for Material Management

```bash
# Command-line interface for material management
bmcs-material create "Steel_S355" --description "Structural steel"
bmcs-material add-params "Steel_S355" --model GSM1D_EP E=210000 K=1000 f_c=355
bmcs-material list-compatible "Steel_S355"  # Show compatible models
bmcs-material export "Steel_S355" --format json
bmcs-material validate "Steel_S355" --model GSM1D_EP
```

### Long-term Goals (Month 3-6)

#### 8. Advanced Features

- **Parameter Uncertainty**: Monte Carlo analysis with parameter distributions
- **Model Comparison**: Automated comparison of different GSM models with same material
- **Calibration Workflows**: Integrated experimental data fitting
- **Version Control**: Parameter evolution tracking and rollback
- **Performance Optimization**: Caching and lazy evaluation

#### 9. Integration with Existing Systems

- **AiiDA Integration**: Store MaterialParams as AiiDA data nodes
- **FEniCS Integration**: Direct material model injection
- **Documentation Generation**: Automatic model documentation from definitions

## Priority Implementation Order

1. **Week 1**: MaterialParams class (core functionality)
2. **Week 1-2**: GSMModel class (basic engine creation)
3. **Week 2**: GSMDef.legendre_transform() method (critical for architecture)
4. **Week 3**: Widget updates to use new architecture
5. **Week 3-4**: Database schema and persistence
6. **Week 4**: Migration tools for existing code
7. **Month 2**: CLI tools and advanced features
8. **Month 3+**: Integration and optimization

## Success Metrics

- [ ] MaterialParams can store and validate parameters for multiple GSM models
- [ ] GSMModel can create both Helmholtz and Gibbs engines correctly
- [ ] Legendre transform produces thermodynamically consistent results
- [ ] Widget displays both symbolic definitions and concrete parameter values
- [ ] Database operations (save/load MaterialParams) work reliably
- [ ] Existing model definitions migrate successfully
- [ ] Performance is equal or better than current embedded engine approach

---

**Note**: This architecture refinement maintains backward compatibility during transition while providing a clear path toward a more maintainable and theoretically sound framework.
