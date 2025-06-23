# Generalized Standard Material (GSM) Lagrange Framework

This folder implements a symbolic-to-executable framework for thermodynamically consistent material modeling, based on the Generalized Standard Material (GSM) concept. The design enables clear separation between symbolic model definitions, parameter records, executable models, and real-world material collections.

## Key Concepts

### 1. Symbolic Model Definition (`GSMDef`)
- Abstract base class for all GSM models.
- Encapsulates symbolic variables, thermodynamic potentials, constraints, and derivation logic.
- Subclasses (e.g., `GSM1D_ED`, `GSM1D_EP`, etc.) define specific material models by specifying symbolic expressions for free energy, state variables, and constraints.
- Provides both Helmholtz and Gibbs free energy formulations and their associated methods.

### 2. Symbolic Engine (`GSMEngine`)
- Handles symbolic manipulation and derivation of evolution equations, constraints, and potentials.
- Supports automatic differentiation, lambdification, and time-stepping algorithms.

### 3. Executable Model (`GSMModel`)
- Bridges a symbolic model definition (`GSMDef`) with a set of parameter values.
- Dynamically creates traits for all model parameters, enabling instance-level configuration.
- Provides methods for numerical simulation (e.g., `get_F_response`, `get_G_response`).

### 4. Material Parameter Record (`MaterialParams`)
- Stores a set of parameter values for a specific symbolic model (`GSMDef`).
- Enables parameter management, calibration, and database integration.

### 5. Real-World Material (`Material`)
- Represents a physical material (e.g., a concrete mixture or steel grade).
- Aggregates multiple `MaterialParams` records, each corresponding to a different GSM model definition.
- Supports retrieval and management of parameter sets for different models.

## Workflow

1. **Define a Symbolic Model:**  
   Subclass `GSMDef` and specify symbolic variables, potentials, and constraints.

2. **Create Parameter Records:**  
   Use `MaterialParams` to store parameter values for a given model.

3. **Build Executable Models:**  
   Instantiate `GSMModel` with a symbolic model and parameter values for simulation.

4. **Aggregate Materials:**  
   Use `Material` to collect parameter records for real-world materials.

5. **Simulate and Analyze:**  
   Use the executable model's methods to run simulations and analyze responses under various loading scenarios.

## File Structure

- `gsm_def.py`         : Symbolic model base class (`GSMDef`)
- `gsm_mpdp.py`        : Symbolic engine for potentials and constraints (`GSMEngine`)
- `gsm_model.py`       : Executable model class (`GSMModel`)
- `material_params.py` : Material parameter record class (`MaterialParams`)
- `material.py`        : Real-world material collection class (`Material`)
- `gsm1d_*.py`         : Specific 1D GSM model definitions (e.g., elastic-damage, viscoelastic, etc.)
- `gsm_lagrange.puml`  : UML class diagram of the framework

## Design Highlights

- **Symbolic-to-Executable Pipeline:**  
  Symbolic models are defined once and automatically converted to efficient numerical code.

- **Parameter Mapping:**  
  Parameter codenames are derived automatically, requiring explicit mapping only for symbols with non-standard names.

- **Extensibility:**  
  New models are added by subclassing `GSMDef` and specifying symbolic expressions.

- **Separation of Concerns:**  
  Symbolic definitions, parameter records, executable models, and material collections are clearly separated for maintainability and clarity.

- **Consistent Naming:**  
  All classes and files follow a systematic naming convention to emphasize the GSM concept and maintain clarity.

## Example Usage

```python
from bmcs_matmod.gsm_lagrange.gsm1d_ed import GSM1D_ED
from bmcs_matmod.gsm_lagrange.gsm_model import GSMModel

# Create an executable model with parameters
model = GSMModel(gsm_def=GSM1D_ED, E=20000, omega_0=0.01, omega_1=0.2, kappa_0=0.0)

# Simulate a monotonic tension test
import numpy as np
strain = np.linspace(0, 0.006, 100)
time = np.linspace(0, 1.0, 100)
response = model.get_F_response(strain, time)
```

## UML Diagram

See `gsm_lagrange.puml` for a class diagram of the framework.

---

This structure supports robust, extensible, and maintainable development of advanced material models for computational mechanics.
