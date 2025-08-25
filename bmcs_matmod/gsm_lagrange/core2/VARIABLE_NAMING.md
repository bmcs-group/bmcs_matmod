# GSM Material Model Variable Naming Convention

This document establishes the standardized naming convention for internal variables and their corresponding thermodynamic forces across all GSM material models, based on analysis of the models in `/bmcs_matmod/gsm_lagrange/models/`.

## Overview

The GSM framework uses a systematic coupling between:
- **Internal Variables (Eps_vars)**: State variables that capture internal material processes
- **Thermodynamic Forces (Sig_vars)**: Conjugate forces driving the evolution of internal variables

## Standard Variable Categories

### 1. Plasticity Variables

**Internal Variable**: `eps_p` (plastic strain)
- **Symbol**: `ε^p` or `ε^{vp}` (viscoplasticity)
- **LaTeX**: `\varepsilon^{\mathrm{p}}` or `\varepsilon^{\mathrm{vp}}`
- **Codename**: `eps_p`
- **Physical meaning**: Accumulated plastic deformation

**Conjugate Force**: `sig_p` (plastic stress)
- **Symbol**: `σ^p` or `σ^{vp}`
- **LaTeX**: `\sigma^{\mathrm{p}}` or `\sigma^{\mathrm{vp}}`
- **Codename**: `sig_p`
- **Physical meaning**: Back stress opposing plastic flow
- **Sign convention**: `-1` (opposes flow)

### 2. Damage Variables

**Internal Variable**: `omega` (damage parameter)
- **Symbol**: `ω`
- **LaTeX**: `\omega`
- **Codename**: `omega`
- **Physical meaning**: Material degradation (0=undamaged, 1=fully damaged)

**Conjugate Force**: `Y` (damage driving force)
- **Symbol**: `Y`
- **LaTeX**: `Y`
- **Codename**: `Y`
- **Physical meaning**: Energy release rate driving damage evolution
- **Sign convention**: `-1` (drives damage growth)

### 3. Hardening Variables

**Internal Variable**: `z` (hardening parameter)
- **Symbol**: `z`
- **LaTeX**: `z`
- **Codename**: `z`
- **Physical meaning**: Accumulated hardening measure

**Conjugate Force**: `Z` (hardening force)
- **Symbol**: `Z`
- **LaTeX**: `Z`
- **Codename**: `Z`
- **Physical meaning**: Isotropic hardening stress
- **Sign convention**: `+1` (strengthens material)

### 4. Kinematic Hardening Variables

**Internal Variable**: `alpha` (kinematic hardening)
- **Symbol**: `α`
- **LaTeX**: `\alpha`
- **Codename**: `alpha`
- **Physical meaning**: Kinematic hardening backstress measure

**Conjugate Force**: `X` (kinematic hardening force)
- **Symbol**: `X`
- **LaTeX**: `X`
- **Codename**: `X`
- **Physical meaning**: Kinematic hardening stress
- **Sign convention**: `+1` (contributes to yield surface translation)

### 5. Viscoelastic Variables

**Internal Variable**: `eps_v` (viscous strain)
- **Symbol**: `ε^v` or `ε^{ve}`
- **LaTeX**: `\varepsilon^{\mathrm{v}}` or `\varepsilon^{\mathrm{ve}}`
- **Codename**: `eps_v`
- **Physical meaning**: Viscous deformation component

**Conjugate Force**: `sig_v` (viscous stress)
- **Symbol**: `σ^v` or `σ^{ve}`
- **LaTeX**: `\sigma^{\mathrm{v}}` or `\sigma^{\mathrm{ve}}`
- **Codename**: `sig_v`
- **Physical meaning**: Viscous stress component
- **Sign convention**: `-1` (opposes viscous flow)

## Model-Specific Variable Patterns

### ED (Elastic-Damage)
```python
Eps_vars = (omega_a, z_a)
Sig_vars = (Y_a, Z_a)
Sig_signs = (-1, 1)
```

### EP (Elastic-Plastic)
```python
Eps_vars = (eps_p_a, z_a)
Sig_vars = (sig_p_a, Z_a)
Sig_signs = (-1, 1)
```

### EPD (Elastic-Plastic-Damage)
```python
Eps_vars = (eps_p_a, omega_a, z_a, alpha_a)
Sig_vars = (sig_p_a, Y_a, Z_a, X_a)
Sig_signs = (-1, -1, 1, 1)
```

### VE (Viscoelastic)
```python
Eps_vars = (eps_v_a,)
Sig_vars = (sig_v_a,)
Sig_signs = (-1,)
```

### VEVPD (Viscoelastic-Viscoplastic-Damage)
```python
Eps_vars = (eps_v_a, eps_p_a, omega_a, z_a)
Sig_vars = (sig_v_a, sig_p_a, Y_a, Z_a)
Sig_signs = (-1, -1, -1, 1)
```

## Sign Convention Rules

### Negative Signs (-1)
Applied to variables that **dissipate energy** or **oppose resistance**:
- `eps_p` → `sig_p`: Plastic stress opposes plastic flow
- `eps_v` → `sig_v`: Viscous stress opposes viscous flow  
- `omega` → `Y`: Damage force drives material degradation

### Positive Signs (+1)
Applied to variables that **store energy** or **strengthen material**:
- `z` → `Z`: Hardening force increases yield strength
- `alpha` → `X`: Kinematic hardening contributes to yield surface evolution

## Naming Pattern Rules

### Internal Variables (Eps_vars)
1. **Base symbols**: Use lowercase Greek letters or Latin letters with subscripts
2. **Strain-like**: `eps_p`, `eps_v` (with superscript notation in LaTeX)
3. **Scalar measures**: `omega`, `z`, `alpha`
4. **Vector forms**: Add `_a` suffix (e.g., `eps_p_a`, `omega_a`)

### Thermodynamic Forces (Sig_vars)
1. **Stress-like conjugates**: Use `sig_` prefix with same subscript as internal variable
2. **Energy conjugates**: Use uppercase letters (`Y`, `Z`, `X`)
3. **Vector forms**: Add `_a` suffix matching internal variable

### LaTeX Rendering
1. **Material processes**: Use `\mathrm{}` for subscripts/superscripts (e.g., `\varepsilon^{\mathrm{p}}`)
2. **Indices**: Use regular math mode (e.g., `\sigma_a`)
3. **Greek letters**: Use standard LaTeX commands (`\varepsilon`, `\omega`, `\alpha`)

## Display Codenames

### Alphanumeric (Code Output)
For programmatic output, use alphanumeric representations:
- `eps` → `eps`, `eps_p` → `eps_p`
- `sigma` → `sigma`, `sig_p` → `sig_p`  
- `omega` → `omega`, `Y` → `Y`

### LaTeX (Mathematical Display)
For mathematical rendering, use proper LaTeX:
- `eps` → `\varepsilon`, `eps_p` → `\varepsilon^{\mathrm{p}}`
- `sigma` → `\sigma`, `sig_p` → `\sigma^{\mathrm{p}}`
- `omega` → `\omega`, `Y` → `Y`

## Implementation Requirements

### Model Definition
1. All models must define `Eps_vars` and `Sig_vars` tuples in matching order
2. `Sig_signs` must have same length as `Eps_vars`/`Sig_vars`
3. Variable codenames should follow alphanumeric convention

### Documentation  
1. Include physical meaning for each variable pair
2. Specify sign convention reasoning
3. Reference standard physics literature where applicable

### Code Style
1. Use consistent naming across all model files
2. Group related variables together in model definitions
3. Include comments explaining physical coupling

---

*This specification should be followed for all new model implementations and updates to ensure consistency across the GSM framework.*

# Material Parameter Naming Convention

### Elastic Parameters

**Young's Modulus**: `E`
- **Symbol**: `E`
- **LaTeX**: `E`
- **Codename**: `E`
- **Physical meaning**: Linear elastic stiffness
- **Units**: Stress units (e.g., MPa, Pa)

### Hardening Parameters

**Isotropic Hardening Modulus**: `K`
- **Symbol**: `K`
- **LaTeX**: `K`
- **Codename**: `K`
- **Physical meaning**: Controls isotropic hardening rate
- **Usage**: Appears in free energy as `½Kz²`
- **Units**: Stress units (e.g., MPa, Pa)

**Kinematic Hardening Modulus**: `γ`
- **Symbol**: `γ` (gamma)
- **LaTeX**: `\gamma`
- **Codename**: `gamma`
- **Physical meaning**: Controls kinematic hardening rate
- **Usage**: Appears in free energy as `½γα²`
- **Units**: Stress units (e.g., MPa, Pa)

**Yield Stress**: `f_c`
- **Symbol**: `f_c`
- **LaTeX**: `f_\mathrm{c}`
- **Codename**: `f_c`
- **Physical meaning**: Initial yield strength
- **Usage**: Appears in yield function as `f(σ) = |σ_eff| - (f_c + Z)`
- **Units**: Stress units (e.g., MPa, Pa)

### Damage Evolution Parameters

**Damage Evolution Parameter**: `S`
- **Symbol**: `S`
- **LaTeX**: `S`
- **Codename**: `S`
- **Physical meaning**: Controls damage evolution rate
- **Usage**: Appears in external potential as `φ_ext = (1-ω)^c * (S/(r+1)) * (Y/S)^(r+1)`
- **Units**: Energy density units (e.g., J/m³, N·m/m³)

**Damage Evolution Exponent**: `c`
- **Symbol**: `c`
- **LaTeX**: `c`
- **Codename**: `c`
- **Physical meaning**: Controls damage evolution sensitivity to current damage state
- **Usage**: Appears as exponent in `(1-ω)^c` term
- **Units**: Dimensionless

**Damage Evolution Exponent**: `r`
- **Symbol**: `r`
- **LaTeX**: `r`
- **Codename**: `r`
- **Physical meaning**: Controls damage evolution sensitivity to driving force
- **Usage**: Appears as exponent in `(Y/S)^(r+1)` term
- **Units**: Dimensionless

## Standard Parameter Combinations

### Elastic-Damage (ED)
```python
m_params = (E, S, c, r, eps_0)
```
Where `eps_0` is the damage threshold strain.

### Elastic-Plastic-Damage (EPD)
```python
m_params = (E, K, gamma, S, c, r, f_c)
```

### Viscoelastic (VE)
```python
m_params = (E, eta)
```
Where `eta` (η) is the viscosity parameter.

## External Potential Formulations

### Damage Evolution
The standard form for damage evolution external potential:
```
φ_ext = (1 - ω)^c * (S/(r+1)) * (Y/S)^(r+1)
```

**Physical interpretation**:
- `(1-ω)^c`: Reduces evolution rate as damage increases
- `S`: Sets the energy scale for damage evolution
- `r`: Controls nonlinearity of damage evolution
- Higher `r` values lead to more sudden damage evolution

### Plastic Flow (if non-associative)
For non-associative plasticity, external potential may include:
```
φ_ext = additional terms for plastic flow direction
```

## Parameter Validation Rules

1. **Elastic parameters**: Must be positive (`E > 0`, `K > 0`, `γ > 0`)
2. **Strength parameters**: Must be non-negative (`f_c ≥ 0`)
3. **Damage parameters**: Must be positive (`S > 0`, `c > 0`, `r > 0`)
4. **Exponents**: Typically `r > 0` and `c > 0` for physical consistency
5. **Viscosity**: Must be positive (`η > 0`)

## Usage in Free Energy

### Standard free energy decomposition:
```
F = U_elastic + U_hardening
U_elastic = ½(1-ω)E(ε-ε_p)²     # Elastic energy with damage
U_hardening = ½Kz² + ½γα²        # Isotropic + kinematic hardening
```

### Yield function:
```
f = √((σ_p/(1-ω) - X)²) - (f_c + Z) ≤ 0
```

Where:
- `σ_p/(1-ω)`: Effective plastic stress accounting for damage
- `X`: Kinematic hardening backstress
- `f_c + Z`: Current yield strength (initial + isotropic hardening)
