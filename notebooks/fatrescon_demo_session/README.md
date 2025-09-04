# GSM Framework Demonstration Notebooks

This collection of notebooks provides a comprehensive introduction to the **Generalized Standard Material (GSM) framework** for material modeling. The GSM framework offers a thermodynamically consistent approach to modeling complex material behaviors including viscoelasticity, damage, and plasticity.

## Overview

The notebooks are organized to provide both theoretical background and practical demonstrations of the GSM framework capabilities. They cover material models of increasing complexity and various loading scenarios.

## Notebook Collection

### 1. [GSM-VE: Visco-elastic Model](01_gsm_ve.ipynb)

Introduction to the basic visco-elastic material model using the GSM framework.
Demonstrates linear strain loading, creep behavior, and cyclic loading with relaxation.
Shows both strain-controlled and stress-controlled simulations.

---

### 2. [GSM-VED: Visco-elasto-damage Model](02_gsm_ved.ipynb)

Extended material model combining viscoelasticity with damage mechanics.
Explores damage evolution under monotonic loading and its coupling with viscous effects.
Compares elastic control vs. stress control responses.

---

### 3. [GSM-VEVPD-LIH: Visco-elasto-visco-plastic-damage with Linear Isotropic Hardening](03_gsm_vevpd_lih.ipynb)

Comprehensive material model incorporating viscoelasticity, viscoplasticity, and damage.
Features linear isotropic hardening and demonstrates complex material responses.
Suitable for modeling advanced material behaviors under various loading conditions.

---

### 4. [GSM Framework Derivation and Theory](04_present_gsm_derivation.ipynb)

Theoretical foundation and mathematical derivation of the GSM framework.
Covers thermodynamic principles, variational formulations, and constitutive equations.
Essential for understanding the underlying physics and mathematics of GSM models.

---

### 5. [GSM under Monotonic Ascending Loading](05_present_gsm_monotonic_asc.ipynb)

Detailed analysis of GSM material behavior under monotonic increasing loads.
Demonstrates stress-strain evolution, internal variable development, and energy dissipation.
Provides insights into material response characteristics during progressive loading.

---

### 6. [GSM under Cyclic Strain Loading](06_present_gsm_cyclic_strain.ipynb)

Investigation of GSM material response under cyclic and fatigue loading conditions.
Explores hysteresis behavior, energy dissipation per cycle, and progressive damage accumulation.
Critical for understanding material degradation and lifetime prediction.

---

## Getting Started

**Recommended Order:**
1. Start with the theoretical background (Notebook 4)
2. Explore basic models (Notebooks 1-3) in order of increasing complexity
3. Study specific loading scenarios (Notebooks 5-6)

**Prerequisites:**
- Basic knowledge of continuum mechanics and material modeling
- Familiarity with Python and Jupyter notebooks
- Understanding of thermodynamic principles (helpful but not required)

**Software Requirements:**
- `bmcs_matmod` package installed
- Standard scientific Python libraries (numpy, matplotlib, sympy)

## Additional Resources

- **GSM Framework Documentation**: Comprehensive documentation and API reference
- **BMCS Project**: Visit the [BMCS Group](https://www.bmcs.rw.rwth-aachen.de/) for more research and development updates
- **Citation**: If you use these notebooks or the GSM framework in your research, please cite the relevant publications

## Support

For questions, issues, or contributions:
- GitHub Repository: [bmcs_matmod](https://github.com/bmcs-group/bmcs_matmod)
- Contact: BMCS Group, RWTH Aachen University

---

*Last updated: August 2025*
