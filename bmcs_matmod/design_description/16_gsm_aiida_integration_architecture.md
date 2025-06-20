# Architecture for Integrating GSM-Based Models with AiiDA and AiiDA Lab

This document outlines a recommended architecture for integrating GSM-based material models (as implemented in `bmcs_matmod`) with AiiDA and AiiDA Lab, focusing on data node design, workflow orchestration, and interactive visualization. It also addresses best practices for balancing standalone package development with AiiDA integration, and generalizes the approach for other simulation and experimental codes.

---

## 1. Standalone vs. Integration: Dependency Management

### **Key Concern**

- **bmcs_matmod** is a standalone package for GSM-based modeling and should remain usable without AiiDA.
- **Directly importing `aiida.orm` in bmcs_matmod** would make AiiDA a required dependency, complicating installation and use for users who do not need AiiDA integration.

### **Best Practice**

- **Separation of Concerns:**  
  Keep the core package (`bmcs_matmod`) independent of AiiDA and any workflow/data management system.
- **Integration Package:**  
  Create a separate package (e.g., `bmcs_matmod_aiida` or `bmcs_matmod.aiida_integration`) that depends on both `bmcs_matmod` and `aiida-core`. This package implements all AiiDA-specific Data nodes, Calculation plugins, and WorkChains.
- **Optional Integration:**  
  Users who need AiiDA integration install the integration package; others use only the core package.

### **Known Practice**

- This pattern is widely used in the scientific Python ecosystem:
  - **ase** (Atomic Simulation Environment) has `aiida-ase` for AiiDA integration.
  - **pymatgen** has `aiida-pymatgen`.
  - **cp2k**, **vasp**, and other codes have separate AiiDA plugin packages.
- **Advantages:**
  - Keeps core packages lightweight and focused.
  - Avoids unnecessary dependencies for users not using AiiDA.
  - Allows independent release cycles for core and integration packages.

---

## 2. Recommended Structure

```
bmcs_matmod/                # Core GSM modeling package (no AiiDA dependency)
    __init__.py
    gsm_lagrange/
    ...
bmcs_matmod_aiida/          # Separate integration package (depends on both bmcs_matmod and aiida-core)
    __init__.py
    data_nodes.py           # GSMResponseData, serialization logic
    calculations.py         # Calculation plugins for GSM simulations
    workchains.py           # WorkChains for scenario orchestration
    widgets.py              # (Optional) ipywidgets for AiiDA Lab visualization
    utils.py                # Data curation, extraction, etc.
```

- **bmcs_matmod**: Pure modeling, no AiiDA imports.
- **bmcs_matmod_aiida**: Imports both `bmcs_matmod` and `aiida.orm`, implements all AiiDA-specific logic.

---

## 3. Integration for Other Codes (e.g., Finite-Element Codes, Experimental Data)

### **General Pattern**

- **Each remote code (e.g., atena, abaqus, custom FE code, experimental data acquisition)** should have its own integration package for AiiDA.
- **Integration package responsibilities:**
  - Parse native output files (e.g., atena result files, CSVs, HDF5, etc.).
  - Transform results into AiiDA Data nodes (either built-in or custom subclasses).
  - Optionally, perform data curation (e.g., extract significant points, parameterize curves, compute summary statistics).
  - Provide Calculation plugins and WorkChains for automated ingestion and postprocessing.

### **Example Structure for a Remote Code Integration**

```
atena_aiida/
    __init__.py
    data_nodes.py           # Custom Data nodes for atena results
    calculations.py         # Calculation plugins for running atena jobs
    parsers.py              # Output file parsing logic
    curation.py             # Extraction/parameterization of significant results
    workchains.py           # WorkChains for orchestrating atena workflows
```

- **On the remote HPC:**  
  The integration package is installed alongside the code (e.g., atena). It can be used by AiiDA to submit jobs, parse results, and store them in the AiiDA database.

- **For experimental data:**  
  A similar integration package can be developed to parse lab data, curate it, and store it as AiiDA Data nodes.

---

## 4. Data Curation and Postprocessing

- **Integration packages** can include modules for:
  - Extracting significant points (e.g., peak stress, fatigue lifetime, characteristic times).
  - Parameterizing curves (e.g., fitting models to experimental/simulation data).
  - Generating summary statistics or derived quantities.
- **These curated results** can be stored as additional attributes or separate Data nodes in AiiDA, supporting downstream analysis and reporting.

---

## 5. Communication and Workflow

- **Remote Execution:**  
  The core simulation/experiment code runs on a remote resource. The integration package handles job submission, output parsing, and data transformation.
- **AiiDA Server:**  
  Orchestrates workflows, stores all data and provenance, and provides a central "truth" for multi-user environments.
- **AiiDA Lab:**  
  Provides user-facing interfaces for launching workflows, monitoring progress, and visualizing results.

---

## 6. Summary Table

| Component/Package         | Role/Responsibility                                         | AiiDA Dependency? |
|---------------------------|------------------------------------------------------------|-------------------|
| bmcs_matmod               | GSM models, ResponseData, simulation logic                 | No                |
| bmcs_matmod_aiida         | Data nodes, Calculation plugins, serialization, widgets    | Yes               |
| atena_aiida (example)     | atena result parsing, Data nodes, curation, plugins        | Yes               |
| expdata_aiida (example)   | Experimental data parsing, Data nodes, curation, plugins   | Yes               |
| AiiDA Server              | Workflow engine, provenance, data storage                  | Yes               |
| Compute Resource          | Runs simulations/experiments (remote/local)                | No                |

---

## 7. Recommendations

- **Keep core modeling packages (like bmcs_matmod) independent of AiiDA.**
- **Develop separate integration packages for each code or data source that needs to interact with AiiDA.**
- **Perform data curation and transformation in the integration package, not in the core code.**
- **Document the integration points and provide clear APIs for converting native results to AiiDA Data nodes.**
- **Follow this pattern for all simulation codes, experimental data, and analysis tools to ensure modularity, maintainability, and ease of installation.**

---

## 8. Next Steps

1. Refactor any AiiDA-specific code out of `bmcs_matmod` into a new integration package (e.g., `bmcs_matmod_aiida`).
2. Implement custom Data nodes, Calculation plugins, and WorkChains in the integration package.
3. For other codes (e.g., atena, experimental data), develop similar integration packages.
4. Provide documentation and examples for users on how to use the integration packages with AiiDA and AiiDA Lab.

---

**This modular architecture ensures that core scientific codes remain lightweight and broadly usable, while enabling robust, reproducible, and extensible integration with AiiDA for workflow automation, provenance, and collaborative research.**
