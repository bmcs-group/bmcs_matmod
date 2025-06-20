# Architecture for Distributed, Modular Workchains in the Fatrescon Platform

This document elaborates on the distribution of roles, responsibilities, and package structure for a multidisciplinary, collaborative material research platform (e.g., Fatrescon), combining autonomous simulation/experimental codes, integration layers, and a central AiiDA/AiiDA Lab-based workflow environment.

---

## 1. Architectural Principles

- **Autonomy:** Each simulation or experimental codebase (e.g., `bmcs_matmod.gsm_lagrange`, FE codes, experimental data pipelines) is developed and maintained independently, with its own release cycle and scientific focus.
- **Integration Layer:** Dedicated integration packages (e.g., `aiida-bmcs-matmod`, `aiida-atena`, `aiida-expdata`) bridge the autonomous codes with the central AiiDA platform, providing Data nodes, Calculation plugins, WorkChains, and widgets.
- **Central Platform:** A shared AiiDA/AiiDA Lab instance (e.g., `fatrescon-aiidalab`) hosts high-level, cross-disciplinary workflows, interactive widgets, and collaborative tools, enabling team-wide access, reproducibility, and learning.
- **Extensibility:** Shared utilities (e.g., visualization widgets, base Data nodes) are factored into a common package (e.g., `aiida-fatrescon-integ`) to avoid duplication and promote best practices.

---

## 2. Package and Responsibility Distribution

### **A. Autonomous Simulation/Experimental Packages**
- **Examples:** `bmcs_matmod`, `atena`, `expdata_pipeline`
- **Responsibilities:**
  - Core scientific/modeling logic
  - No AiiDA or workflow dependencies
  - Usable standalone for development, testing, and publication

### **B. Integration Packages**
- **Examples:** `aiida-bmcs-matmod`, `aiida-atena`, `aiida-expdata`
- **Responsibilities:**
  - Define AiiDA Data nodes for simulation/experimental results
  - Implement Calculation plugins and WorkChains for running, parsing, and curating results
  - Provide widgets for AiiDA Lab (e.g., `GSM1DVerificationWorkChain` with model selection and visualization)
  - Installed both on remote compute resources (for job execution) and on the central AiiDA/AiiDA Lab server (for workflow orchestration and visualization)
  - Expose the full range of models and workflows from the underlying codebase to the central platform

### **C. Central Platform and Shared Utilities**
- **Examples:** `fatrescon-aiidalab`, `aiida-fatrescon-integ`
- **Responsibilities:**
  - Host top-level, cross-disciplinary workflows (e.g., hydration process, fatigue resistance characterization)
  - Provide shared widgets and visualization tools (e.g., time-space field viewers, summary dashboards)
  - Offer base classes and utilities for integration packages to ensure consistency and reduce duplication
  - Serve as the main entry point for users to launch, monitor, and analyze workflows

---

## 3. Hierarchy and Sharing of Workchains

- **Top-Level Workchains:**  
  Defined in `fatrescon-aiidalab`, these orchestrate complex, multi-step research processes (e.g., hydration prediction, fatigue resistance assessment), potentially invoking lower-level WorkChains from integration packages.
- **Model-Specific Workchains:**  
  Provided by integration packages (e.g., `aiida-bmcs-matmod`), these encapsulate domain-specific workflows (e.g., `GSM1DVerificationWorkChain` for 1D GSM models), including model selection, parameterization, and result analysis.
- **Widget Integration:**  
  Each integration package can contribute widgets for interactive exploration and visualization, which are registered with the central platform for discoverability and reuse.
- **Shared Utilities:**  
  Common visualization and data handling components are factored into `aiida-fatrescon-integ` or similar, and imported by all integration packages.

---

## 4. Example: GSM Model Verification WorkChain

- **Location:** `aiida-bmcs-matmod`
- **Functionality:**
  - Presents a widget for selecting a GSM model (e.g., visco-elasto-visco-plastic-damage with hardening)
  - Allows parameter input and scenario configuration
  - Runs the simulation remotely, stores results as Data nodes
  - Visualizes results using shared widgets (e.g., time-history plots, 3D fields)
- **Integration:**  
  Exposed in `fatrescon-aiidalab` as a reusable workflow for team members to learn, compare, and benchmark models.

---

## 5. Installation and Deployment

- **Remote Compute Resources:**  
  Install the relevant integration package(s) (e.g., `aiida-bmcs-matmod`) and the core simulation code (e.g., `bmcs_matmod`).
- **Central AiiDA/AiiDA Lab Server:**  
  Install all integration packages, the central platform package (`fatrescon-aiidalab`), and shared utilities (`aiida-fatrescon-integ`).
- **AiiDA Lab Plugins:**  
  Integration packages can be distributed as AiiDA Lab plugins, enabling easy installation and update via the AiiDA Lab interface.

---

## 6. Managing Fragmentation and Complexity

- **Clear API Contracts:**  
  Define clear interfaces for Data nodes, Calculation plugins, and widgets to ensure interoperability.
- **Shared Base Packages:**  
  Use a common integration utility package for shared logic and visualization.
- **Documentation and Examples:**  
  Provide comprehensive documentation and example workflows for each integration package and the central platform.
- **Community Coordination:**  
  Regularly review and refactor shared components to avoid duplication and drift.

---

## 7. Summary Table

| Layer/Package              | Role/Responsibility                                      | Example(s)                |
|----------------------------|---------------------------------------------------------|---------------------------|
| Core Simulation/Experiment | Autonomous scientific code, no AiiDA dependency         | bmcs_matmod, atena        |
| Integration Layer          | Data nodes, plugins, workchains, widgets for AiiDA      | aiida-bmcs-matmod, aiida-atena |
| Shared Utilities           | Visualization, base classes, common tools               | aiida-fatrescon-integ     |
| Central Platform           | Top-level workflows, orchestration, user interface      | fatrescon-aiidalab        |

---

## 8. Recommendations

- **Promote autonomy and modularity** by keeping scientific codes and integration logic in separate packages.
- **Centralize shared workflows and visualization** in the main platform, with clear extension points for integration packages.
- **Encourage reuse and consistency** through shared utility packages and documentation.
- **Support collaborative, multidisciplinary research** by exposing all relevant models and workflows in the central AiiDA Lab environment.

---

**This architecture balances the autonomy of scientific code development with the power of a shared, collaborative research platform, enabling both innovation and integration across the Fatrescon team.**
