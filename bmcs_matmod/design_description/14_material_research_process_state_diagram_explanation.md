# State Diagram Explanation: Material Research Process (Refined)

This document explains the refined state transition diagram in `14_material_research_process_state_diagram.puml`, which models the workflow for parallel binder characterization, independent calibration/validation of DM and MS models, and introduces an upscaling process.

## Diagram Overview

The refined state diagram captures the following:

1. **Experimental Campaign**
   - Starts with new binder incorporation, specimen preparation, and a sequence of standard tests (CT monotonic and fatigue).
   - Experimental data is acquired for use in both DM and MS model calibration.

2. **Independent Model Calibration**
   - **DM Calibration:** The Discrete Model (DM) is calibrated using the experimental data, then proceeds through advanced testing, refinement, and validation as an independent process.
   - **MS Calibration:** The Microplane Model (MS) is calibrated in parallel, with its own advanced testing, refinement, and validation.

3. **Advanced Testing and Refinement**
   - Both DM and MS undergo advanced experimental procedures (relaxation, visco-separation, hysteretic loop tests) and use the resulting data for further refinement.

4. **Validation**
   - Each model is validated independently through sustained and fatigue loading validation, followed by a validation comparison.
   - If further refinement is needed, the process loops back to the respective calibration step.

5. **Upscaling Process**
   - After DM refinement, an upscaling process is introduced.
   - This process uses energetic equivalence criteria to automatically transform the calibrated DM model to the macro-scale MS model.
   - The upscaled MS model can then be further refined and validated as needed.

6. **Finalization**
   - Each model (DM and MS) is finalized independently once validation is successful.

## Key Features

- **Independent Calibration and Validation:** DM and MS models are developed, refined, and validated as separate, parallel processes.
- **Upscaling Step:** The workflow introduces an explicit upscaling process, enabling automatic transformation from DM to MS based on energetic equivalence.
- **Iterative Loops:** Both DM and MS processes can iterate independently until validation criteria are met.
- **Parallelism:** The structure supports parallel execution and refinement of both models, reflecting real-world research and simulation practices.

## Usage

- This refined state diagram can guide the implementation of automated workflows (e.g., in AiiDA), project planning, and resource allocation for complex material research projects involving multi-scale modeling and upscaling.

---
