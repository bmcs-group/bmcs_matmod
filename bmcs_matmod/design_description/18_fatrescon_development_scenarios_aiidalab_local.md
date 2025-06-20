# 18. Practical Development Scenarios for AiiDA/AiiDA Lab Integration in Fatrescon

This document summarizes practical strategies for developing and integrating autonomous simulation codes (e.g., `bmcs_matmod`) with AiiDA and AiiDA Lab, focusing on local development, dependency management, and the workflow for moving from standalone code to full platform integration.

---

## 1. Recommended Development and Integration Workflow

### A. Start with Local Integration Subpackage

- Develop an `aiida_integration` subpackage within your main codebase (e.g., `bmcs_matmod/aiida_integration/`).
- This allows rapid prototyping and debugging of AiiDA plugins, Data nodes, and WorkChains without switching between multiple repositories.
- Install `aiida-core` in your local Python environment to enable direct use of AiiDA APIs and `verdi` commands.

### B. Use VSCode and Notebooks for Rapid Iteration

- Develop and test AiiDA WorkChains, Calculation functions, and Data nodes directly in Python scripts or Jupyter notebooks within VSCode.
- You can inspect, debug, and iterate quickly, using the familiar notebook interface.

### C. Split Integration Package When Stable

- Once the integration logic stabilizes, move the `aiida_integration` subpackage into a standalone package (e.g., `bmcs_matmod_aiidalab`).
- This separation ensures that the core code remains free of AiiDA dependencies, while the integration package can depend on both `bmcs_matmod` and `aiida-core`.

---

## 2. AiiDA and AiiDA Lab: Local vs. Remote Usage

### A. AiiDA Core

- **AiiDA core** (`aiida-core`) is required for developing and running WorkChains, Data nodes, and calculations.
- You can run AiiDA workflows directly in a notebook or Python script as long as you have a configured AiiDA profile (via `verdi quicksetup`).
- For simple, local development, you do **not** need to run the AiiDA daemon (`verdi daemon start`) if you use the `run()` or `submit()` functions in a blocking (foreground) mode. The daemon is only required for asynchronous, background execution and for production deployments.

### B. AiiDA Lab

- **AiiDA Lab** is a web-based platform for launching, monitoring, and visualizing AiiDA workflows with interactive widgets.
- For local development, you can install AiiDA Lab in your environment (`pip install aiidalab`) or run it via Docker.
- **You do not need AiiDA Lab to develop or debug AiiDA plugins, Data nodes, or WorkChains.** You can do all development and testing in VSCode and notebooks.
- AiiDA Lab becomes important when you want to provide user-friendly, interactive workflows and visualization for end users or collaborators.

### C. Typical Local Development Flow

1. Install `aiida-core` in your environment.
2. Run `verdi quicksetup` to create a local AiiDA profile and database.
3. Develop and test WorkChains, Data nodes, and calculations in VSCode or Jupyter notebooks.
4. Optionally, install and run AiiDA Lab locally for widget-based workflow execution and visualization.
5. When ready, deploy integration packages and widgets to a central AiiDA Lab instance for team-wide use.

---

## 3. Dependencies and Execution Modes

- **aiida-core** is required for all AiiDA-based development and execution.
- **aiidalab** is only needed for web-based, widget-driven workflows and is not required for backend/plugin development.
- **verdi daemon** is only needed for background, asynchronous execution (e.g., on a server or for production). For local, interactive development, you can run workflows synchronously in a notebook or script.
- **Remote execution** and widget integration can be added later, once the core logic is stable.

---

## 4. More Details: Using AiiDA Lab Locally

- You can install AiiDA Lab in your environment with `pip install aiidalab` or run it in Docker.
- To use AiiDA Lab locally:
  1. Ensure your AiiDA profile is set up and working (`verdi status`).
  2. Start AiiDA Lab (`aiidalab-launch` or via Docker).
  3. Access the web interface (usually at `http://localhost:8888`).
  4. Install or develop AiiDA Lab apps (widgets, workflow GUIs) as needed.
- For most plugin and workflow development, you do **not** need to use AiiDA Lab until you want to provide a user-facing interface.

---

## 5. Summary Table

| Scenario                        | aiida-core | aiidalab | verdi daemon | Typical Use Case                |
|----------------------------------|:----------:|:--------:|:------------:|:-------------------------------:|
| Local plugin/workchain dev       |    ✔       |    ✗     |      ✗       | VSCode, notebooks, CLI          |
| Local widget/GUI dev             |    ✔       |    ✔     |      ✗       | AiiDA Lab, Jupyter, browser     |
| Production/remote workflow exec  |    ✔       |    ✔     |      ✔       | Central server, multi-user      |
| Interactive notebook execution   |    ✔       |    ✗     |      ✗       | Debugging, prototyping          |

✔ = required, ✗ = not required

---

## 6. Recommendations

- **For rapid development:**  
  Work in a single VSCode workspace, add an `aiida_integration` subpackage, and install `aiida-core` locally.
- **For debugging and prototyping:**  
  Use notebooks or scripts to run and inspect WorkChains without the need for AiiDA Lab or the daemon.
- **For user-facing workflows:**  
  Add AiiDA Lab and develop widgets when you want to expose workflows to end users.
- **For production or team deployment:**  
  Move integration code to a standalone package, deploy to a central AiiDA/AiiDA Lab server, and use the daemon for background execution.

---

## 7. Key Points

- You do **not** need AiiDA Lab or the daemon for most local development and debugging.
- You can run and inspect WorkChains in a notebook or script as long as `aiida-core` is installed and a profile is set up.
- Split the integration package later for clean dependency management and deployment.
- Use AiiDA Lab when you want to provide interactive, web-based workflows for users.

---
