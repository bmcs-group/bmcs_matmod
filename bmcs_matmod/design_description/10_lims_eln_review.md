# Review of LIMS and ELN Systems for Lab Workflow Integration

## What are LIMS and ELN?

- **LIMS (Laboratory Information Management System):**
  - Software designed to manage samples, laboratory workflows, test results, inventory, and compliance.
  - Tracks specimens, manages test requests, schedules equipment, and stores results.
  - Often integrates with instruments and supports regulatory requirements (e.g., ISO, GLP).

- **ELN (Electronic Lab Notebook):**
  - Digital replacement for paper lab notebooks.
  - Used to record experimental procedures, observations, raw data, and analysis.
  - Supports collaboration, search, versioning, and data security.

## Open-Source and Open-Architecture LIMS/ELN Options

### **LIMS**

- **Bika LIMS / Senaite**
  - Open-source, web-based, modular.
  - Used in research, clinical, and environmental labs.
  - Supports sample tracking, workflow automation, reporting, and instrument integration.
  - [https://www.senaite.com/](https://www.senaite.com/)

- **LabKey Server**
  - Open-source, extensible, strong in data integration and collaboration.
  - [https://www.labkey.org/](https://www.labkey.org/)

- **OpenLIMS**
  - Lightweight, open-source, basic sample and workflow management.
  - [http://open-lims.org/](http://open-lims.org/)

### **ELN**

- **eLabFTW**
  - Open-source, compliant, supports experiment documentation, inventory, scheduling.
  - [https://www.elabftw.net/](https://www.elabftw.net/)

- **Chemotion ELN**
  - Open-source, chemistry-focused, supports data management and sharing.
  - [https://chemotion.net/](https://chemotion.net/)

- **LabArchives (not open-source, but widely used)**
  - Cloud-based, strong integration, APIs for automation.

- **Jupyter Notebooks**
  - Not a traditional ELN, but widely used for computational experiments, data analysis, and documentation.

## Hybrid Solutions: Project Management + LIMS/ELN

### **Integration Approaches**

- **API-Based Integration**
  - Many LIMS/ELN systems provide REST APIs for data exchange.
  - Project management tools (e.g., OpenProject, Redmine) can be extended to pull/push data from/to LIMS/ELN.
  - Enables automatic status updates, sample tracking, and experiment scheduling.

- **Custom Dashboards**
  - Use BI tools (e.g., Grafana, Metabase) or web frameworks (e.g., Django, Flask) to create dashboards that aggregate data from LIMS/ELN and project management systems.

- **Workflow Orchestration**
  - Use workflow engines (e.g., AiiDA, Apache Airflow) to coordinate computational/experimental steps, with LIMS/ELN for data capture and project management for scheduling.

### **Examples of Successful Hybrid Solutions**

- **Senaite + OpenProject**
  - Labs have integrated Senaite LIMS with OpenProject for project/task management and sample tracking.
  - Custom plugins or middleware synchronize project milestones with sample/test status.

- **eLabFTW + Redmine**
  - Some academic labs use eLabFTW for experiment documentation and Redmine for project management, with scripts to link experiment records to project tasks.

- **Commercial Examples**
  - Many commercial LIMS/ELN vendors offer integrations with ERP or project management systems, but open-source solutions require more custom development.

## Summary

- **LIMS and ELN systems** are essential for managing lab samples, workflows, and documentation.
- **Open-source options** (Senaite, eLabFTW, Chemotion) are available and can be integrated with project management tools via APIs or custom plugins.
- **Hybrid solutions** are feasible and have been implemented in research and industrial labs, but often require IT/development support for integration.
- **Best practice:** Use LIMS/ELN for sample and experiment management, project management tools for scheduling and coordination, and connect them via APIs for a seamless lab workflow.

---
