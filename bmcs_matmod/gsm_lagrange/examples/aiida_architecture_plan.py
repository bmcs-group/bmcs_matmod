#!/usr/bin/env python3
"""
AiiDA Integration Architecture - Implementation Plan

Based on analysis in notebooks/aiida_integration_strategies.ipynb, here's the 
recommended refactoring to properly handle AiiDA profiles and entry points.

CURRENT ISSUES:
1. ResponseDataNode in core/ tries to create AiiDA nodes without profile checks
2. CLIResponseData redundancy (already removed)
3. Missing entry point definitions for direct Python integration
4. No graceful degradation when AiiDA unavailable

RECOMMENDED CHANGES:
"""

# 1. MOVE ResponseDataNode to aiida_plugin/
"""
BEFORE:
bmcs_matmod/gsm_lagrange/core/response_data_node.py

AFTER:  
bmcs_matmod/gsm_lagrange/aiida_plugin/response_data_node.py

RATIONALE:
- Separates AiiDA-dependent code from core functionality
- Makes it clear that this module requires AiiDA
- Allows core/ to remain AiiDA-free
"""

# 2. ADD entry points to setup.cfg
setup_cfg_additions = """
[options.entry_points]
# Console scripts for CLI access
console_scripts =
    gsm-simulate = bmcs_matmod.gsm_lagrange.cli.cli_gsm:main

# AiiDA calculation entry points (direct Python integration)
aiida.calculations =
    gsm.lagrange = bmcs_matmod.gsm_lagrange.aiida_plugin.calculations:GSMLagrangeCalculation

# AiiDA data entry points
aiida.data =
    gsm.response = bmcs_matmod.gsm_lagrange.aiida_plugin.response_data_node:ResponseDataNode

# AiiDA workflow entry points  
aiida.workflows =
    gsm.material_characterization = bmcs_matmod.gsm_lagrange.aiida_plugin.workflows:MaterialCharacterizationWorkChain

# Custom plugin discovery (enables: pkg_resources.iter_entry_points('bmcs_matmod.gsm_models'))
bmcs_matmod.gsm_models =
    gsm1d_ed = bmcs_matmod.gsm_lagrange.models.gsm1d_ed:GSM1D_ED
    gsm1d_ep = bmcs_matmod.gsm_lagrange.models.gsm1d_ep:GSM1D_EP
    gsm1d_epd = bmcs_matmod.gsm_lagrange.models.gsm1d_epd:GSM1D_EPD
    gsm1d_evp = bmcs_matmod.gsm_lagrange.models.gsm1d_evp:GSM1D_EVP
    gsm1d_evpd = bmcs_matmod.gsm_lagrange.models.gsm1d_evpd:GSM1D_EVPD
    gsm1d_ve = bmcs_matmod.gsm_lagrange.models.gsm1d_ve:GSM1D_VE
    gsm1d_ved = bmcs_matmod.gsm_lagrange.models.gsm1d_ved:GSM1D_VED
    gsm1d_vevp = bmcs_matmod.gsm_lagrange.models.gsm1d_vevp:GSM1D_VEVP
    gsm1d_vevpd = bmcs_matmod.gsm_lagrange.models.gsm1d_vevpd:GSM1D_VEVPD
"""

# 3. CREATE aiida_plugin/ directory structure
aiida_plugin_structure = """
bmcs_matmod/gsm_lagrange/aiida_plugin/
├── __init__.py                 # Safe imports with fallbacks
├── response_data_node.py       # Moved from core/
├── calculations.py             # AiiDA Calculation classes
├── workflows.py                # AiiDA WorkChain classes
├── utils.py                    # Profile management utilities
└── parsers.py                  # Output parsers for calculations
"""

# 4. UPDATE CLI to use aiida_plugin
cli_updates = """
# OLD: from ..core.response_data_node import ResponseDataNode
# NEW: 
try:
    from ..aiida_plugin.response_data_node import ResponseDataNode, create_response_data_node
    RESPONSE_DATA_NODE_AVAILABLE = True
except ImportError:
    ResponseDataNode = None
    create_response_data_node = None
    RESPONSE_DATA_NODE_AVAILABLE = False
"""

# 5. BENEFITS of this architecture
benefits = """
✓ CLEAN SEPARATION: Core modules have zero AiiDA dependencies
✓ GRACEFUL DEGRADATION: Package works without AiiDA installed
✓ MULTIPLE INTEGRATION PATHS: CLI + direct Python entry points
✓ DISCOVERABILITY: Entry points enable plugin discovery
✓ PROFILE MANAGEMENT: Automatic profile loading with fallbacks
✓ MAINTAINABILITY: Clear module boundaries and responsibilities
"""

# 6. USAGE EXAMPLES after refactoring

# Standalone usage (no AiiDA required)
standalone_example = """
from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
from bmcs_matmod.gsm_lagrange.models.gsm1d_ed import GSM1D_ED

material = GSMModel(GSM1D_ED)
response_data = material.get_F_response(strain_history, time_array)
# Works without AiiDA!
"""

# AiiDA integration (optional)
aiida_example = """
from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
from bmcs_matmod.gsm_lagrange.models.gsm1d_ed import GSM1D_ED

# Core simulation (no AiiDA)
material = GSMModel(GSM1D_ED)
response_data = material.get_F_response(strain_history, time_array)

# Optional AiiDA storage
try:
    from bmcs_matmod.gsm_lagrange.aiida_plugin.response_data_node import create_response_data_node
    node = create_response_data_node(response_data, auto_load_profile=True)
    print(f"Stored in AiiDA: {node.uuid}")
except ImportError:
    print("AiiDA not available, using JSON storage")
    with open("results.json", "w") as f:
        json.dump(response_data.to_dict(), f)
"""

# Entry point usage (no CLI needed!)
entry_point_example = """
import pkg_resources

# Discover GSM models via entry points
for entry_point in pkg_resources.iter_entry_points('bmcs_matmod.gsm_models'):
    model_class = entry_point.load()
    print(f"Found model: {entry_point.name} -> {model_class}")

# Discover AiiDA calculations via entry points
for entry_point in pkg_resources.iter_entry_points('aiida.calculations'):
    if entry_point.name.startswith('gsm.'):
        calc_class = entry_point.load()
        print(f"Found AiiDA calculation: {entry_point.name} -> {calc_class}")
"""

if __name__ == "__main__":
    print("=== AiiDA Integration Architecture Plan ===")
    print("\nSee notebooks/aiida_integration_strategies.ipynb for full analysis.")
    print("\nKey recommendations:")
    print("1. Move ResponseDataNode to aiida_plugin/")
    print("2. Add entry points for discoverability")
    print("3. Implement graceful AiiDA profile handling")
    print("4. Keep core/ modules AiiDA-free")
    print("5. Use entry points for direct Python integration")
    print("\nThis enables both standalone and AiiDA-integrated usage!")
