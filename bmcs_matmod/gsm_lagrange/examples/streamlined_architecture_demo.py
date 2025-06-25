#!/usr/bin/env python3
"""
Streamlined Two-Level Architecture Demo

This script demonstrates the new streamlined architecture for GSM simulation data:

1. ResponseData (Active): In-memory simulation data during computation
2. ResponseDataNode (Persistent): AiiDA storage with all serialization capabilities

The redundant CLIResponseData layer has been removed, simplifying the design.
"""

import sys
from pathlib import Path

# Add project to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from bmcs_matmod.gsm_lagrange.core.gsm_model import GSMModel
from bmcs_matmod.gsm_lagrange.models.gsm1d_ed import GSM1D_ED

# Try to import ResponseDataNode for demonstration
try:
    from bmcs_matmod.gsm_lagrange.aiida_plugin.response_data_node import ResponseDataNode, create_response_data_node
    AIIDA_AVAILABLE = True
    print("✓ AiiDA available - full two-level architecture demonstration")
except ImportError:
    AIIDA_AVAILABLE = False
    print("⚠ AiiDA not available - showing ResponseData only")

def main():
    print("\n" + "="*60)
    print("STREAMLINED TWO-LEVEL ARCHITECTURE DEMONSTRATION")
    print("="*60)
    
    # === LEVEL 1: Active ResponseData ===
    print("\n1. LEVEL 1: Active ResponseData (simulation)")
    print("-" * 40)
    
    # Create and configure material model
    material = GSMModel(GSM1D_ED)
    material.E = 30000  # Young's modulus
    material.S = 1.0    # Strength
    
    # Define loading
    n_steps = 100
    strain_max = 0.01
    strain_history = np.linspace(0, strain_max, n_steps)
    time_array = np.linspace(0, 1.0, n_steps)
    
    # Execute simulation - creates active ResponseData
    print(f"Executing simulation with {n_steps} steps...")
    rd = material.get_F_response(strain_history, time_array)
    
    print(f"✓ Active ResponseData created:")
    print(f"  - Time steps: {len(rd.t_t)}")
    print(f"  - Strain range: [{rd.eps_t.min():.6f}, {rd.eps_t.max():.6f}]")
    print(f"  - Stress range: [{rd.sig_t.min():.6f}, {rd.sig_t.max():.6f}]")
    print(f"  - Internal variables: {list(rd.Eps_t.keys())}")
    print(f"  - Thermodynamic forces: {list(rd.Sig_t.keys())}")
    
    # === LEVEL 2: Persistent ResponseDataNode ===
    if AIIDA_AVAILABLE:
        print("\n2. LEVEL 2: Persistent ResponseDataNode (storage & serialization)")
        print("-" * 50)
        
        # Convert to persistent storage
        simulation_metadata = {
            "material_model": "GSM1D_ED",
            "parameters": {"E": 30000, "S": 1.0},
            "loading_type": "monotonic_tension",
            "max_strain": strain_max
        }
        
        print("Converting to ResponseDataNode...")
        rd_node = create_response_data_node(rd, simulation_metadata, store=False)
        
        print("✓ ResponseDataNode created with:")
        print(f"  - UUID: {rd_node.uuid}")
        print(f"  - Stored: {rd_node.is_stored}")
        print(f"  - Arrays: {list(rd_node.get_arraynames())}")
        print(f"  - Metadata: {rd_node.get_simulation_metadata()}")
        
        # === Demonstration of Unified Interface ===
        print("\n3. UNIFIED INTERFACE: Same methods for both levels")
        print("-" * 45)
        
        print("ResponseData interface:")
        print(f"  - rd.t_t.shape: {rd.t_t.shape}")
        print(f"  - rd.eps_t.shape: {rd.eps_t.shape}")
        print(f"  - rd.Eps_t['eps_el'][0]: {rd.Eps_t['eps_el'][0]}")
        
        print("\nResponseDataNode interface (identical):")
        print(f"  - rd_node.t_t.shape: {rd_node.t_t.shape}")
        print(f"  - rd_node.eps_t.shape: {rd_node.eps_t.shape}")
        print(f"  - rd_node.Eps_t['eps_el'][0]: {rd_node.Eps_t['eps_el'][0]}")
        
        # === JSON Serialization Capabilities ===
        print("\n4. JSON SERIALIZATION: Built into ResponseDataNode")
        print("-" * 45)
        
        # Summary JSON
        summary = rd_node.to_summary_dict()
        print("Summary JSON structure:")
        for key in summary.keys():
            print(f"  - {key}")
        
        # Simple results (CLI compatibility)
        simple = rd_node.get_simple_results()
        print(f"\nSimple results: {len(simple)} keys")
        print(f"  - Time steps: {simple['n_steps']}")
        print(f"  - Has UUID: {'uuid' in simple}")
        
        # Format for different outputs
        print("\n5. FLEXIBLE OUTPUT FORMATS")
        print("-" * 30)
        
        json_summary = rd_node.format_for_output('json')
        print(f"JSON summary format: {list(json_summary.keys())}")
        
        json_detailed = rd_node.format_for_output('json_detailed')
        print(f"JSON detailed format: {list(json_detailed.keys())}")
        
        node_format = rd_node.format_for_output('node')
        print(f"Node format returns: {type(node_format).__name__}")
        
        # === CLI Integration ===
        print("\n6. CLI INTEGRATION: Direct usage in cli_gsm.py")
        print("-" * 45)
        
        print("Before (three-level with redundancy):")
        print("  ResponseData -> CLIResponseData -> JSON/AiiDA")
        print("  ❌ CLIResponseData duplicated all JSON methods")
        
        print("\nAfter (streamlined two-level):")
        print("  ResponseData -> ResponseDataNode -> JSON/AiiDA")
        print("  ✓ ResponseDataNode has all serialization built-in")
        print("  ✓ CLI uses ResponseDataNode directly")
        print("  ✓ No redundancy, cleaner architecture")
        
    else:
        print("\n2. LEVEL 2: ResponseDataNode not available")
        print("-" * 35)
        print("Install AiiDA to see the full two-level architecture:")
        print("  pip install aiida-core")
    
    # === Benefits Summary ===
    print("\n" + "="*60)
    print("ARCHITECTURE BENEFITS")
    print("="*60)
    print("✓ Eliminated redundancy (CLIResponseData removed)")
    print("✓ Unified interface between active and persistent data")
    print("✓ All serialization logic centralized in ResponseDataNode")
    print("✓ CLI uses ResponseDataNode directly (no wrapper needed)")
    print("✓ Efficient binary storage + JSON serialization in one class")
    print("✓ AiiDA provenance + database querying + visualization")
    print("✓ Cleaner, maintainable two-level design")

if __name__ == "__main__":
    main()
