#!/usr/bin/env python3
"""
Example: Two-Level Data Architecture Usage

This example demonstrates how to use the two-level data architecture
with ResponseData (active) and ResponseDataNode (persistent) objects.
"""

import numpy as np
from bmcs_matmod.gsm_lagrange.core.response_data import ResponseData
from bmcs_matmod.gsm_lagrange.cli.cli_response_data import CLIResponseData

# Try to import AiiDA components
try:
    from bmcs_matmod.gsm_lagrange.aiida_plugin.response_data_node import ResponseDataNode, create_response_data_node
    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False
    print("AiiDA not available - will demonstrate JSON output only")


def create_mock_simulation_data():
    """Create mock simulation data for demonstration"""
    # Create mock simulation data
    n_steps = 100
    t_t = np.linspace(0, 1.0, n_steps)
    eps_t = np.linspace(0, 0.01, n_steps)[:, np.newaxis]  # Shape (n_steps, 1)
    sig_t = 20000 * eps_t[:, :, np.newaxis]  # Shape (n_steps, 1, 1)
    
    # Mock internal variables
    damage = np.maximum(0, (eps_t[:, 0] - 0.005) / 0.005)  # Damage evolution
    plastic_strain = np.maximum(0, eps_t[:, 0] - 0.003)    # Plastic strain
    
    # Create variable containers
    from bmcs_matmod.gsm_lagrange.core.response_data import ResponseDataContainer
    
    # Mock variable objects with codenames
    class MockVar:
        def __init__(self, codename, shape=()):
            self.codename = codename
            self.shape = shape
        def __str__(self):
            return f"Var_{self.codename}"
    
    Eps_vars = (MockVar('omega', ()), MockVar('eps_p', ()))
    Sig_vars = (MockVar('Y', ()), MockVar('X', ()))
    
    Eps_t_dict = {
        'omega': damage,
        'eps_p': plastic_strain
    }
    
    Sig_t_dict = {
        'Y': -0.5 * damage,  # Damage force
        'X': 1000 * plastic_strain  # Plastic force
    }
    
    # Create ResponseData
    rd = ResponseData(
        t_t=t_t,
        eps_t=eps_t,
        sig_t=sig_t,
        Eps_t_flat=np.column_stack([damage, plastic_strain]),
        Sig_t_flat=np.column_stack([-0.5 * damage, 1000 * plastic_strain]),
        Eps_vars=Eps_vars,
        Sig_vars=Sig_vars,
        Eps_t=ResponseDataContainer(Eps_t_dict),
        Sig_t=ResponseDataContainer(Sig_t_dict),
        iter_t=np.ones(n_steps),
        lam_t=np.ones(n_steps)
    )
    
    return rd


def demonstrate_two_level_architecture():
    """Demonstrate the two-level data architecture"""
    print("=== GSM Two-Level Data Architecture Demo ===\n")
    
    # 1. Create active simulation data (Level 1)
    print("1. Creating active simulation data (ResponseData)...")
    rd = create_mock_simulation_data()
    print(f"   Created: {rd}")
    
    # 2. Wrap with CLI serialization capabilities
    print("\n2. Wrapping with CLI serialization...")
    cli_rd = CLIResponseData(rd)
    print(f"   Wrapped: {cli_rd}")
    
    # 3. JSON serialization (always available)
    print("\n3. JSON Serialization:")
    print("   a) Summary JSON:")
    summary = cli_rd.to_summary_dict()
    print(f"      - Steps: {summary['simulation_summary']['n_steps']}")
    print(f"      - Variables: {summary['available_variables']['internal_variables']}")
    
    print("   b) Detailed JSON:")
    detailed = cli_rd.to_json_dict()
    print(f"      - Time series length: {len(detailed['time_series']['time'])}")
    print(f"      - Internal variables: {list(detailed['internal_variables'].keys())}")
    
    # 4. Visualization (if matplotlib available)
    print("\n4. Visualization capabilities:")
    try:
        stats = rd.get_summary_stats()
        print(f"   - Max strain: {stats['strain_stats']['max']:.6f}")
        print(f"   - Max stress: {stats['stress_stats']['max']:.2f}")
        print("   - Plotting methods available: plot_stress_strain(), plot_time_series(), create_dashboard()")
    except ImportError:
        print("   - Matplotlib not available for plotting")
    
    # 5. AiiDA integration (if available)
    if AIIDA_AVAILABLE:
        print("\n5. AiiDA Integration (Level 2):")
        try:
            # Create persistent storage node
            print("   a) Creating ResponseDataNode...")
            simulation_metadata = {
                'model_type': 'GSM1D_EPD',
                'parameters': {'E': 20000.0, 'S': 1.0},
                'description': 'Demo elastic-plastic-damage simulation'
            }
            
            # Note: In practice, you'd want to store this
            node = cli_rd.to_aiida_responsedata_node(simulation_metadata, store=False)
            print(f"      Created (not stored): {node}")
            
            # Demonstrate round-trip conversion
            print("   b) Round-trip conversion...")
            rd_reconstructed = node.to_response_data()
            cli_rd_reconstructed = CLIResponseData(rd_reconstructed)
            print(f"      Reconstructed: {cli_rd_reconstructed}")
            
            # Show both objects have same interface
            print("   c) Shared interface demonstration:")
            print(f"      Original n_steps: {len(rd.t_t)}")
            print(f"      Node n_steps: {len(node.t_t)}")
            print(f"      Same variables: {list(rd.Eps_t.keys()) == list(node.Eps_t.keys())}")
            
        except Exception as e:
            print(f"   AiiDA node creation failed: {e}")
    else:
        print("\n5. AiiDA Integration: Not available")
    
    # 6. Usage patterns
    print("\n6. Usage Patterns:")
    print("   a) During simulation: Use ResponseData for active monitoring")
    print("   b) For storage: Convert to ResponseDataNode with metadata")
    print("   c) For analysis: Both support same visualization interface")
    print("   d) For data transfer: Use CLIResponseData for format flexibility")
    
    return rd, cli_rd


def demonstrate_incremental_simulation():
    """Demonstrate how this architecture supports incremental data writing"""
    print("\n\n=== Incremental Simulation Support ===\n")
    
    print("Future development pattern for IBVP solvers:")
    print("1. ResponseData with .append_step() method for real-time updates")
    print("2. Periodic snapshots to ResponseDataNode for checkpointing")  
    print("3. Real-time monitoring through shared visualization interface")
    print("4. Field variables extension: strain_field_t, stress_field_t, etc.")
    print("5. Boundary condition tracking in metadata")
    
    # Mock incremental pattern
    print("\nMock incremental writing pattern:")
    rd = create_mock_simulation_data()
    
    # Simulate incremental steps
    for i in [10, 25, 50, 75, 100]:
        print(f"   Step {i}: Current max strain = {rd.eps_t[i-1, 0]:.6f}")
        # In real implementation: rd.append_step(new_data)
        # In real implementation: if i % 25 == 0: create_checkpoint_node(rd)


if __name__ == "__main__":
    # Run demonstrations
    rd, cli_rd = demonstrate_two_level_architecture()
    demonstrate_incremental_simulation()
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("Architecture benefits:")
    print("✅ Clean separation: Core simulation vs. persistence")
    print("✅ Performance: In-memory active data, efficient storage")
    print("✅ Flexibility: Multiple output formats, optional AiiDA")
    print("✅ Shared interface: Same visualization for both levels")
    print("✅ Future-ready: Supports incremental writing and IBVP extension")
