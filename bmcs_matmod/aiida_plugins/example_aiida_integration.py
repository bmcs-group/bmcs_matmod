#!/usr/bin/env python3
"""
Example AiiDA integration script for GSM material modeling

This script demonstrates how to:
1. Set up AiiDA codes and computers for GSM simulations
2. Run monotonic characterization workflows
3. Execute fatigue testing and S-N curve construction
4. Export and analyze results

Prerequisites:
- AiiDA core installed and configured
- bmcs_matmod package installed with AiiDA plugins
- GSM CLI properly configured
"""

import numpy as np
from aiida import orm, engine, load_profile
from aiida.plugins import WorkflowFactory, CalculationFactory, DataFactory

# Load AiiDA profile
load_profile()

# Load GSM plugins
GSMMonotonicWorkChain = WorkflowFactory('gsm_monotonic')
GSMFatigueWorkChain = WorkflowFactory('gsm_fatigue')
GSMSNCurveWorkChain = WorkflowFactory('gsm_sn_curve')


def setup_gsm_code():
    """Set up AiiDA code for GSM CLI"""
    
    # Check if code already exists
    try:
        code = orm.Code.get_from_string('gsm-cli@localhost')
        print(f"Using existing GSM code: {code}")
        return code
    except:
        pass
    
    # Create new computer if needed
    try:
        computer = orm.Computer.get('localhost')
    except:
        computer = orm.Computer(
            label='localhost',
            hostname='localhost',
            transport_type='core.local',
            scheduler_type='core.direct'
        ).store()
        computer.configure()
    
    # Create GSM CLI code
    code = orm.Code(
        input_plugin_name='gsm_simulation',
        remote_computer_uuid=computer.uuid,
        remote_absolute_path='/usr/local/bin/gsm-cli',  # Adjust path as needed
        label='gsm-cli'
    )
    code.description = 'GSM CLI for material model simulations'
    code.store()
    
    print(f"Created GSM code: {code}")
    return code


def example_monotonic_simulation():
    """Example monotonic loading simulation"""
    
    print("\\n=== Monotonic Loading Simulation ===")
    
    # Set up code
    code = setup_gsm_code()
    
    # Define material parameters for elasto-damage model
    material_params = {
        'E': 30000.0,    # Young's modulus (MPa)
        'S': 1.0,        # Damage parameter
        'c': 2.0,        # Damage evolution parameter
        'r': 0.9         # Damage threshold
    }
    
    # Set up workchain inputs
    inputs = {
        'gsm_code': code,
        'gsm_model': orm.Str('GSM1D_ED'),
        'formulation': orm.Str('F'),
        'material_parameters': orm.Dict(dict=material_params),
        'max_strain': orm.Float(0.01),  # 1% strain
        'num_steps': orm.Int(100),
        'metadata': {
            'label': 'Example Monotonic ED Test',
            'description': 'Monotonic loading of elasto-damage model to 1% strain'
        }
    }
    
    # Submit workchain
    print("Submitting monotonic workchain...")
    workchain = engine.submit(GSMMonotonicWorkChain, **inputs)
    print(f"Workchain submitted: {workchain}")
    
    return workchain


def example_fatigue_simulation():
    """Example fatigue simulation"""
    
    print("\\n=== Fatigue Simulation ===")
    
    # Set up code
    code = setup_gsm_code()
    
    # Define material parameters
    material_params = {
        'E': 30000.0,
        'S': 1.0,
        'c': 2.0,
        'r': 0.9,
        'alpha': 0.1,    # Fatigue parameter
        'beta': 1.5      # Fatigue parameter
    }
    
    # Set up fatigue inputs
    inputs = {
        'gsm_code': code,
        'gsm_model': orm.Str('GSM1D_VED'),  # Visco-elasto-damage for fatigue
        'formulation': orm.Str('F'),
        'material_parameters': orm.Dict(dict=material_params),
        'stress_amplitude': orm.Float(150.0),  # 150 MPa amplitude
        'stress_mean': orm.Float(50.0),        # 50 MPa mean stress
        'max_cycles': orm.Int(5000),
        'failure_strain': orm.Float(0.05),     # 5% failure strain
        'metadata': {
            'label': 'Example Fatigue Test',
            'description': 'Fatigue test at 150 MPa amplitude'
        }
    }
    
    # Submit workchain
    print("Submitting fatigue workchain...")
    workchain = engine.submit(GSMFatigueWorkChain, **inputs)
    print(f"Workchain submitted: {workchain}")
    
    return workchain


def example_sn_curve_construction():
    """Example S-N curve construction"""
    
    print("\\n=== S-N Curve Construction ===")
    
    # Set up code
    code = setup_gsm_code()
    
    # Define material parameters
    material_params = {
        'E': 30000.0,
        'S': 1.0,
        'c': 2.0,
        'r': 0.9,
        'alpha': 0.1,
        'beta': 1.5
    }
    
    # Define stress levels for S-N curve
    stress_levels = [200.0, 175.0, 150.0, 125.0, 100.0, 75.0, 50.0]
    
    # Set up S-N curve inputs
    inputs = {
        'gsm_code': code,
        'gsm_model': orm.Str('GSM1D_VED'),
        'formulation': orm.Str('F'),
        'material_parameters': orm.Dict(dict=material_params),
        'stress_levels': orm.List(list=stress_levels),
        'max_cycles': orm.Int(10000),
        'failure_strain': orm.Float(0.05),
        'metadata': {
            'label': 'Example S-N Curve',
            'description': 'S-N curve construction for VED model'
        }
    }
    
    # Submit workchain
    print("Submitting S-N curve workchain...")
    workchain = engine.submit(GSMSNCurveWorkChain, **inputs)
    print(f"Workchain submitted: {workchain}")
    
    return workchain


def wait_and_analyze_results(workchain):
    """Wait for workchain completion and analyze results"""
    
    print(f"\\nWaiting for workchain {workchain.pk} to complete...")
    
    # In practice, you would use verdi process show <PK> to monitor
    # or set up monitoring in a separate script
    
    print("To monitor the workchain, run:")
    print(f"  verdi process show {workchain.pk}")
    print(f"  verdi process list -a")
    
    print("\\nTo analyze results after completion:")
    print(f"  verdi node show {workchain.pk}")
    print(f"  verdi data dict show <output_node_pk>")
    print(f"  verdi data array show <array_output_pk>")


def export_results_example():
    """Example of exporting results to JSON"""
    
    print("\\n=== Result Export Example ===")
    
    # This would be run after workchains complete
    print("After workchain completion, you can export results:")
    print("""
# Python script to export results
from bmcs_matmod.aiida_plugins.exporters import GSMJSONExporter
from aiida import orm

# Get completed workchain
wc = orm.load_node(<workchain_pk>)

# Export monotonic results
if 'monotonic_results' in wc.outputs:
    GSMJSONExporter.export_simulation_results(
        wc.outputs.monotonic_results, 
        'monotonic_results.json'
    )

# Export S-N curve data
if 'sn_curve_data' in wc.outputs:
    GSMJSONExporter.export_sn_curve(
        wc.outputs.sn_curve_data, 
        'sn_curve.json'
    )
    GSMJSONExporter.export_sn_curve(
        wc.outputs.sn_curve_data, 
        'sn_curve.csv', 
        format='csv'
    )
""")


def main():
    """Main demonstration function"""
    
    print("=== GSM AiiDA Integration Example ===")
    print("This script demonstrates AiiDA integration for GSM material modeling")
    
    try:
        # Run examples
        mono_wc = example_monotonic_simulation()
        fatigue_wc = example_fatigue_simulation()
        sn_wc = example_sn_curve_construction()
        
        # Show monitoring information
        for wc in [mono_wc, fatigue_wc, sn_wc]:
            wait_and_analyze_results(wc)
        
        # Show export examples
        export_results_example()
        
        print("\\n=== Summary ===")
        print("Submitted workchains for:")
        print(f"  Monotonic loading: {mono_wc.pk}")
        print(f"  Fatigue testing: {fatigue_wc.pk}")
        print(f"  S-N curve construction: {sn_wc.pk}")
        
        print("\\nMonitor progress with: verdi process list -a")
        print("View results with: verdi node show <pk>")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure AiiDA is properly configured and GSM CLI is available")


if __name__ == "__main__":
    main()
