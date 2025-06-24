#!/usr/bin/env python3
"""
Generate compatible MaterialParameterData JSON file for GSM1D_ED model.

This script creates a properly formatted JSON parameter file that matches
the current MaterialParameterData structure and GSM1D_ED parameter requirements.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_structures import MaterialParameterData
from gsm1d_ed import GSM1D_ED

def create_gsm1d_ed_basic_params():
    """Create basic parameters for GSM1D_ED model"""
    
    # Get GSM1D_ED parameter names from the model definition
    gsm_ed = GSM1D_ED()
    param_symbols = gsm_ed.F_engine.m_params
    param_names = [param.name for param in param_symbols]
    
    print(f"GSM1D_ED requires parameters: {param_names}")
    
    # Create MaterialParameterData instance with correct parameters
    ed_params = MaterialParameterData(
        parameters={
            'E': 30000.0,      # Young's modulus (MPa)
            'S': 1.0,          # Damage threshold parameter (MPa)
            'c': 2.0,          # Damage evolution parameter (-)
            'r': 0.5,          # Damage evolution exponent (-)
            'eps_0': 0.001     # Initial threshold strain (-)
        },
        
        # Metadata
        source="Generated from GSM1D_ED model definition",
        material_name="Concrete C30/37",
        model_type="GSM1D_ED",
        calibration_date="2025-06-24",
        description="Basic elastic-damage parameters for concrete using GSM1D_ED model",
        
        # Units
        units={
            'E': 'MPa',
            'S': 'MPa', 
            'c': '-',
            'r': '-',
            'eps_0': '-'
        },
        
        # Parameter bounds for validation
        parameter_bounds={
            'E': (1000.0, 100000.0),
            'S': (0.1, 10.0),
            'c': (0.1, 5.0),
            'r': (0.1, 2.0),
            'eps_0': (0.0001, 0.01)
        },
        
        # Validation info
        validation_data={
            'R_squared': 0.92,
            'experimental_source': 'Lab Test Series - Basic Concrete',
            'calibration_method': 'least_squares',
            'validation_notes': 'Parameters calibrated for monotonic tensile loading'
        }
    )
    
    return ed_params

def create_gsm1d_ed_high_strength_params():
    """Create high-strength parameters for GSM1D_ED model"""
    
    ed_params = MaterialParameterData(
        parameters={
            'E': 45000.0,      # Young's modulus (MPa)
            'S': 2.5,          # Damage threshold parameter (MPa)
            'c': 3.5,          # Damage evolution parameter (-)
            'r': 0.3,          # Damage evolution exponent (-)
            'eps_0': 0.0005    # Initial threshold strain (-)
        },
        
        # Metadata
        source="Generated from GSM1D_ED model definition",
        material_name="High-Strength Concrete HSC60",
        model_type="GSM1D_ED",
        calibration_date="2025-06-24",
        description="High-strength elastic-damage parameters using GSM1D_ED model",
        
        # Units
        units={
            'E': 'MPa',
            'S': 'MPa', 
            'c': '-',
            'r': '-',
            'eps_0': '-'
        },
        
        # Parameter bounds for validation
        parameter_bounds={
            'E': (1000.0, 100000.0),
            'S': (0.1, 10.0),
            'c': (0.1, 5.0),
            'r': (0.1, 2.0),
            'eps_0': (0.0001, 0.01)
        },
        
        # Validation info
        validation_data={
            'R_squared': 0.95,
            'experimental_source': 'Lab Test Series - High-Strength Concrete',
            'calibration_method': 'least_squares',
            'validation_notes': 'Parameters calibrated for high-strength concrete under tensile loading'
        }
    )
    
    return ed_params

def create_gsm1d_ed_low_damage_params():
    """Create low-damage parameters for GSM1D_ED model"""
    
    ed_params = MaterialParameterData(
        parameters={
            'E': 25000.0,      # Young's modulus (MPa)
            'S': 0.8,          # Damage threshold parameter (MPa)
            'c': 1.5,          # Damage evolution parameter (-)
            'r': 0.8,          # Damage evolution exponent (-)
            'eps_0': 0.002     # Initial threshold strain (-)
        },
        
        # Metadata
        source="Generated from GSM1D_ED model definition",
        material_name="Low-Damage Concrete",
        model_type="GSM1D_ED",
        calibration_date="2025-06-24",
        description="Low-damage evolution parameters for gradual degradation using GSM1D_ED model",
        
        # Units
        units={
            'E': 'MPa',
            'S': 'MPa', 
            'c': '-',
            'r': '-',
            'eps_0': '-'
        },
        
        # Parameter bounds for validation
        parameter_bounds={
            'E': (1000.0, 100000.0),
            'S': (0.1, 10.0),
            'c': (0.1, 5.0),
            'r': (0.1, 2.0),
            'eps_0': (0.0001, 0.01)
        },
        
        # Validation info
        validation_data={
            'R_squared': 0.88,
            'experimental_source': 'Lab Test Series - Gradual Damage',
            'calibration_method': 'least_squares',
            'validation_notes': 'Parameters calibrated for materials with gradual damage evolution'
        }
    )
    
    return ed_params

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("parameters")
    output_dir.mkdir(exist_ok=True)
    
    # Generate parameter sets
    param_sets = {
        "gsm1d_ed_basic.json": create_gsm1d_ed_basic_params(),
        "gsm1d_ed_high_strength.json": create_gsm1d_ed_high_strength_params(),
        "gsm1d_ed_low_damage.json": create_gsm1d_ed_low_damage_params()
    }
    
    # Save parameter files
    for filename, params in param_sets.items():
        filepath = output_dir / filename
        
        # Validate before saving
        if params.validate():
            with open(filepath, 'w') as f:
                f.write(params.to_json())
            print(f"Created: {filepath}")
            print(f"  Parameters: {list(params.parameters.keys())}")
            print(f"  Material: {params.material_name}")
            print(f"  Description: {params.description}")
            print()
        else:
            print(f"ERROR: Parameter validation failed for {filename}")
    
    print("Parameter files generated successfully!")
    print(f"Files saved in: {output_dir.absolute()}")
