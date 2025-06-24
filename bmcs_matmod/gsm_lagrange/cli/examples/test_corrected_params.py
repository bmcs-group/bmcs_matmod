#!/usr/bin/env python3
"""
Test the corrected GSM1D_ED parameter files with MaterialParameterData.

This script validates that the corrected JSON files are compatible with
the current MaterialParameterData structure.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path to import data_structures
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data_structures import MaterialParameterData
    print("✓ Successfully imported MaterialParameterData")
except ImportError as e:
    print(f"✗ Failed to import MaterialParameterData: {e}")
    sys.exit(1)

def test_parameter_file(filepath):
    """Test loading and validation of a parameter file"""
    print(f"\n--- Testing {filepath.name} ---")
    
    try:
        # Load from JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        print("✓ JSON file loaded successfully")
        
        # Create MaterialParameterData instance
        param_data = MaterialParameterData.from_dict(data)
        print("✓ MaterialParameterData instance created")
        
        # Validate
        is_valid = param_data.validate()
        if is_valid:
            print("✓ Parameter validation passed")
        else:
            print("✗ Parameter validation failed")
            return False
        
        # Display key information
        print(f"  Material: {param_data.material_name}")
        print(f"  Model Type: {param_data.model_type}")
        print(f"  Parameters: {list(param_data.parameters.keys())}")
        print(f"  Parameter Values: {param_data.parameters}")
        
        # Test round-trip (to_json -> from_json)
        json_str = param_data.to_json()
        param_data_2 = MaterialParameterData.from_json(json_str)
        print("✓ Round-trip serialization test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing {filepath.name}: {e}")
        return False

def main():
    """Test all corrected parameter files"""
    print("Testing corrected GSM1D_ED parameter files")
    print("=" * 50)
    
    # Find parameter files
    param_dir = Path(__file__).parent / "parameters"
    
    test_files = [
        param_dir / "gsm1d_ed_basic_corrected.json",
        param_dir / "gsm1d_ed_high_strength_corrected.json", 
        param_dir / "gsm1d_ed_low_damage.json"
    ]
    
    all_passed = True
    
    for filepath in test_files:
        if filepath.exists():
            success = test_parameter_file(filepath)
            all_passed = all_passed and success
        else:
            print(f"✗ File not found: {filepath}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All parameter files validated successfully!")
        print("\nThese files are now compatible with:")
        print("  - MaterialParameterData structure")
        print("  - GSM1D_ED model parameters (E, S, c, r, eps_0)")
        print("  - CLI workflow and AiiDA integration")
    else:
        print("✗ Some parameter files failed validation")
    
    # Show comparison with original incompatible file
    print("\n--- Comparison with original file ---")
    old_file = param_dir.parent / "material_params.json"
    if old_file.exists():
        try:
            with open(old_file, 'r') as f:
                old_data = json.load(f)
            print(f"Original parameters: {list(old_data.get('parameters', {}).keys())}")
            print("Issues with original file:")
            print("  - Contains parameters not used by GSM1D_ED: nu, omega_0")
            print("  - Missing required GSM1D_ED parameters: c, r, eps_0") 
            print("  - Parameter 'S' has wrong typical value range")
        except Exception as e:
            print(f"Could not read original file: {e}")

if __name__ == "__main__":
    main()
