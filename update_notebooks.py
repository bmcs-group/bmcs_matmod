#!/usr/bin/env python3
"""
Script to update test notebook files for the new GSMSymbBox constructor signature.
The initial_state_fn parameter is now required and must be the first parameter.
"""

import json
import sys
import re

def update_gsm_box_constructor_calls(notebook_path):
    """Update GSMSymbBox constructor calls in a Jupyter notebook."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        updated = False
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_text = ''.join(source)
                else:
                    source_text = source
                
                # Pattern to match GSMSymbBox constructor calls
                pattern = r'(box_\w+\s*=\s*GSMSymbBox\s*\(\s*)(.*?)(initial_state_fn\s*=\s*StateFunction\.\w+)(.*?)(\))'
                
                def replace_constructor(match):
                    prefix = match.group(1)
                    before_initial = match.group(2)
                    initial_state_param = match.group(3)
                    after_initial = match.group(4)
                    suffix = match.group(5)
                    
                    # Remove the initial_state_fn from its current position
                    remaining_params = before_initial + after_initial
                    
                    # Clean up any extra commas
                    remaining_params = re.sub(r',\s*,', ',', remaining_params)
                    remaining_params = re.sub(r'^\s*,\s*', '', remaining_params)
                    remaining_params = re.sub(r',\s*$', '', remaining_params)
                    
                    # Build the new constructor call
                    if remaining_params.strip():
                        new_call = f"{prefix}{initial_state_param},\n    {remaining_params}{suffix}"
                    else:
                        new_call = f"{prefix}{initial_state_param}{suffix}"
                    
                    return new_call
                
                new_source_text = re.sub(pattern, replace_constructor, source_text, flags=re.DOTALL)
                
                if new_source_text != source_text:
                    updated = True
                    # Convert back to list format if it was originally a list
                    if isinstance(cell['source'], list):
                        cell['source'] = new_source_text.splitlines(keepends=True)
                    else:
                        cell['source'] = new_source_text
        
        if updated:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            print(f"✓ Updated {notebook_path}")
            return True
        else:
            print(f"- No changes needed in {notebook_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error updating {notebook_path}: {e}")
        return False

def main():
    # List of notebook files to update
    notebooks = [
        '/home/rch/Coding/bmcs_matmod/test_gsm_symb_box_simplification.ipynb'
    ]
    
    print("Updating notebook files for new GSMSymbBox constructor signature...")
    print("=" * 60)
    
    total_updated = 0
    for notebook_path in notebooks:
        if update_gsm_box_constructor_calls(notebook_path):
            total_updated += 1
    
    print("=" * 60)
    print(f"Updated {total_updated} notebook file(s)")
    print("\nChanges made:")
    print("- Moved initial_state_fn to be the first required parameter")
    print("- Updated all GSMSymbBox constructor calls accordingly")

if __name__ == "__main__":
    main()
