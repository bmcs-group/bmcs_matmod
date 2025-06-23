# GSM Definition Function Renaming - Summary

## Changes Made

### 1. Updated Function Names in `gsm_def_registry_new.py`

**New Primary Functions (with correct GSM def terminology):**
- `get_available_gsm_defs()` - Get available GSM definitions without importing
- `check_gsm_def_exists()` - Check if a GSM definition exists without importing  
- `get_gsm_def_module_path()` - Get module path for a GSM definition
- `get_gsm_def_class()` - Get the actual GSMDef class (imports on first access)

**Backward Compatibility Functions (kept for existing code):**
- `get_available_models()` - Calls `get_available_gsm_defs()`
- `check_model_exists()` - Calls `check_gsm_def_exists()`
- `get_model_module_path()` - Calls `get_gsm_def_module_path()`
- `get_model_class()` - Calls `get_gsm_def_class()`

### 2. Updated CLI Implementation in `cli_clean.py`

**Import Changes:**
```python
# OLD
from .cli_registry import get_available_models, check_model_exists, get_model_module_path

# NEW  
from .gsm_def_registry_new import get_available_gsm_defs, check_gsm_def_exists, get_gsm_def_module_path
```

**Class Name Changes:**
- `GSMRegistry` → `GSMDefRegistry`
- `GSMModelCLI` → `GSMDefCLI`

**New Method Names (with correct terminology):**
- `list_definitions()` - List all GSM definition names
- `get_definition()` - Get GSM definition module path by key
- `definition_exists()` - Check if a GSM definition exists

**Backward Compatibility Methods (kept for CLI compatibility):**
- `list_models()` - Calls `list_definitions()`
- `get_model()` - Calls `get_definition()`
- `model_exists()` - Calls `definition_exists()`

### 3. CLI Interface Maintained

**External CLI commands unchanged for user compatibility:**
```bash
# These still work exactly the same
python cli_clean.py --list-models        # Lists GSM definitions
python cli_clean.py --model-info ED      # Shows GSM definition info
python cli_clean.py --check-model VE     # Checks GSM definition existence
```

**Internal implementation now uses correct GSM definition functions.**

## Benefits

1. **Correct Terminology**: Functions clearly indicate they handle GSM **definitions**, not full models
2. **Backward Compatibility**: All existing code continues to work unchanged
3. **Clear API**: New code can use the properly named functions
4. **No Breaking Changes**: CLI interface remains the same for users

## Usage Examples

### For New Code (Recommended)
```python
from gsm_def_registry_new import get_available_gsm_defs, check_gsm_def_exists, get_gsm_def_class

# Get available GSM definitions
defs, paths = get_available_gsm_defs()

# Check if a definition exists
if check_gsm_def_exists('ED'):
    # Get the GSMDef class
    ED_Def = get_gsm_def_class('ED')
    # Create a model instance with parameters
    ed_model = ED_Def(E=30000, S=0.005, c=2.8, r=1.2)
```

### For Existing Code (Still Works)
```python
from gsm_def_registry_new import get_available_models, check_model_exists, get_model_class

# Old function names still work via backward compatibility
models, paths = get_available_models()
if check_model_exists('ED'):
    ED_Def = get_model_class('ED')
    ed_model = ED_Def(E=30000, S=0.005, c=2.8, r=1.2)
```

## Migration Path

- **New projects**: Use the `*_gsm_def*` function names
- **Existing projects**: No changes needed, backward compatibility maintained
- **CLI users**: No changes needed, same commands work
- **Gradual migration**: Can gradually update function calls to use new names

This ensures the distinction between GSM **definitions** (templates) and GSM **models** (instances) is clear in the codebase while maintaining full backward compatibility.
