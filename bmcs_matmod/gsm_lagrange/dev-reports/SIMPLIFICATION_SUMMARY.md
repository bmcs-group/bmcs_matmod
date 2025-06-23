# GSM Model Discovery Simplification Summary

## Changes Made

The GSM model discovery has been simplified to use only the `issubclass(obj, GSMDef)` criterion as requested, removing the additional `F_engine` attribute checks.

## Updated Files

### 1. Notebook: `present_gsm_derivation.ipynb`
- **Cell 3**: Updated model discovery logic to use only `issubclass(obj, GSMDef) and obj is not GSMDef`
- Removed the `hasattr(obj, 'F_engine')` condition
- Now discovers all classes that inherit from GSMDef regardless of engine attributes

### 2. Simple Registry: `simple_registry.py`
- Updated `_discover_models()` method
- Simplified class detection to use only the issubclass criterion
- Removed F_engine requirement for model registration

### 3. Clean CLI: `cli_clean.py`
- Updated model listing to show engine information as optional
- Made engine checking more robust (shows "Not defined" if engines don't exist)

## Detection Logic

**Before:**
```python
if (issubclass(obj, GSMDef) and 
    obj is not GSMDef and 
    hasattr(obj, 'F_engine')):
```

**After:**
```python
if issubclass(obj, GSMDef) and obj is not GSMDef:
```

## Benefits

1. **Simpler Detection**: No dependency on specific attributes like F_engine
2. **More Inclusive**: Discovers all GSMDef subclasses, even incomplete ones
3. **Cleaner Code**: Easier to understand and maintain
4. **Better for Development**: Useful for discovering work-in-progress models

## Usage

The notebook cell now automatically discovers all GSM model classes in the current directory that inherit from GSMDef, creates a registry for easy access, and provides multiple access methods (full name, mechanism, lowercase aliases).

Example output from the notebook:
- Lists all discovered GSM1D_* classes
- Shows mechanism extracted from class name
- Creates a convenient registry dictionary
- Provides multiple access keys for each model

This simplified approach makes the model discovery more robust and easier to use for development and documentation purposes.
