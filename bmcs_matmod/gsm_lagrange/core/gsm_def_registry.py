#!/usr/bin/env python3
"""
Unified GSM Definition Registry

This module provides automatic discovery of GSM definitions by scanning for files
matching the pattern 'gsm1d_*.py' and importing GSMDef classes only when accessed.

GSM Definitions define the constitutive behavior and can be instantiated with 
specific material parameters to create GSM Models for particular loading scenarios.

Key features:
- Path-based discovery (safe for CLI)
- Lazy loading of GSMDef classes (import only when needed)
- Single unified approach for both CLI and notebook use
"""

import os
import importlib
import inspect
from typing import Dict, List, Type, Tuple, Optional, Union, Any


class LazyGSMDefRegistry:
    """
    Registry that discovers GSM definitions by file scanning and imports GSMDef classes on-demand.
    
    GSM Definitions are classes that define constitutive behavior. They can be instantiated
    with specific material parameters to create GSM Models for loading scenarios.
    
    This approach combines the safety of path-based discovery (no upfront imports)
    with the functionality of class-based access (import when needed).
    """
    
    def __init__(self, search_dir: Optional[str] = None, debug: bool = False):
        # Default to the models directory relative to this file
        if search_dir is None:
            current_dir = os.path.dirname(__file__)
            search_dir = os.path.join(current_dir, '..', 'models')
        self.search_dir = search_dir
        self.debug = debug
        self._def_paths = {}  # def_name -> module_path
        self._access_keys = {}  # all_keys -> def_name  
        self._loaded_def_classes = {}  # def_name -> actual_GSMDef_class (cache)
        self._discover_definitions()
    
    def _discover_definitions(self) -> None:
        """Discover GSM definitions by scanning files (no imports yet)"""
        # Get list of potential GSM definition files
        gsm_files = [f for f in os.listdir(self.search_dir) 
                     if f.endswith('.py') and f.startswith('gsm1d_')]
        
        if self.debug:
            print(f"Found {len(gsm_files)} potential GSM definition files:")
            for file_name in gsm_files:
                print(f"  {file_name}")
        
        # Extract definition information from file names
        for file_name in gsm_files:
            module_name = file_name[:-3]  # Remove .py extension
            
            # Infer definition name from module name: gsm1d_ed.py -> GSM1D_ED
            if module_name.startswith('gsm1d_'):
                mechanism_part = module_name[6:]  # Remove 'gsm1d_'
                def_name = f"GSM1D_{mechanism_part.upper()}"
                module_path = f"bmcs_matmod.gsm_lagrange.models.{module_name}"
                
                # Store the definition path
                self._def_paths[def_name] = module_path
                
                # Create all access variants
                self._access_keys[def_name] = def_name
                self._access_keys[def_name.lower()] = def_name
                self._access_keys[mechanism_part.upper()] = def_name
                self._access_keys[mechanism_part.lower()] = def_name
                
                if self.debug:
                    print(f"✓ Detected definition: {def_name}")
                    print(f"  Module: {module_name}")
                    print(f"  Mechanism: {mechanism_part.upper()}")
        
        if self.debug:
            print(f"Total discovered definitions: {len(self._def_paths)}")
            print(f"Total access keys: {len(self._access_keys)}")
    
    def _load_def_class(self, def_name: str) -> Type:
        """Lazily load the actual GSMDef class when needed"""
        if def_name in self._loaded_def_classes:
            return self._loaded_def_classes[def_name]
        
        if def_name not in self._def_paths:
            raise ValueError(f"GSM definition '{def_name}' not found")
        
        module_path = self._def_paths[def_name]
        
        try:
            # Import GSMDef for type checking
            try:
                from .gsm_def import GSMDef
            except ImportError:
                try:
                    from gsm_def import GSMDef
                except ImportError:
                    # Try absolute import
                    from bmcs_matmod.gsm_lagrange.core.gsm_def import GSMDef
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Find the GSMDef subclass in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, GSMDef) and 
                    obj is not GSMDef and 
                    name == def_name):
                    self._loaded_def_classes[def_name] = obj
                    if self.debug:
                        print(f"✓ Loaded GSMDef class: {def_name} from {module_path}")
                    return obj
            
            raise ValueError(f"No GSMDef subclass '{def_name}' found in {module_path}")
            
        except Exception as e:
            raise ImportError(f"Failed to load GSM definition {def_name} from {module_path}: {e}")
    
    def get_def_names(self) -> List[str]:
        """Get list of discovered GSM definition names (no imports)"""
        return sorted(self._def_paths.keys())
    
    def get_access_keys(self) -> List[str]:
        """Get all available access keys including aliases (no imports)"""
        return sorted(self._access_keys.keys())
    
    def has_def(self, key: str) -> bool:
        """Check if a GSM definition exists (no imports)"""
        return key in self._access_keys
    
    def get_module_path(self, key: str) -> Optional[str]:
        """Get module path for a GSM definition (no imports)"""
        def_name = self._access_keys.get(key)
        if def_name:
            return self._def_paths[def_name]
        return None
    
    def get_def_class(self, key: str) -> Type:
        """Get the actual GSMDef class (imports on first access)"""
        def_name = self._access_keys.get(key)
        if not def_name:
            available = list(self._access_keys.keys())
            raise ValueError(f"GSM definition '{key}' not found. Available: {available}")
        
        return self._load_def_class(def_name)
    
    def get_all_def_classes(self) -> Dict[str, Type]:
        """Get all GSMDef classes as a dictionary (imports all)"""
        result = {}
        for key, def_name in self._access_keys.items():
            if def_name not in result:  # Avoid duplicates
                result[key] = self._load_def_class(def_name)
        return result
    
    def list_defs_with_classes(self) -> List[Tuple[str, Type]]:
        """Get list of (def_name, GSMDef_class) tuples (imports all)"""
        result = []
        for def_name in self.get_def_names():
            def_class = self._load_def_class(def_name)
            result.append((def_name, def_class))
        return result


# Global registry instance
_registry = None


def _get_registry(debug: bool = False) -> LazyGSMDefRegistry:
    """Get or create the global registry instance"""
    global _registry
    if _registry is None:
        _registry = LazyGSMDefRegistry(debug=debug)
    return _registry


# Public API functions that match the original interfaces

def discover_gsm_defs(search_dir: Optional[str] = None, debug: bool = False) -> Tuple[List[Tuple[str, Type]], Dict[str, Type]]:
    """
    Discover GSM definitions (compatible with original function).
    
    Returns:
    --------
    Tuple containing:
    - List of (def_name, GSMDef_class) tuples
    - Registry dictionary with multiple access keys
    """
    registry = LazyGSMDefRegistry(search_dir, debug)
    defs = registry.list_defs_with_classes()
    classes_dict = registry.get_all_def_classes()
    return defs, classes_dict


def get_gsm_defs(debug: bool = False) -> Dict[str, Type]:
    """
    Get GSM defs registry dictionary (compatible with original function).
    
    Returns a dictionary that allows access by:
    - Full class name (e.g., 'GSM1D_ED')
    - Lowercase class name (e.g., 'gsm1d_ed')
    - Mechanism name (e.g., 'ED')
    - Lowercase mechanism name (e.g., 'ed')
    """
    registry = _get_registry(debug)
    return registry.get_all_def_classes()


def list_gsm_defs(debug: bool = False) -> List[Tuple[str, Type]]:
    """
    Get list of discovered GSM defs (compatible with original function).
    
    Returns:
    --------
    List of (def_name, GSMDef_class) tuples sorted by name
    """
    registry = _get_registry(debug)
    return registry.list_defs_with_classes()


# CLI-friendly functions (no imports until needed)

def get_available_gsm_defs(debug: bool = False) -> Tuple[List[str], Dict[str, str]]:
    """Get available GSM definitions without importing them (for CLI)"""
    registry = _get_registry(debug)
    defs = registry.get_def_names()
    paths = {key: registry.get_module_path(key) for key in registry.get_access_keys()}
    return defs, paths


def check_gsm_def_exists(def_key: str, debug: bool = False) -> bool:
    """Check if a GSM definition exists without importing it (for CLI)"""
    registry = _get_registry(debug)
    return registry.has_def(def_key)


def get_gsm_def_module_path(def_key: str, debug: bool = False) -> Optional[str]:
    """Get the module path for a GSM definition (for CLI)"""
    registry = _get_registry(debug)
    return registry.get_module_path(def_key)


def get_gsm_def_class(def_key: str, debug: bool = False) -> Type:
    """Get the actual GSMDef class (imports on first access)"""
    registry = _get_registry(debug)
    return registry.get_def_class(def_key)


# Display and utility functions

def print_gsm_defs(debug: bool = True) -> None:
    """
    Print discovered GSM definitions with details.
    """
    registry = _get_registry(debug)
    defs = registry.list_defs_with_classes()
    
    if not debug:
        print("\nDiscovered GSM Definitions:")
        print("=" * 60)
        for i, (name, obj) in enumerate(defs, 1):
            print(f"{i:2d}. {name}")
            
            # Extract mechanism from name
            if name.startswith('GSM1D_'):
                mechanism = name[6:]  # Remove 'GSM1D_'
                print(f"    Mechanism: {mechanism}")
            
            print(f"    GSMDef Class: {obj}")
            print()


# Mechanism descriptions for display purposes
MECHANISM_DESCRIPTIONS = {
    'ED': 'Elasto-Damage Model',
    'VE': 'Visco-Elastic Model',
    'VED': 'Visco-Elasto-Damage Model', 
    'EP': 'Elasto-Plastic Model',
    'EPD': 'Elasto-Plastic-Damage Model',
    'EVP': 'Elasto-Visco-Plastic Model',
    'EVPD': 'Elasto-Visco-Plastic-Damage Model',
    'VEVP': 'Visco-Elasto-Visco-Plastic Model',
    'VEVPD': 'Visco-Elasto-Visco-Plastic-Damage Model'
}


def get_mechanism_description(mechanism: str) -> str:
    """Get description for a mechanism code."""
    return MECHANISM_DESCRIPTIONS.get(mechanism.upper(), f"{mechanism} Model")





if __name__ == "__main__":
    # When run as script, demonstrate both modes
    print("Unified GSM Definition Registry Demo")
    print("=" * 50)
    
    print("\n1. Path-based discovery (no imports):")
    registry = LazyGSMDefRegistry(debug=True)
    print(f"Found definitions: {registry.get_def_names()}")
    print(f"Access keys: {registry.get_access_keys()}")
    
    print("\n2. On-demand GSMDef class loading:")
    if registry.get_def_names():
        first_def = registry.get_def_names()[0]
        print(f"Loading GSMDef class for: {first_def}")
        try:
            def_class = registry.get_def_class(first_def)
            print(f"✓ Successfully loaded: {def_class}")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
    
    print("\n3. Traditional API compatibility:")
    try:
        defs, registry_dict = discover_gsm_defs(debug=False)
        print(f"✓ discover_gsm_defs: {len(defs)} definitions")
        
        defs_dict = get_gsm_defs(debug=False)
        print(f"✓ get_gsm_defs: {len(defs_dict)} entries")
        
        defs_list = list_gsm_defs(debug=False)
        print(f"✓ list_gsm_defs: {len(defs_list)} definitions")
        
    except Exception as e:
        print(f"✗ Traditional API failed: {e}")
