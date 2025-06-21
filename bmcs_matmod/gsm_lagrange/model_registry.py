#!/usr/bin/env python3
"""
GSM Model Registry

This module provides systematic discovery and organization of GSM models
according to the established nomenclature:

GSM1D_[MECHANISM][HARDENING]
where:
- MECHANISM: ED, VED, VEP, VEVPD, etc.
- HARDENING: (future) LIHK, NILHK, etc.

Examples:
- GSM1D_ED: Elasto-Damage
- GSM1D_VED: Visco-Elasto-Damage  
- GSM1D_VEP: Visco-Elasto-Plasticity
- GSM1D_VEVPD: Visco-Elasto-Visco-Plasticity-Damage
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Type, Optional, Tuple, Set
from dataclasses import dataclass

try:
    from .gsm_def import GSMDef
except ImportError:
    from gsm_def import GSMDef

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a GSM model"""
    name: str
    cls: Type[GSMDef]
    mechanism: str
    hardening: Optional[str] = None
    dimension: str = "1D"
    description: str = ""
    
    @property
    def full_name(self) -> str:
        """Full model name following nomenclature"""
        if self.hardening:
            return f"GSM{self.dimension}_{self.mechanism}_{self.hardening}"
        return f"GSM{self.dimension}_{self.mechanism}"

class GSMModelRegistry:
    """Registry for discovering and organizing GSM models"""
    
    # Mechanism codes and their descriptions
    MECHANISMS = {
        'E': 'Elastic',
        'ED': 'Elasto-Damage',
        'EP': 'Elasto-Plastic', 
        'EPD': 'Elasto-Plastic-Damage',
        'VE': 'Visco-Elastic',
        'VED': 'Visco-Elasto-Damage',
        'VEP': 'Visco-Elasto-Plastic',
        'EVP': 'Elasto-Visco-Plastic',
        'EVPD': 'Elasto-Visco-Plastic-Damage',
        'VEVP': 'Visco-Elasto-Visco-Plastic',
        'VEVPD': 'Visco-Elasto-Visco-Plastic-Damage',
    }
    
    # Hardening codes (future expansion)
    HARDENING_TYPES = {
        'LI': 'Linear Isotropic',
        'NI': 'Nonlinear Isotropic', 
        'LK': 'Linear Kinematic',
        'NK': 'Nonlinear Kinematic',
        'LIHK': 'Linear Isotropic + Hardening Kinematic',
        'NILHK': 'Nonlinear Isotropic + Linear Hardening Kinematic',
        # Add more combinations as needed
    }
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._discover_models()
    
    def _discover_models(self) -> None:
        """Discover all GSM models in the gsm_lagrange package"""
        try:
            # Get the current package directory
            current_dir = Path(__file__).parent
            
            # Look for GSM model files
            model_files = list(current_dir.glob('gsm1d_*.py'))
            
            for model_file in model_files:
                module_name = model_file.stem
                try:
                    # Import the module dynamically
                    module = importlib.import_module(f'.{module_name}', package=__package__)
                    
                    # Find GSMDef subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, GSMDef) and 
                            obj is not GSMDef and 
                            hasattr(obj, 'F_engine')):
                            
                            model_info = self._parse_model_name(name, obj)
                            if model_info:
                                self.models[name.lower()] = model_info
                                # Also add with full nomenclature key
                                nomenclature_key = model_info.mechanism.lower()
                                if nomenclature_key not in self.models:
                                    self.models[nomenclature_key] = model_info
                                logger.debug(f"Registered GSM model: {name} -> {model_info.mechanism}")
                            
                except ImportError as e:
                    logger.warning(f"Could not import {module_name}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing {module_name}: {e}")
            
            logger.info(f"Discovered {len(self.models)} GSM models")
            
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
    
    def _parse_model_name(self, name: str, cls: Type[GSMDef]) -> Optional[ModelInfo]:
        """Parse model name according to nomenclature"""
        try:
            # Expected format: GSM1D_MECHANISM or GSM1D_MECHANISM_HARDENING
            if not name.startswith('GSM1D_'):
                return None
            
            parts = name.split('_')
            if len(parts) < 2:
                return None
            
            dimension = parts[0][3:]  # Extract dimension (1D, 2D, 3D)
            mechanism = parts[1]
            hardening = parts[2] if len(parts) > 2 else None
            
            # Validate mechanism
            if mechanism not in self.MECHANISMS:
                logger.warning(f"Unknown mechanism '{mechanism}' in model {name}")
                # Still register it but without description
                mechanism_desc = f"Unknown mechanism: {mechanism}"
            else:
                mechanism_desc = self.MECHANISMS[mechanism]
            
            # Validate hardening if present
            hardening_desc = ""
            if hardening:
                if hardening in self.HARDENING_TYPES:
                    hardening_desc = f" with {self.HARDENING_TYPES[hardening]}"
                else:
                    hardening_desc = f" with {hardening} hardening"
            
            description = mechanism_desc + hardening_desc
            
            return ModelInfo(
                name=name,
                cls=cls,
                mechanism=mechanism,
                hardening=hardening,
                dimension=dimension,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error parsing model name {name}: {e}")
            return None
    
    def get_model(self, key: str) -> Optional[Type[GSMDef]]:
        """Get model class by key (case-insensitive)"""
        model_info = self.models.get(key.lower())
        return model_info.cls if model_info else None
    
    def get_model_info(self, key: str) -> Optional[ModelInfo]:
        """Get model info by key (case-insensitive)"""
        return self.models.get(key.lower())
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models"""
        # Return unique models (avoid duplicates from multiple keys)
        seen_names = set()
        models = []
        for model_info in self.models.values():
            if model_info.name not in seen_names:
                models.append(model_info)
                seen_names.add(model_info.name)
        return sorted(models, key=lambda m: m.name)
    
    def list_by_mechanism(self, mechanism: str) -> List[ModelInfo]:
        """List models by mechanism type"""
        return [model for model in self.list_models() 
                if model.mechanism.upper() == mechanism.upper()]
    
    def list_by_dimension(self, dimension: str) -> List[ModelInfo]:
        """List models by dimension"""
        return [model for model in self.list_models() 
                if model.dimension.upper() == dimension.upper()]
    
    def get_available_keys(self) -> List[str]:
        """Get all available model keys"""
        return list(self.models.keys())
    
    def get_available_mechanisms(self) -> List[str]:
        """Get all available mechanism types"""
        mechanisms = set()
        for model in self.list_models():
            mechanisms.add(model.mechanism)
        return sorted(list(mechanisms))
    
    def format_model_table(self) -> str:
        """Format available models as a table"""
        models = self.list_models()
        if not models:
            return "No GSM models found."
        
        # Calculate column widths
        name_width = max(len("Model Name"), max(len(m.name) for m in models))
        mech_width = max(len("Mechanism"), max(len(m.mechanism) for m in models))
        desc_width = max(len("Description"), max(len(m.description) for m in models))
        
        # Format table
        header = f"{'Model Name':<{name_width}} | {'Mechanism':<{mech_width}} | {'Description':<{desc_width}}"
        separator = "-" * len(header)
        
        lines = [header, separator]
        for model in models:
            line = f"{model.name:<{name_width}} | {model.mechanism:<{mech_width}} | {model.description:<{desc_width}}"
            lines.append(line)
        
        return "\n".join(lines)

# Global registry instance
_registry: Optional[GSMModelRegistry] = None

def get_registry() -> GSMModelRegistry:
    """Get the global model registry (singleton)"""
    global _registry
    if _registry is None:
        _registry = GSMModelRegistry()
    return _registry
