#!/usr/bin/env python3
"""
GSM Model Registry - Production Version

This module provides robust discovery and organization of GSM models
with fallback to mock models for demonstration purposes.
"""

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Type, Optional, Tuple, Set, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import with fallback
try:
    from .gsm_def import GSMDef
    REAL_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Real GSM models not available, using mock models")
    REAL_MODELS_AVAILABLE = False
    
    # Create a mock base class for demonstration
    class GSMDef:
        """Mock GSMDef for demonstration"""
        F_engine = None
        G_engine = None
        
        def get_F_response(self, eps_history, time_array):
            import numpy as np
            return {
                'stress': 30000.0 * eps_history * (1 - 0.1 * eps_history),
                'strain': eps_history, 
                'time': time_array
            }
            
        def get_G_response(self, sig_history, time_array):
            import numpy as np
            return {
                'strain': sig_history / 30000.0,
                'stress': sig_history,
                'time': time_array
            }

@dataclass
class ModelInfo:
    """Information about a GSM model"""
    name: str
    cls: Type[GSMDef]
    mechanism: str
    hardening: Optional[str] = None
    dimension: str = "1D"
    description: str = ""
    is_mock: bool = False
    
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
    }
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._discover_models()
    
    def _discover_models(self) -> None:
        """Discover all GSM models"""
        if REAL_MODELS_AVAILABLE:
            self._discover_real_models()
        else:
            self._create_mock_models()
    
    def _discover_real_models(self) -> None:
        """Discover real GSM models in the package"""
        try:
            # Get the current package directory  
            current_dir = Path(__file__).parent
            
            # Look for GSM model files
            model_files = list(current_dir.glob('gsm1d_*.py'))
            logger.info(f"Found {len(model_files)} potential GSM model files")
            
            for model_file in model_files:
                module_name = model_file.stem
                try:
                    # Import the module dynamically
                    if __package__:
                        module = importlib.import_module(f'.{module_name}', package=__package__)
                    else:
                        # Fallback for direct execution
                        import sys
                        sys.path.insert(0, str(current_dir))
                        module = importlib.import_module(module_name)
                    
                    # Find GSMDef subclasses in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, GSMDef) and 
                            obj is not GSMDef and 
                            hasattr(obj, 'F_engine')):
                            
                            model_info = self._parse_model_name(name, obj)
                            if model_info:
                                self._register_model(model_info)
                                logger.debug(f"Registered real GSM model: {name}")
                            
                except Exception as e:
                    logger.warning(f"Could not import or process {module_name}: {e}")
            
            if not self.models:
                logger.warning("No real GSM models found, falling back to mock models")
                self._create_mock_models()
            else:
                logger.info(f"Successfully discovered {len(self.models)} real GSM models")
                
        except Exception as e:
            logger.error(f"Error discovering real models: {e}")
            self._create_mock_models()
    
    def _create_mock_models(self) -> None:
        """Create mock models for demonstration"""
        logger.info("Creating mock GSM models for demonstration")
        
        # Define model configurations following the nomenclature
        model_configs = [
            ('GSM1D_ED', 'ED'),
            ('GSM1D_VE', 'VE'),
            ('GSM1D_VED', 'VED'),
            ('GSM1D_EP', 'EP'),
            ('GSM1D_EPD', 'EPD'),
            ('GSM1D_VEP', 'VEP'),
            ('GSM1D_EVP', 'EVP'),
            ('GSM1D_EVPD', 'EVPD'),
            ('GSM1D_VEVP', 'VEVP'),
            ('GSM1D_VEVPD', 'VEVPD'),
        ]
        
        for name, mechanism in model_configs:
            # Create a mock class dynamically
            mock_cls = type(name, (GSMDef,), {
                '__doc__': f"Mock {name} model for demonstration",
                'F_engine': True,  # Mock attribute
                'G_engine': True,  # Mock attribute
            })
            
            description = self.MECHANISMS.get(mechanism, f"Unknown mechanism: {mechanism}")
            
            model_info = ModelInfo(
                name=name,
                cls=mock_cls,
                mechanism=mechanism,
                dimension="1D",
                description=description,
                is_mock=True
            )
            
            self._register_model(model_info)
        
        logger.info(f"Created {len(self.models)} mock GSM models")
    
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
            
            # Get description
            mechanism_desc = self.MECHANISMS.get(mechanism, f"Unknown mechanism: {mechanism}")
            hardening_desc = ""
            if hardening:
                hardening_desc = f" with {self.HARDENING_TYPES.get(hardening, hardening)} hardening"
            
            description = mechanism_desc + hardening_desc
            
            return ModelInfo(
                name=name,
                cls=cls,
                mechanism=mechanism,
                hardening=hardening,
                dimension=dimension,
                description=description,
                is_mock=False
            )
            
        except Exception as e:
            logger.error(f"Error parsing model name {name}: {e}")
            return None
    
    def _register_model(self, model_info: ModelInfo) -> None:
        """Register a model with multiple key variants"""
        # Register with full name
        self.models[model_info.name.lower()] = model_info
        
        # Register with mechanism name for convenience
        mech_key = model_info.mechanism.lower()
        if mech_key not in self.models:
            self.models[mech_key] = model_info
    
    def get_model(self, key: str) -> Optional[Type[GSMDef]]:
        """Get model class by key (case-insensitive)"""
        model_info = self.models.get(key.lower())
        return model_info.cls if model_info else None
    
    def get_model_info(self, key: str) -> Optional[ModelInfo]:
        """Get model info by key (case-insensitive)"""
        return self.models.get(key.lower())
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models (unique by name)"""
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
    
    def get_available_keys(self) -> List[str]:
        """Get all available model keys"""
        return sorted(self.models.keys())
    
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
        
        # Add status column for mock vs real models
        status_width = max(len("Status"), max(len("Mock" if m.is_mock else "Real") for m in models))
        name_width = max(len("Model Name"), max(len(m.name) for m in models))
        mech_width = max(len("Mechanism"), max(len(m.mechanism) for m in models))
        desc_width = max(len("Description"), max(len(m.description) for m in models))
        
        # Format table
        header = f"{'Model Name':<{name_width}} | {'Mechanism':<{mech_width}} | {'Status':<{status_width}} | {'Description':<{desc_width}}"
        separator = "-" * len(header)
        
        lines = [header, separator]
        for model in models:
            status = "Mock" if model.is_mock else "Real"
            line = f"{model.name:<{name_width}} | {model.mechanism:<{mech_width}} | {status:<{status_width}} | {model.description:<{desc_width}}"
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
