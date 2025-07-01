#!/usr/bin/env python3
"""
AiiDA DataNode for GSM Response Data

This module provides a persistent, provenance-tracked storage solution for GSM simulation results.
It extends AiiDA's ArrayData to store ResponseData with metadata and provides a shared interface
for visualization and analysis.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
from ..core.response_data_viz import ResponseDataVisualizationMixin

if TYPE_CHECKING:
    from ..core.response_data import ResponseData

try:
    from aiida.orm import ArrayData
    from aiida.common import AttributeDict
    from aiida.manage import get_manager
    import aiida
    AIIDA_AVAILABLE = True
    
    # Check if profile is loaded on import
    try:
        get_manager().get_profile()
        AIIDA_PROFILE_READY = True
    except Exception:
        AIIDA_PROFILE_READY = False
        
except ImportError:
    # Fallback base class when AiiDA is not available
    class ArrayData:
        def __init__(self):
            raise ImportError("AiiDA not available. Install aiida-core to use ResponseDataNode.")
    AIIDA_AVAILABLE = False
    AIIDA_PROFILE_READY = False
    aiida = None


class ResponseDataNode(ArrayData, ResponseDataVisualizationMixin):
    """
    AiiDA DataNode for persistent storage of GSM simulation response data.
    
    This class provides:
    - Immutable, versioned storage of simulation results
    - Automatic provenance tracking
    - Database queryability
    - Shared interface with ResponseData for visualization
    - Efficient binary storage of large arrays
    
    The class maintains the same interface as ResponseData for seamless
    integration with plotting and analysis tools.
    """
    
    def __init__(self, **kwargs):
        """Initialize ResponseDataNode with profile checking"""
        if not AIIDA_AVAILABLE:
            raise ImportError("AiiDA not available. Install aiida-core to use ResponseDataNode.")
        
        # Check if profile is loaded, try to load if not
        if not AIIDA_PROFILE_READY:
            try:
                aiida.load_profile()
            except Exception as e:
                raise RuntimeError(
                    f"AiiDA profile not loaded and auto-load failed: {e}\n"
                    "Please load a profile with aiida.load_profile() before creating ResponseDataNode."
                )
        
        super().__init__(**kwargs)
    
    @classmethod
    def from_response_data(cls, response_data: 'ResponseData', 
                          simulation_metadata: Optional[Dict[str, Any]] = None) -> 'ResponseDataNode':
        """
        Create a ResponseDataNode from active ResponseData.
        
        Parameters
        ----------
        response_data : ResponseData
            The active simulation data to store
        simulation_metadata : dict, optional
            Additional metadata about the simulation (parameters, model info, etc.)
            
        Returns
        -------
        ResponseDataNode
            Persistent storage node with all simulation data and metadata
        """
        node = cls()
        
        # Store main time series arrays
        node.set_array('time', response_data.t_t)
        node.set_array('eps', response_data.eps_t)
        node.set_array('sig', response_data.sig_t)
        node.set_array('iterations', response_data.iter_t)
        node.set_array('lambda', response_data.lam_t)
        
        # Store flattened internal variables and forces
        node.set_array('Eps_t_flat', response_data.Eps_t_flat)
        node.set_array('Sig_t_flat', response_data.Sig_t_flat)
        
        # Store individual variables with codenames
        for var_name, var_data in response_data.Eps_t.items():
            if isinstance(var_data, np.ndarray):
                node.set_array(f'Eps_{var_name}', var_data)
        
        for var_name, var_data in response_data.Sig_t.items():
            if isinstance(var_data, np.ndarray):
                node.set_array(f'Sig_{var_name}', var_data)
        
        # Store metadata as queryable attributes
        node.base.attributes.set('simulation_info', {
            'n_steps': len(response_data.t_t),
            'time_range': [float(response_data.t_t[0]), float(response_data.t_t[-1])],
            'eps_range': [float(np.min(response_data.eps_t)), float(np.max(response_data.eps_t))],
            'sig_range': [float(np.min(response_data.sig_t)), float(np.max(response_data.sig_t))]
        })
        
        # Store variable metadata
        node.base.attributes.set('variable_info', {
            'Eps_vars': [getattr(var, 'codename', str(var)) for var in response_data.Eps_vars],
            'Sig_vars': [getattr(var, 'codename', str(var)) for var in response_data.Sig_vars],
            'Eps_codenames': response_data.Eps_codenames,
            'Sig_codenames': response_data.Sig_codenames,
            'Eps_var_shapes': [getattr(var, 'shape', ()) for var in response_data.Eps_vars],
            'Sig_var_shapes': [getattr(var, 'shape', ()) for var in response_data.Sig_vars],
            'Eps_var_names': [str(var) for var in response_data.Eps_vars],
            'Sig_var_names': [str(var) for var in response_data.Sig_vars]
        })
        
        # Store simulation metadata if provided
        if simulation_metadata:
            node.base.attributes.set('simulation_metadata', simulation_metadata)
        
        return node
    
    def to_response_data(self) -> 'ResponseData':
        """
        Convert stored data back to active ResponseData format.
        
        Returns
        -------
        ResponseData
            Active simulation data reconstructed from storage
        """
        from ..core.response_data import ResponseData, ResponseDataContainer
        
        # Get main arrays
        t_t = self.get_array('time')
        eps_t = self.get_array('eps')
        sig_t = self.get_array('sig')
        iter_t = self.get_array('iterations')
        lam_t = self.get_array('lambda')
        Eps_t_flat = self.get_array('Eps_t_flat')
        Sig_t_flat = self.get_array('Sig_t_flat')
        
        # Reconstruct variable metadata (simplified - would need proper Variable objects)
        var_info = self.base.attributes.get('variable_info')
        
        # Create mock variable objects with essential attributes
        class MockVariable:
            def __init__(self, codename, shape, name):
                self.codename = codename
                self.shape = shape
                self._name = name
            def __str__(self):
                return self._name
        
        Eps_vars = tuple(MockVariable(name, tuple(shape), var_name) 
                        for name, shape, var_name in zip(
                            var_info['Eps_vars'],
                            var_info['Eps_var_shapes'], 
                            var_info['Eps_var_names']))
        
        Sig_vars = tuple(MockVariable(name, tuple(shape), var_name)
                        for name, shape, var_name in zip(
                            var_info['Sig_vars'],
                            var_info['Sig_var_shapes'],
                            var_info['Sig_var_names']))
        
        # Reconstruct individual variable dictionaries
        Eps_t_dict = {}
        for var_name in var_info['Eps_vars']:
            try:
                Eps_t_dict[var_name] = self.get_array(f'Eps_{var_name}')
            except KeyError:
                # Fallback to reconstruction from flat arrays if individual arrays not found
                pass
        
        Sig_t_dict = {}
        for var_name in var_info['Sig_vars']:
            try:
                Sig_t_dict[var_name] = self.get_array(f'Sig_{var_name}')
            except KeyError:
                # Fallback to reconstruction from flat arrays if individual arrays not found
                pass
        
        # Create ResponseData instance
        return ResponseData(
            t_t=t_t,
            eps_t=eps_t,
            sig_t=sig_t,
            Eps_t_flat=Eps_t_flat,
            Sig_t_flat=Sig_t_flat,
            Eps_vars=Eps_vars,
            Sig_vars=Sig_vars,
            Eps_codenames=var_info['Eps_codenames'],
            Sig_codenames=var_info['Sig_codenames'],
            Eps_t=ResponseDataContainer(Eps_t_dict),
            Sig_t=ResponseDataContainer(Sig_t_dict),
            iter_t=iter_t,
            lam_t=lam_t
        )
    
    # Shared interface methods for visualization compatibility
    @property
    def t_t(self):
        """Time array"""
        return self.get_array('time')
    
    @property
    def eps_t(self):
        """Strain array"""
        return self.get_array('eps')
    
    @property
    def sig_t(self):
        """Stress array"""
        return self.get_array('sig')
    
    @property
    def Eps_t(self):
        """Internal variables dictionary-like access"""
        var_info = self.base.attributes.get('variable_info')
        result = {}
        for var_name in var_info['Eps_vars']:
            try:
                result[var_name] = self.get_array(f'Eps_{var_name}')
            except KeyError:
                pass
        return result
    
    @property
    def Sig_t(self):
        """Thermodynamic forces dictionary-like access"""
        var_info = self.base.attributes.get('variable_info')
        result = {}
        for var_name in var_info['Sig_vars']:
            try:
                result[var_name] = self.get_array(f'Sig_{var_name}')
            except KeyError:
                pass
        return result
    
    def get_Eps_array(self, key: str):
        """Get internal variable by codename (compatible with ResponseData)"""
        return self.get_array(f'Eps_{key}')
    
    def get_Sig_array(self, key: str):
        """Get conjugate variable by codename (compatible with ResponseData)"""
        return self.get_array(f'Sig_{key}')
    
    def get_simulation_info(self) -> Dict[str, Any]:
        """Get simulation summary information"""
        return self.base.attributes.get('simulation_info', {})
    
    def get_variable_info(self) -> Dict[str, Any]:
        """Get variable metadata information"""
        return self.base.attributes.get('variable_info', {})
    
    def get_simulation_metadata(self) -> Dict[str, Any]:
        """Get additional simulation metadata"""
        return self.base.attributes.get('simulation_metadata', {})
    
    def __repr__(self) -> str:
        """String representation"""
        try:
            info = self.get_simulation_info()
            var_info = self.get_variable_info()
            return (
                f"ResponseDataNode(uuid={self.uuid}, "
                f"n_steps={info.get('n_steps', 'unknown')}, "
                f"Eps_vars={var_info.get('Eps_vars', [])}, "
                f"Sig_vars={var_info.get('Sig_vars', [])})"
            )
        except:
            return f"ResponseDataNode(uuid={self.uuid})"
    
    def to_json_dict(self, include_arrays=True, include_metadata=True):
        """
        Convert ResponseDataNode to a JSON-serializable dictionary.
        
        Parameters
        ----------
        include_arrays : bool
            Whether to include full time series arrays
        include_metadata : bool
            Whether to include variable metadata
            
        Returns
        -------
        dict
            JSON-serializable dictionary with simulation results
        """
        result = {
            "status": "success",
            "simulation_info": {
                "n_steps": len(self.t_t),
                "time_range": [float(self.t_t[0]), float(self.t_t[-1])],
                "eps_range": [float(np.min(self.eps_t)), float(np.max(self.eps_t))],
                "sig_range": [float(np.min(self.sig_t)), float(np.max(self.sig_t))]
            }
        }
        
        if include_arrays:
            result["time_series"] = {
                "time": self.t_t.tolist(),
                "strain": self.eps_t[:, 0].tolist() if self.eps_t.ndim > 1 else self.eps_t.tolist(),
                "stress": self.sig_t[:, 0, 0].tolist() if self.sig_t.ndim > 2 else self.sig_t.tolist()
            }
            
            # Add internal variables using codenames
            result["internal_variables"] = {}
            for var_name, var_data in self.Eps_t.items():
                if isinstance(var_data, np.ndarray):
                    if var_data.ndim == 1:
                        result["internal_variables"][var_name] = var_data.tolist()
                    elif var_data.ndim == 2 and var_data.shape[1] == 1:
                        result["internal_variables"][var_name] = var_data[:, 0].tolist()
                    else:
                        # Flatten for higher dimensions
                        flattened = var_data.reshape(var_data.shape[0], -1)
                        if flattened.shape[1] == 1:
                            result["internal_variables"][var_name] = flattened[:, 0].tolist()
                        else:
                            result["internal_variables"][var_name] = flattened.tolist()
            
            # Add thermodynamic forces using codenames
            result["thermodynamic_forces"] = {}
            for var_name, var_data in self.Sig_t.items():
                if isinstance(var_data, np.ndarray):
                    if var_data.ndim == 1:
                        result["thermodynamic_forces"][var_name] = var_data.tolist()
                    elif var_data.ndim == 2 and var_data.shape[1] == 1:
                        result["thermodynamic_forces"][var_name] = var_data[:, 0].tolist()
                    else:
                        # Flatten for higher dimensions
                        flattened = var_data.reshape(var_data.shape[0], -1)
                        if flattened.shape[1] == 1:
                            result["thermodynamic_forces"][var_name] = flattened[:, 0].tolist()
                        else:
                            result["thermodynamic_forces"][var_name] = flattened.tolist()
        
        if include_metadata:
            var_info = self.get_variable_info()
            result["metadata"] = {
                "eps_variables": var_info.get('Eps_vars', []),
                "sig_variables": var_info.get('Sig_vars', []),
                "eps_var_shapes": var_info.get('Eps_var_shapes', []),
                "sig_var_shapes": var_info.get('Sig_var_shapes', [])
            }
        
        # Add AiiDA-specific metadata
        result["aiida_metadata"] = {
            "uuid": str(self.uuid),
            "node_type": "ResponseDataNode",
            "is_stored": self.is_stored,
            "creation_time": self.ctime.isoformat() if hasattr(self, 'ctime') and self.ctime else None
        }
        
        return result
    
    def to_json_string(self, include_arrays=True, include_metadata=True, indent=2):
        """
        Convert ResponseDataNode to a JSON string.
        
        Parameters
        ----------
        include_arrays : bool
            Whether to include full time series arrays
        include_metadata : bool
            Whether to include variable metadata
        indent : int
            JSON indentation level
            
        Returns
        -------
        str
            JSON string representation
        """
        import json
        data = self.to_json_dict(include_arrays=include_arrays, include_metadata=include_metadata)
        return json.dumps(data, indent=indent)
    
    def to_summary_dict(self):
        """
        Convert ResponseDataNode to a summary dictionary (no full arrays).
        
        Returns
        -------
        dict
            Summary dictionary with key simulation metrics
        """
        sim_info = self.get_simulation_info()
        result = {
            "status": "success",
            "simulation_summary": {
                "n_steps": sim_info.get('n_steps', len(self.t_t)),
                "time_range": sim_info.get('time_range', [float(self.t_t[0]), float(self.t_t[-1])]),
                "eps_range": sim_info.get('eps_range', [float(np.min(self.eps_t)), float(np.max(self.eps_t))]),
                "sig_range": sim_info.get('sig_range', [float(np.min(self.sig_t)), float(np.max(self.sig_t))]),
                "max_strain": float(np.max(self.eps_t)),
                "max_stress": float(np.max(self.sig_t)),
                "final_strain": float(self.eps_t[-1, 0]) if self.eps_t.ndim > 1 else float(self.eps_t[-1]),
                "final_stress": float(self.sig_t[-1, 0, 0]) if self.sig_t.ndim > 2 else float(self.sig_t[-1])
            },
            "available_variables": {
                "internal_variables": list(self.Eps_t.keys()),
                "thermodynamic_forces": list(self.Sig_t.keys())
            },
            "aiida_info": {
                "uuid": str(self.uuid),
                "is_stored": self.is_stored,
                "creation_time": self.ctime.isoformat() if hasattr(self, 'ctime') and self.ctime else None
            }
        }
        
        # Add final values of internal variables
        result["final_internal_variables"] = {}
        for var_name, var_data in self.Eps_t.items():
            if isinstance(var_data, np.ndarray):
                if var_data.ndim == 1:
                    result["final_internal_variables"][var_name] = float(var_data[-1])
                elif var_data.ndim >= 2:
                    final_val = var_data[-1].flatten()
                    if len(final_val) == 1:
                        result["final_internal_variables"][var_name] = float(final_val[0])
                    else:
                        result["final_internal_variables"][var_name] = final_val.tolist()
        
        return result
    
    def format_for_output(self, output_format='json', detailed=False):
        """
        Format ResponseDataNode for different output types.
        
        Parameters
        ----------
        output_format : str
            Output format: 'json', 'json_detailed', or 'node'
        detailed : bool
            Include full arrays (for JSON formats)
            
        Returns
        -------
        dict or ResponseDataNode
            Formatted data ready for output or transfer
        """
        if output_format == 'node':
            return self  # Return self for direct AiiDA usage
        elif output_format == 'json_detailed' or (output_format == 'json' and detailed):
            return self.to_json_dict(include_arrays=True, include_metadata=True)
        elif output_format == 'json':
            return self.to_summary_dict()
        else:
            raise ValueError(f"Unsupported output format: {output_format}. "
                           f"Supported: 'json', 'json_detailed', 'node'")

# Factory function for creating nodes
def create_response_data_node(response_data: 'ResponseData', 
                             simulation_metadata: Optional[Dict[str, Any]] = None,
                             store: bool = True,
                             auto_load_profile: bool = True) -> ResponseDataNode:
    """
    Factory function to create and optionally store a ResponseDataNode.
    
    Parameters
    ----------
    response_data : ResponseData
        The active simulation data
    simulation_metadata : dict, optional
        Additional simulation metadata
    store : bool, default True
        Whether to store the node in the database immediately
    auto_load_profile : bool, default True
        Whether to automatically load AiiDA profile if needed
        
    Returns
    -------
    ResponseDataNode
        Created (and optionally stored) node
        
    Raises
    ------
    ImportError
        If AiiDA is not available
    RuntimeError
        If AiiDA profile cannot be loaded
    """
    if not AIIDA_AVAILABLE:
        raise ImportError("AiiDA not available. Install aiida-core to create ResponseDataNode.")
    
    # Handle profile loading
    if not AIIDA_PROFILE_READY and auto_load_profile:
        try:
            aiida.load_profile()
        except Exception as e:
            raise RuntimeError(
                f"AiiDA profile not loaded and auto-load failed: {e}\n"
                "Please load a profile with aiida.load_profile() or set auto_load_profile=False."
            )
    elif not AIIDA_PROFILE_READY:
        raise RuntimeError(
            "AiiDA profile not loaded. Please load a profile with aiida.load_profile() "
            "or set auto_load_profile=True."
        )
        
    node = ResponseDataNode.from_response_data(response_data, simulation_metadata)
    
    if store:
        node.store()
    
    return node
