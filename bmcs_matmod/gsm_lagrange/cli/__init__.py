"""
CLI interface for GSM Lagrange framework.

This module provides command-line interface tools, data structures,
and utilities for working with GSM models in batch processing,
network interfaces, and integration with external frameworks like AiiDA.
"""

from .cli_gsm import main as cli_main
from .data_structures import (
    MaterialParameterData, 
    LoadingData, 
    SimulationConfig, 
    SimulationResults,
    create_monotonic_loading,
    create_cyclic_loading
)
from .parameter_loader import ParameterLoader

__all__ = [
    'cli_main',
    'MaterialParameterData',
    'LoadingData', 
    'SimulationConfig',
    'SimulationResults',
    'create_monotonic_loading',
    'create_cyclic_loading',
    'ParameterLoader'
]
