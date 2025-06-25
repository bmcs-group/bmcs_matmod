"""
CLI interface for GSM Lagrange framework.

This module provides command-line interface tools for executing
GSM simulations with real material models.
"""

from .cli_gsm import main as cli_main

__all__ = [
    'cli_main'
]
