"""
AiiDA Plugin for GSM Lagrange Material Models

This package provides AiiDA integration for the bmcs_matmod.gsm_lagrange framework,
enabling persistent storage and provenance tracking of GSM simulation results.

Components:
- response_data_node: AiiDA DataNode for storing ResponseData with metadata
- (Future: calculations, workflows, parsers as needed)

Usage:
    from bmcs_matmod.gsm_lagrange.aiida_plugin.response_data_node import ResponseDataNode
"""

from .response_data_node import ResponseDataNode, create_response_data_node

__all__ = ['ResponseDataNode', 'create_response_data_node']
