#!/usr/bin/env python
"""
AiiDA ResponseDataNode Retrieval Script

This script demonstrates how to query and retrieve ResponseDataNode instances
from an AiiDA database that were created within the last week.

Usage:
    python aiida_response_data_retrieval.py
"""

import aiida
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np


def ensure_aiida_profile(profile_name: Optional[str] = None):
    """
    Ensure AiiDA profile is loaded.
    
    Parameters
    ----------
    profile_name : str, optional
        Specific profile name to load. If None, loads default profile.
    """
    try:
        from aiida.manage import get_manager
        manager = get_manager()
        current_profile = manager.get_profile()
        
        if profile_name and current_profile.name != profile_name:
            print(f"Switching from profile '{current_profile.name}' to '{profile_name}'")
            aiida.load_profile(profile_name)
        else:
            print(f"✓ AiiDA profile loaded: {current_profile.name}")
            
    except Exception:
        # No profile loaded, try to load one
        if profile_name:
            print(f"Loading AiiDA profile: {profile_name}")
            aiida.load_profile(profile_name)
        else:
            print("Loading default AiiDA profile")
            aiida.load_profile()


def query_response_data_nodes_last_week(profile_name: str = 'presto-1') -> List:
    """
    Query ResponseDataNode instances created in the last week.
    
    Parameters
    ----------
    profile_name : str
        AiiDA profile name to use
        
    Returns
    -------
    List[ResponseDataNode]
        List of ResponseDataNode instances from the last week
    """
    # Ensure correct profile is loaded
    ensure_aiida_profile(profile_name)
    
    from aiida.orm import QueryBuilder, Node
    from aiida.common import timezone
    
    # Calculate date range (last 7 days)
    now = timezone.now()
    week_ago = now - timedelta(days=7)
    
    print(f"Querying nodes created between {week_ago} and {now}")
    
    # Build query for ResponseDataNode instances
    qb = QueryBuilder()
    
    # Method 1: Query by node type (if ResponseDataNode is registered as a specific type)
    try:
        # Try to import ResponseDataNode to get its type
        from bmcs_matmod.gsm_lagrange.aiida_plugin.response_data_node import ResponseDataNode
        
        qb.append(ResponseDataNode, filters={
            'ctime': {'>=': week_ago, '<=': now}
        }, tag='response_nodes')
        
        print(f"Found {qb.count()} ResponseDataNode instances in the last week")
        
    except ImportError:
        # Fallback: Query ArrayData nodes with specific attributes
        print("ResponseDataNode not available, querying ArrayData with response characteristics")
        
        from aiida.orm import ArrayData
        qb = QueryBuilder()
        qb.append(ArrayData, filters={
            'ctime': {'>=': week_ago, '<=': now},
            'attributes.array_names': {'contains': ['time', 'eps', 'sig']}  # Response data signature
        }, tag='response_nodes')
        
        print(f"Found {qb.count()} ArrayData nodes with response characteristics in the last week")
    
    # Execute query and return results
    results = qb.all(flat=True)
    return results


def analyze_retrieved_nodes(nodes: List) -> None:
    """
    Analyze and display information about retrieved nodes.
    
    Parameters
    ----------
    nodes : List[Node]
        List of AiiDA nodes to analyze
    """
    if not nodes:
        print("No nodes found in the specified time period.")
        return
    
    print(f"\n=== Analysis of {len(nodes)} Retrieved Nodes ===")
    
    for i, node in enumerate(nodes, 1):
        print(f"\nNode {i}:")
        print(f"  UUID: {node.uuid}")
        print(f"  PK: {node.pk}")
        print(f"  Created: {node.ctime}")
        print(f"  Type: {node.__class__.__name__}")
        
        # Try to access array information
        try:
            array_names = node.get_arraynames()
            print(f"  Arrays: {array_names}")
            
            # Display array shapes if available
            for array_name in array_names:
                try:
                    array = node.get_array(array_name)
                    print(f"    {array_name}: shape {array.shape}, dtype {array.dtype}")
                except Exception as e:
                    print(f"    {array_name}: Error accessing array - {e}")
                    
        except AttributeError:
            print("  Arrays: Not an ArrayData node")
        
        # Display attributes
        try:
            attrs = node.attributes
            if attrs:
                print(f"  Attributes: {list(attrs.keys())}")
                # Show simulation metadata if available
                if 'simulation_metadata' in attrs:
                    metadata = attrs['simulation_metadata']
                    print(f"    Simulation metadata keys: {list(metadata.keys())}")
        except Exception:
            pass


def demonstrate_data_access(nodes: List) -> None:
    """
    Demonstrate how to access and use the retrieved ResponseDataNode data.
    
    Parameters
    ----------
    nodes : List[Node]
        List of retrieved nodes
    """
    if not nodes:
        return
    
    print(f"\n=== Data Access Demonstration ===")
    
    # Take the first node as example
    node = nodes[0]
    print(f"Using node {node.pk} as example:")
    
    try:
        # Access time series data
        if hasattr(node, 'get_array'):
            arrays = node.get_arraynames()
            
            if 'time' in arrays and 'sig' in arrays:
                time = node.get_array('time')
                stress = node.get_array('sig')
                
                print(f"  Time range: {time[0]:.3f} to {time[-1]:.3f}")
                print(f"  Stress range: {stress.min():.3f} to {stress.max():.3f}")
                print(f"  Data points: {len(time)}")
                
                # Simple analysis
                if len(stress) > 1:
                    max_stress_idx = np.argmax(np.abs(stress))
                    print(f"  Peak stress: {stress[max_stress_idx]:.3f} at time {time[max_stress_idx]:.3f}")
        
        # Show how to convert back to ResponseData if needed
        print("\n  Conversion example:")
        try:
            # This would be the actual method to convert back
            # response_data = node.to_response_data()
            print("  # response_data = node.to_response_data()")
            print("  # Available for further analysis and plotting")
        except Exception as e:
            print(f"  Conversion not available: {e}")
            
    except Exception as e:
        print(f"  Error accessing data: {e}")


def main():
    """Main function to demonstrate ResponseDataNode retrieval."""
    
    print("=== AiiDA ResponseDataNode Retrieval Demo ===")
    print("Retrieving ResponseDataNode instances from the last week...\n")
    
    try:
        # Query nodes from the last week
        nodes = query_response_data_nodes_last_week(profile_name='presto-1')
        
        # Analyze retrieved nodes
        analyze_retrieved_nodes(nodes)
        
        # Demonstrate data access
        demonstrate_data_access(nodes)
        
        print(f"\n=== Summary ===")
        print(f"Successfully retrieved {len(nodes)} ResponseDataNode instances")
        print("from the 'presto-1' profile created in the last week.")
        
        if nodes:
            print("\nThese nodes can be:")
            print("- Converted back to ResponseData for analysis")
            print("- Used for plotting and visualization")
            print("- Queried for specific simulation parameters")
            print("- Exported for sharing or archiving")
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        print("\nPossible issues:")
        print("- AiiDA profile 'presto-1' not found")
        print("- No ResponseDataNode instances in the database")
        print("- AiiDA not properly configured")
        
        # Provide troubleshooting info
        try:
            from aiida.manage import get_manager
            profile = get_manager().get_profile()
            print(f"Current profile: {profile.name}")
            
            # Show available profiles
            from aiida.manage.configuration import get_config
            config = get_config()
            available_profiles = list(config.profiles.keys())
            print(f"Available profiles: {available_profiles}")
            
        except Exception as config_error:
            print(f"Cannot access AiiDA configuration: {config_error}")


if __name__ == "__main__":
    main()
