"""
Disk-based Property Caching with Source File Dependency Tracking

This module provides a disk-based caching decorator for properties that automatically 
invalidates the cache when source files are modified. It's designed as a drop-in 
replacement for @cached_property with persistent storage capabilities.

Key Features:
- Automatic cache invalidation based on source file modification time
- Content hash verification for additional robustness  
- Graceful fallback to @cached_property when joblib is unavailable
- Configurable cache directory (default: ".gsm_cache")
- Thread-safe operations via joblib.Memory

Usage:
    from disk_cached_property import disk_cached_property
    
    class MyClass:
        @disk_cached_property(cache_dir=".my_cache")
        def expensive_computation(self):
            # Complex computation here
            return result

Dependencies:
    - joblib (optional): For disk caching functionality
    - functools: For fallback @cached_property
    - Standard library: os, inspect, hashlib
"""

import os
import inspect
import hashlib
from functools import cached_property

try:
    from joblib import Memory
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    Memory = None


def disk_cached_property(cache_dir=".cache", use_disk_cache=True, verbose=0):
    """
    Decorator that provides disk-based caching for properties with source file dependency tracking.
    
    This decorator caches property results to disk and automatically invalidates the cache
    when the source file containing the class definition is modified. Falls back to 
    regular @cached_property if joblib is not available or disk caching is disabled.
    
    Args:
        cache_dir (str): Directory to store cache files (default: ".cache")
        use_disk_cache (bool): Whether to use disk caching (default: True)
        verbose (int): Verbosity level for joblib.Memory (default: 0)
        
    Returns:
        Property decorator with disk caching capabilities
        
    Example:
        class ExpensiveModel:
            @disk_cached_property(cache_dir=".model_cache")
            def complex_calculation(self):
                # Expensive computation
                return expensive_result()
                
        model = ExpensiveModel()
        result = model.complex_calculation  # Computed and cached
        result = model.complex_calculation  # Loaded from cache
    """
    def decorator(func):
        if not _JOBLIB_AVAILABLE or not use_disk_cache:
            # Fallback to regular cached_property if joblib unavailable or disabled
            return cached_property(func)
            
        # Initialize joblib Memory with the cache directory
        memory = Memory(cache_dir, verbose=verbose)
        
        def wrapper(self):
            # Get source file information for cache invalidation
            try:
                source_file = inspect.getfile(self.__class__)
                source_mtime = os.path.getmtime(source_file)
                
                # Generate content hash for additional robustness
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_content = f.read()
                content_hash = hashlib.md5(source_content.encode()).hexdigest()[:8]
                
            except (OSError, TypeError) as e:
                # Fallback for cases where source file cannot be determined
                # (e.g., interactive sessions, dynamically created classes)
                if verbose > 0:
                    print(f"Warning: Could not determine source file for {self.__class__.__name__}: {e}")
                    print("Falling back to in-memory caching")
                return cached_property(func).__get__(self, type(self))
            
            # Create a cache key based on class, method, and source modification time
            class_name = self.__class__.__name__
            method_name = func.__name__
            cache_key = f"{class_name}_{method_name}_{source_mtime}_{content_hash}"
            
            # Create a cached version of the computation function
            @memory.cache
            def _compute_cached_property(cache_key_param, instance_id):
                """
                Cached computation function that computes the property value.
                
                Args:
                    cache_key_param (str): Cache invalidation key based on source file state
                    instance_id (int): Object instance identifier for debugging
                    
                Returns:
                    The computed property value
                """
                if verbose > 1:
                    print(f"Computing {class_name}.{method_name} (cache key: {cache_key_param[:20]}...)")
                
                return func(self)
            
            # Use object id as instance identifier
            instance_id = id(self)
            
            # Call the cached computation
            try:
                return _compute_cached_property(cache_key, instance_id)
            except Exception as e:
                if verbose > 0:
                    print(f"Warning: Disk caching failed for {class_name}.{method_name}: {e}")
                    print("Falling back to direct computation")
                # Fallback to direct computation if caching fails
                return func(self)
        
        return property(wrapper)
    return decorator


def clear_cache(cache_dir=".cache"):
    """
    Clear all cached files in the specified cache directory.
    
    Args:
        cache_dir (str): Cache directory to clear
        
    Returns:
        int: Number of files removed
    """
    if not os.path.exists(cache_dir):
        return 0
        
    removed_count = 0
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                removed_count += 1
            except OSError:
                pass  # Ignore files that can't be removed
    
    return removed_count


def get_cache_info(cache_dir=".cache"):
    """
    Get information about the cache directory.
    
    Args:
        cache_dir (str): Cache directory to analyze
        
    Returns:
        dict: Cache information including file count, total size, etc.
    """
    if not os.path.exists(cache_dir):
        return {
            "exists": False,
            "file_count": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0.0
        }
    
    file_count = 0
    total_size = 0
    
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
            except OSError:
                pass  # Ignore files that can't be accessed
    
    return {
        "exists": True,
        "file_count": file_count,
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "cache_dir": cache_dir
    }


# Module-level information
__version__ = "1.0.0"
__author__ = "GSM Framework"
__description__ = "Disk-based property caching with automatic invalidation"

# Export main functionality
__all__ = [
    "disk_cached_property",
    "clear_cache", 
    "get_cache_info",
    "_JOBLIB_AVAILABLE"
]
