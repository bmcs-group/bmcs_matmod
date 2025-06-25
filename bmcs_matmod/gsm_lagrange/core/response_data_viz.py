#!/usr/bin/env python3
"""
Shared Visualization Interface for GSM Response Data

This module provides a common interface for visualizing simulation results
from both ResponseData (active) and ResponseDataNode (persistent) objects.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ResponseDataVisualizationMixin:
    """
    Mixin class providing shared visualization methods for response data.
    
    This mixin can be used by both ResponseData and ResponseDataNode to provide
    consistent plotting and analysis capabilities regardless of the storage format.
    """
    
    def plot_stress_strain(self, ax=None, **kwargs):
        """
        Plot stress-strain curve.
        
        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments passed to plot()
            
        Returns
        -------
        matplotlib.Axes
            The axes object with the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available. Install matplotlib for plotting.")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Extract strain and stress data
        strain = self.eps_t[:, 0] if self.eps_t.ndim > 1 else self.eps_t
        stress = self.sig_t[:, 0, 0] if self.sig_t.ndim > 2 else self.sig_t
        
        ax.plot(strain, stress, **kwargs)
        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress')
        ax.set_title('Stress-Strain Response')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_time_series(self, variables=None, ax=None, **kwargs):
        """
        Plot time series of variables.
        
        Parameters
        ----------
        variables : list of str, optional
            Variable codenames to plot. If None, plots strain and stress.
        ax : matplotlib.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments passed to plot()
            
        Returns
        -------
        matplotlib.Axes
            The axes object with the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available. Install matplotlib for plotting.")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        time = self.t_t
        
        if variables is None:
            # Default: plot strain and stress
            strain = self.eps_t[:, 0] if self.eps_t.ndim > 1 else self.eps_t
            stress = self.sig_t[:, 0, 0] if self.sig_t.ndim > 2 else self.sig_t
            
            ax.plot(time, strain, label='Strain', **kwargs)
            ax2 = ax.twinx()
            ax2.plot(time, stress, 'r-', label='Stress', **kwargs)
            ax2.set_ylabel('Stress')
            
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        else:
            # Plot specified variables
            for var_name in variables:
                if var_name in self.Eps_t:
                    data = self.Eps_t[var_name]
                    if data.ndim == 1:
                        ax.plot(time, data, label=f'eps_{var_name}', **kwargs)
                    else:
                        # Plot first component for multi-dimensional variables
                        ax.plot(time, data[:, 0], label=f'eps_{var_name}[0]', **kwargs)
                        
                if var_name in self.Sig_t:
                    data = self.Sig_t[var_name]
                    if data.ndim == 1:
                        ax.plot(time, data, label=f'sig_{var_name}', **kwargs)
                    else:
                        # Plot first component for multi-dimensional variables
                        ax.plot(time, data[:, 0], label=f'sig_{var_name}[0]', **kwargs)
            
            ax.legend()
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Variable Value')
        ax.set_title('Time Series')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_internal_variables(self, ax=None, **kwargs):
        """
        Plot all internal variables over time.
        
        Parameters
        ----------
        ax : matplotlib.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments passed to plot()
            
        Returns
        -------
        matplotlib.Axes
            The axes object with the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available. Install matplotlib for plotting.")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        time = self.t_t
        
        for var_name, var_data in self.Eps_t.items():
            if isinstance(var_data, np.ndarray):
                if var_data.ndim == 1:
                    ax.plot(time, var_data, label=var_name, **kwargs)
                else:
                    # Plot first component for multi-dimensional variables
                    ax.plot(time, var_data[:, 0], label=f'{var_name}[0]', **kwargs)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Internal Variable Value')
        ax.set_title('Internal Variables Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the simulation.
        
        Returns
        -------
        dict
            Dictionary with summary statistics
        """
        strain = self.eps_t[:, 0] if self.eps_t.ndim > 1 else self.eps_t
        stress = self.sig_t[:, 0, 0] if self.sig_t.ndim > 2 else self.sig_t
        
        stats = {
            'simulation_info': {
                'n_steps': len(self.t_t),
                'time_range': [float(self.t_t[0]), float(self.t_t[-1])],
                'duration': float(self.t_t[-1] - self.t_t[0])
            },
            'strain_stats': {
                'min': float(np.min(strain)),
                'max': float(np.max(strain)),
                'final': float(strain[-1]),
                'range': float(np.max(strain) - np.min(strain))
            },
            'stress_stats': {
                'min': float(np.min(stress)),
                'max': float(np.max(stress)),
                'final': float(stress[-1]),
                'range': float(np.max(stress) - np.min(stress))
            },
            'variables': {
                'internal_variables': list(self.Eps_t.keys()) if hasattr(self.Eps_t, 'keys') else [],
                'thermodynamic_forces': list(self.Sig_t.keys()) if hasattr(self.Sig_t, 'keys') else []
            }
        }
        
        # Add final values of internal variables
        final_internal = {}
        for var_name, var_data in self.Eps_t.items():
            if isinstance(var_data, np.ndarray):
                if var_data.ndim == 1:
                    final_internal[var_name] = float(var_data[-1])
                else:
                    final_internal[f'{var_name}_components'] = var_data[-1].flatten().tolist()
        stats['final_internal_variables'] = final_internal
        
        return stats
    
    def create_dashboard(self, figsize=(15, 10)):
        """
        Create a comprehensive dashboard with multiple plots.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
            
        Returns
        -------
        matplotlib.Figure
            Figure with dashboard plots
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available. Install matplotlib for plotting.")
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('GSM Simulation Dashboard', fontsize=16)
        
        # Stress-strain curve
        self.plot_stress_strain(ax=axes[0, 0])
        
        # Time series of strain and stress
        self.plot_time_series(ax=axes[0, 1])
        
        # Internal variables
        self.plot_internal_variables(ax=axes[1, 0])
        
        # Summary stats as text
        stats = self.get_summary_stats()
        axes[1, 1].axis('off')
        summary_text = f"""
Simulation Summary:
• Steps: {stats['simulation_info']['n_steps']}
• Duration: {stats['simulation_info']['duration']:.3f}
• Max strain: {stats['strain_stats']['max']:.6f}
• Max stress: {stats['stress_stats']['max']:.2f}
• Final strain: {stats['strain_stats']['final']:.6f}
• Final stress: {stats['stress_stats']['final']:.2f}

Internal Variables:
{chr(10).join(f'• {var}' for var in stats['variables']['internal_variables'])}

Thermodynamic Forces:
{chr(10).join(f'• {var}' for var in stats['variables']['thermodynamic_forces'])}
"""
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        return fig
