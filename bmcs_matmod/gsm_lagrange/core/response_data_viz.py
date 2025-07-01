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
        
    def create_eps_sig_dashboard(self, figsize=(12, 5)):
        """
        Create a dashboard with two plots:
        - sig_t vs eps_t (stress-strain curve)
        - sig_t and eps_t as time series (with twin y-axes)
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        
        Returns
        -------
        tuple
            (fig, axes) - Figure and axes array for plotting
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('sig-eps Dashboard', fontsize=16)

        # First axis: sig_t vs eps_t
        axes[0].set_xlabel('Strain')
        axes[0].set_ylabel('Stress')
        axes[0].set_title('Stress-Strain Curve')
        axes[0].grid(True, alpha=0.3)

        # Second axis: time series with twin y-axis
        axes[1].set_xlabel('Time')
        axes[1].set_title('Stress/Strain Time Series')
        axes[1].grid(True, alpha=0.3)
        ax2 = axes[1].twinx()
        ax2.set_ylabel('Strain')
        axes[1].set_ylabel('Stress')

        return fig, (axes[0], axes[1], ax2)

    def fill_eps_sig_dashboard(self, axes, index=0, **kwargs):
        """
        Fill the sig-eps dashboard with data.

        Parameters
        ----------
        axes : tuple
            (ax_sig_eps, ax_time_series, ax_time_series_twin)
        index : int, optional
            Component index for 2D arrays (default: 0)
        **kwargs
            Additional arguments passed to plot functions
        """
        ax_sig_eps, ax_time, ax_time_twin = axes

        # Prepare data for plotting
        eps = self.eps_t
        sig = self.sig_t
        t = self.t_t

        # Handle 1D or 2D arrays
        if eps.ndim == 1:
            eps_plot = eps
        else:
            eps_plot = eps[:, index]
        if sig.ndim == 1:
            sig_plot = sig
        elif sig.ndim == 2:
            sig_plot = sig[:, index]
        else:
            sig_plot = sig[:, index, index]  # e.g., for 3D stress tensor

        # Plot sig vs eps
        ax_sig_eps.plot(eps_plot, sig_plot, label='sig vs eps', **kwargs)
        ax_sig_eps.legend()

        # Plot time series: stress (left), strain (right)
        ax_time.plot(t, sig_plot, label='Stress', color='tab:blue', **kwargs)
        ax_time_twin.plot(t, eps_plot, label='Strain', color='tab:orange', **kwargs)
        ax_time.legend(loc='upper left')
        ax_time_twin.legend(loc='upper right')
    
    def create_Eps_Sig_dashboard(self, figsize=(10, 3)):
        """
        Create a dashboard for Eps-Sig visualization.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        tuple
            (fig, axes) - Figure and axes array for plotting
        """
        num_pairs = len(self.Eps_t)
        fig_width, fig_height = figsize
        fig, axes = plt.subplots(num_pairs, 2, figsize=(fig_width, fig_height * num_pairs), squeeze=False)
        fig.suptitle('Eps-Sig Dashboard', fontsize=16)

        named_axes = []
        for i, (Eps_codename, Sig_codename) in enumerate(zip(self.Eps_codenames, self.Sig_codenames)):
            ax_Sig_Eps = axes[i, 0]
            ax_Eps_t = axes[i, 1]
            ax_Sig_t = ax_Eps_t.twinx()
            ax_Sig_Eps.set_xlabel(f'{Eps_codename}')
            ax_Sig_Eps.set_ylabel(f'{Sig_codename}')
            ax_Sig_Eps.set_title(f'{Sig_codename}-{Eps_codename}')
            ax_Sig_Eps.grid(True, alpha=0.3)
            ax_Eps_t.set_xlabel('Time')
            ax_Eps_t.set_title(f'Eps_{Eps_codename} Time Series')
            ax_Eps_t.grid(True, alpha=0.3)
            ax_Sig_t.set_ylabel(f'Sig_{Sig_codename}')
            named_axes.append( (Eps_codename, Sig_codename, 
                                ax_Sig_Eps, ax_Eps_t, ax_Sig_t) )

        return fig, named_axes

    def fill_Eps_Sig_dashboard(self, named_axes, **kwargs):
        """
        Fill the Eps-Sig dashboard with data.

        Parameters
        ----------
        axes : array-like
            Existing axes array
        **kwargs
            Additional arguments passed to plot functions
        """

        for Eps_codename, Sig_codename, ax_Sig_Eps, ax_Eps_t, ax_Sig_t in named_axes:
            Eps_data = self.Eps_t[Eps_codename][:, 0, 0]
            Sig_data = self.Sig_t[Sig_codename][:, 0, 0]
            ax_Sig_Eps.plot(Eps_data, Sig_data, label=f'{Sig_codename} vs {Eps_codename}', **kwargs)
            ax_Sig_Eps.legend()

            ax_Eps_t.plot(self.t_t, Eps_data, label=f'{Eps_codename}', **kwargs)
            ax_Sig_t.plot(self.t_t, Sig_data, label=f'{Sig_codename}', **kwargs)
            ax_Eps_t.legend()

    def create_energy_dashboard(self, figsize=(10, 6)):
        """
        Create a dashboard for energy visualization.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        tuple
            (fig, ax) - Figure and axis for energy
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('Energy Dashboard', fontsize=16)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution')
        ax.grid(True, alpha=0.3)

        return fig, ax

    def fill_energy_dashboard(self, ax, **kwargs):
        """
        Fill the energy dashboard with data.

        Parameters
        ----------
        ax : matplotlib.Axes
            Existing axis
        **kwargs
            Additional arguments passed to plot functions
        """
        if hasattr(self, 'energy_t'):
            ax.plot(self.t_t, self.energy_t, label='Energy', **kwargs)
            ax.legend()

    def create_metadata_dashboard(self, figsize=(6, 4)):
        """
        Create a dashboard for metadata visualization.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)

        Returns
        -------
        tuple
            (fig, ax) - Figure and axis for metadata
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('Metadata Dashboard', fontsize=16)
        ax.axis('off')

        return fig, ax

    def fill_metadata_dashboard(self, ax):
        """
        Fill the metadata dashboard with data.

        Parameters
        ----------
        ax : matplotlib.Axes
            Existing axis
        """
        stats = self.get_summary_stats()
        metadata_text = f"""
Simulation Metadata:
- Steps: {stats['simulation_info']['n_steps']}
- Duration: {stats['simulation_info']['duration']:.3f}
- Time Range: {stats['simulation_info']['time_range']}

Internal Variables:
{', '.join(stats['variables']['internal_variables'])}

Thermodynamic Forces:
{', '.join(stats['variables']['thermodynamic_forces'])}
"""
        ax.text(0.1, 0.9, metadata_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace')
