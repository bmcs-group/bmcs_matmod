import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import traits.api as tr
import bmcs_utils.api as bu
from typing import Dict, List, Optional, Tuple, Any, Union
import sympy as sp

from ..gsm_lagrange.core.response_data import ResponseData

class ResponseDataWidget(bu.Model):
    """Interactive widget for visualizing ResponseData using ipywidgets.
    
    This widget allows for selecting variables (both primary and internal) 
    to plot on X and Y axes.
    """
    
    # The response data to be visualized
    response_data = tr.Instance(ResponseData)
    
    # Available variables for x and y axes
    x_var_options = tr.List()
    y_var_options = tr.List()
    
    # Currently selected variables
    x_var = tr.Str()
    y_var = tr.Str()
    
    # Figure properties
    fig = tr.Instance(Figure)
    ax = tr.Instance(Axes)
    
    # Widget components
    widget = tr.Instance(widgets.VBox)
    
    def setup_options(self):
        """Setup the available options for plotting variables"""
        options = []
        
        # Add primary variables
        options.append(('Time (t)', 't_t'))
        
        # Track specific options for default selection
        eps_option = None
        sig_option = None
        
        # Add strain and stress components (if they exist)
        if hasattr(self.response_data, 'eps_t') and self.response_data.eps_t is not None:
            for i in range(self.response_data.eps_t.shape[1]):
                option = (f'Strain ε[{i}]', f'eps_t_{i}')
                options.append(option)
                if i == 0:  # Track first strain component for default X-axis
                    eps_option = option[1]
                
        if hasattr(self.response_data, 'sig_t') and self.response_data.sig_t is not None:
            for i in range(self.response_data.sig_t.shape[1]):
                option = (f'Stress σ[{i}]', f'sig_t_{i}')
                options.append(option)
                if i == 0:  # Track first stress component for default Y-axis
                    sig_option = option[1]
        
        # Add internal variables from Eps_t
        if hasattr(self.response_data, 'Eps_t'):
            try:
                for key in self.response_data.Eps_t.keys():
                    try:
                        var_data = self.response_data.Eps_t[key]
                        if len(var_data.shape) == 1:  # Scalar variable
                            options.append((f'Eps: {key}', f'Eps_t_{key}'))
                        else:  # Vector or tensor variable
                            # For multi-dimensional data, flatten all but the first dimension
                            flat_shape = int(np.prod(var_data.shape[1:])) if var_data.shape[1:] else 1
                            if flat_shape > 0:  # Ensure we have valid dimensions
                                for i in range(flat_shape):
                                    # Add index notation according to tensor dimensions
                                    if len(var_data.shape) > 2:
                                        # Create multi-index notation like [0,0], [0,1], etc.
                                        indices = np.unravel_index(i, var_data.shape[1:])
                                        idx_str = f"[{','.join(map(str, indices))}]"
                                        options.append((f'Eps: {key}{idx_str}', f'Eps_t_{key}_{i}'))
                                    else:
                                        options.append((f'Eps: {key}[{i}]', f'Eps_t_{key}_{i}'))
                    except Exception as e:
                        # Skip this variable if it causes an error
                        print(f"Error processing Eps_t[{key}]: {e}")
                        continue
            except Exception as e:
                print(f"Error accessing Eps_t keys: {e}")
        
        # Add conjugate variables from Sig_t
        if hasattr(self.response_data, 'Sig_t'):
            try:
                for key in self.response_data.Sig_t.keys():
                    try:
                        var_data = self.response_data.Sig_t[key]
                        if len(var_data.shape) == 1:  # Scalar variable
                            options.append((f'Sig: {key}', f'Sig_t_{key}'))
                        else:  # Vector or tensor variable
                            # For multi-dimensional data, flatten all but the first dimension
                            flat_shape = int(np.prod(var_data.shape[1:])) if var_data.shape[1:] else 1
                            if flat_shape > 0:  # Ensure we have valid dimensions
                                for i in range(flat_shape):
                                    # Add index notation according to tensor dimensions
                                    if len(var_data.shape) > 2:
                                        # Create multi-index notation like [0,0], [0,1], etc.
                                        indices = np.unravel_index(i, var_data.shape[1:])
                                        idx_str = f"[{','.join(map(str, indices))}]"
                                        options.append((f'Sig: {key}{idx_str}', f'Sig_t_{key}_{i}'))
                                    else:
                                        options.append((f'Sig: {key}[{i}]', f'Sig_t_{key}_{i}'))
                    except Exception as e:
                        # Skip this variable if it causes an error
                        print(f"Error processing Sig_t[{key}]: {e}")
                        continue
            except Exception as e:
                print(f"Error accessing Sig_t keys: {e}")
        
        self.x_var_options = options
        self.y_var_options = options
        
        # Set default values for a typical stress-strain plot
        # Prefer strain on x-axis and stress on y-axis as defaults
        if eps_option and (not self.x_var or self.x_var == 't_t'):
            self.x_var = eps_option
            print(f"Setting default x_var to {self.x_var}")
        elif not self.x_var and len(options) > 0:
            self.x_var = options[0][1]
            print(f"Setting fallback x_var to {self.x_var}")
            
        if sig_option and (not self.y_var or self.y_var == 't_t'):
            self.y_var = sig_option
            print(f"Setting default y_var to {self.y_var}")
        elif not self.y_var and len(options) > 1:
            self.y_var = options[1][1]
            print(f"Setting fallback y_var to {self.y_var}")
            
        # Force plot update if data is available
        if hasattr(self, 'widget') and self.widget is not None:
            self.update_plot()
    
    def _get_var_data(self, var_key: str) -> np.ndarray:
        """Extract data based on the variable key"""
        if not hasattr(self, 'response_data') or self.response_data is None:
            print(f"Warning: No response_data available when getting {var_key}")
            return np.array([])
            
        print(f"Getting data for {var_key}")
        
        if var_key == 't_t':
            return self.response_data.t_t
        
        # Handle primary variables (strain and stress)
        if var_key.startswith('eps_t_'):
            try:
                idx = int(var_key.split('_')[-1])
                if hasattr(self.response_data, 'eps_t') and self.response_data.eps_t is not None:
                    if idx < self.response_data.eps_t.shape[1]:
                        data = self.response_data.eps_t[:, idx]
                        print(f"Retrieved eps_t[{idx}] with shape {data.shape}")
                        return data
                    else:
                        print(f"Index {idx} out of bounds for eps_t with shape {self.response_data.eps_t.shape}")
                else:
                    print("eps_t not available in response_data")
            except Exception as e:
                print(f"Error accessing eps_t: {e}")
            return np.array([])
        
        if var_key.startswith('sig_t_'):
            try:
                idx = int(var_key.split('_')[-1])
                if hasattr(self.response_data, 'sig_t') and self.response_data.sig_t is not None:
                    if idx < self.response_data.sig_t.shape[1]:
                        data = self.response_data.sig_t[:, idx, 0]
                        print(f"Retrieved sig_t[{idx}] with shape {data.shape}")
                        return data
                    else:
                        print(f"Index {idx} out of bounds for sig_t with shape {self.response_data.sig_t.shape}")
                else:
                    print("sig_t not available in response_data")
            except Exception as e:
                print(f"Error accessing sig_t: {e}")
            return np.array([])
        
        # Handle internal variables (Eps_t)
        if var_key.startswith('Eps_t_'):
            parts = var_key.split('_')
            if len(parts) < 3:
                print(f"Invalid Eps_t key format: {var_key}")
                return np.array([])
                
            key = parts[2]
            print(f"Extracting Eps_t key: {key}")
            
            try:
                if not hasattr(self.response_data, 'Eps_t'):
                    print("Eps_t not available in response_data")
                    return np.array([])
                    
                if key not in self.response_data.Eps_t.keys():
                    print(f"Key {key} not found in Eps_t. Available keys: {list(self.response_data.Eps_t.keys())}")
                    return np.array([])
                    
                data = self.response_data.Eps_t[key]
                print(f"Retrieved raw Eps_t[{key}] with shape {data.shape}")
                
                # Check if there's an index for vector/tensor vars
                if len(parts) > 3:
                    idx = int(parts[3])
                    print(f"Using index {idx} for Eps_t[{key}]")
                    if len(data.shape) > 1:
                        flat_data = data.reshape(data.shape[0], -1)
                        if idx < flat_data.shape[1]:
                            result = flat_data[:, idx]
                            print(f"Returning flattened Eps_t data with shape {result.shape}")
                            return result
                        else:
                            print(f"Index {idx} out of bounds for flattened Eps_t[{key}] with shape {flat_data.shape}")
                    else:
                        print(f"Cannot index 1D data Eps_t[{key}] with shape {data.shape}")
                else:
                    # Handle scalar value directly
                    if len(data.shape) == 1:  # Already flat
                        print(f"Returning 1D Eps_t[{key}] data with shape {data.shape}")
                        return data
                    # For multi-dimensional data with no index specified, return first element
                    result = data.reshape(data.shape[0], -1)[:, 0]
                    print(f"Returning first element of Eps_t[{key}] with shape {result.shape}")
                    return result
            except Exception as e:
                print(f"Error accessing Eps_t[{key}]: {e}")
                import traceback
                traceback.print_exc()
            return np.array([])
        
        # Handle conjugate variables (Sig_t)
        if var_key.startswith('Sig_t_'):
            parts = var_key.split('_')
            if len(parts) < 3:
                print(f"Invalid Sig_t key format: {var_key}")
                return np.array([])
                
            key = parts[2]
            print(f"Extracting Sig_t key: {key}")
            
            try:
                if not hasattr(self.response_data, 'Sig_t'):
                    print("Sig_t not available in response_data")
                    return np.array([])
                    
                if key not in self.response_data.Sig_t.keys():
                    print(f"Key {key} not found in Sig_t. Available keys: {list(self.response_data.Sig_t.keys())}")
                    return np.array([])
                    
                data = self.response_data.Sig_t[key]
                print(f"Retrieved raw Sig_t[{key}] with shape {data.shape}")
                
                # Check if there's an index for vector/tensor vars
                if len(parts) > 3:
                    idx = int(parts[3])
                    print(f"Using index {idx} for Sig_t[{key}]")
                    if len(data.shape) > 1:
                        flat_data = data.reshape(data.shape[0], -1)
                        if idx < flat_data.shape[1]:
                            result = flat_data[:, idx]
                            print(f"Returning flattened Sig_t data with shape {result.shape}")
                            return result
                        else:
                            print(f"Index {idx} out of bounds for flattened Sig_t[{key}] with shape {flat_data.shape}")
                    else:
                        print(f"Cannot index 1D data Sig_t[{key}] with shape {data.shape}")
                else:
                    # Handle scalar value directly
                    if len(data.shape) == 1:  # Already flat
                        print(f"Returning 1D Sig_t[{key}] data with shape {data.shape}")
                        return data
                    # For multi-dimensional data with no index specified, return first element
                    result = data.reshape(data.shape[0], -1)[:, 0]
                    print(f"Returning first element of Sig_t[{key}] with shape {result.shape}")
                    return result
            except Exception as e:
                print(f"Error accessing Sig_t[{key}]: {e}")
                import traceback
                traceback.print_exc()
            return np.array([])
        
        print(f"No handler found for variable key: {var_key}")
        return np.array([])
    
    def _get_var_label(self, var_key: str) -> str:
        """Get display label for variable"""
        for label, key in self.x_var_options:
            if key == var_key:
                return label
        return var_key
    
    def update_plot(self, *args):
        """Update the plot based on selected variables"""
        # Ensure we have a figure and axes
        if not hasattr(self, 'fig') or self.fig is None or not hasattr(self, 'ax') or self.ax is None:
            # If we don't have a figure yet, we can't plot
            print("No figure available - widget not initialized yet")
            return
            
        try:
            # Check if we have response data
            if not hasattr(self, 'response_data') or self.response_data is None:
                print("No response data available for plotting")
                self.ax.clear()
                self.ax.text(0.5, 0.5, 'No response data available', 
                             ha='center', va='center', transform=self.ax.transAxes)
                if hasattr(self.fig, 'canvas') and self.fig.canvas is not None:
                    self.fig.canvas.draw_idle()
                return
                
            # Clear previous plot
            self.ax.clear()
            
            # Debug information
            print(f"Attempting to plot {self.x_var} vs {self.y_var}")
            
            # Get the data for the selected variables
            x_data = self._get_var_data(self.x_var)
            y_data = self._get_var_data(self.y_var)
            
            # Debug data dimensions
            print(f"Got x_data shape: {x_data.shape if hasattr(x_data, 'shape') else 'no shape'}")
            print(f"Got y_data shape: {y_data.shape if hasattr(y_data, 'shape') else 'no shape'}")
            
            # Check if we have valid data
            if len(x_data) == 0 or len(y_data) == 0:
                print(f"Empty data array returned for x_var={self.x_var} or y_var={self.y_var}")
                self.ax.text(0.5, 0.5, 'No data available for selected variables', 
                            ha='center', va='center', transform=self.ax.transAxes)
                self.fig.canvas.draw_idle()
                return
                
            # Handle case where arrays have different lengths
            if len(x_data) != len(y_data):
                min_len = min(len(x_data), len(y_data))
                x_data = x_data[:min_len]
                y_data = y_data[:min_len]
                print(f"Warning: x and y data have different lengths. Plotting first {min_len} points.")
            
            # Get labels for the axes
            x_label = self._get_var_label(self.x_var)
            y_label = self._get_var_label(self.y_var)
            
            # Plot the data with verification
            print(f"Plotting {len(x_data)} points with labels: {x_label} vs {y_label}")
            line, = self.ax.plot(x_data, y_data, '-', linewidth=2)
            if not line.get_visible():
                print("Warning: Line was plotted but is not visible")
                
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)
            self.ax.grid(True)
            
            # Set title to help debug
            self.ax.set_title(f"{y_label} vs {x_label}")
            
            # Check if the axes have reasonable limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            print(f"Plot limits: x={xlim}, y={ylim}")
            
            # Force reasonable limits if auto-scaling failed
            if xlim[0] == xlim[1]:
                self.ax.set_xlim([min(x_data) - 0.1, max(x_data) + 0.1])
            if ylim[0] == ylim[1]:
                self.ax.set_ylim([min(y_data) - 0.1, max(y_data) + 0.1])
            
            # Add a grid and make sure it's visible
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Update the plot using the Output widget pattern
            try:
                # Access the output widget
                output_widget = self.widget.children[1]
                
                # Clear and redraw in the widget's output area
                with output_widget:
                    from IPython.display import clear_output
                    clear_output(wait=True)
                    plt.figure(self.fig.number)  # Activate our figure
                    
                    # Try the standard draw method too
                    if hasattr(self.fig, 'canvas') and self.fig.canvas is not None:
                        self.fig.canvas.draw_idle()
                    
                    # Show the figure in the output widget
                    plt.show(self.fig)
                    print("Plot updated using IPython display")
                    
            except Exception as canvas_err:
                print(f"Error updating canvas: {canvas_err}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error updating plot: {e}")
            import traceback
            traceback.print_exc()
            try:
                if self.ax is not None:
                    self.ax.clear()
                    self.ax.text(0.5, 0.5, f'Error plotting: {str(e)}', 
                                ha='center', va='center', transform=self.ax.transAxes)
                    if hasattr(self.fig, 'canvas') and self.fig.canvas is not None:
                        self.fig.canvas.draw_idle()
            except Exception as inner_e:
                print(f"Error handling plot error: {inner_e}")
    
    def _create_widget(self):
        """Create and configure the ipywidgets"""
        # Create a new Output widget for matplotlib
        from IPython.display import display as ipy_display
        canvas_widget = widgets.Output()
        
        # Create a new figure for this widget
        with canvas_widget:
            plt.ioff()  # Turn off interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            print("Created new figure and axes in _create_widget")
            plt.show(self.fig)  # This displays the figure initially
        
        # Make sure options are setup
        if not hasattr(self, 'x_var_options') or not self.x_var_options:
            print("Setting up options in _create_widget")
            self.setup_options()
        
        # Setup dropdown selectors
        # Convert options to list of tuples
        dropdown_options = [(label, key) for label, key in self.x_var_options]
        
        print(f"Available options: {[opt[0] for opt in dropdown_options]}")
        print(f"Current x_var: {self.x_var}, y_var: {self.y_var}")
        
        # Find index of default value in options
        x_default_idx = 0
        y_default_idx = min(1, len(dropdown_options) - 1)  # Default to second option or last if only one
        
        # Find the dropdown option that matches our current trait values
        x_found = False
        y_found = False
        for i, (_, key) in enumerate(dropdown_options):
            if key == self.x_var:
                x_default_idx = i
                x_found = True
                print(f"Found x_var {self.x_var} at index {i}")
            if key == self.y_var:
                y_default_idx = i
                y_found = True
                print(f"Found y_var {self.y_var} at index {i}")
        
        # If options didn't contain our trait values, update traits to match options
        if not x_found and dropdown_options:
            self.x_var = dropdown_options[x_default_idx][1]
            print(f"x_var not found in options, setting to {self.x_var}")
        
        if not y_found and dropdown_options:
            self.y_var = dropdown_options[y_default_idx][1]
            print(f"y_var not found in options, setting to {self.y_var}")
        
        # Create the dropdowns with our selected values
        x_dropdown = widgets.Dropdown(
            options=dropdown_options,
            value=self.x_var,  # Use the trait value directly
            description='X-axis:',
            style={'description_width': 'initial'}
        )
        
        y_dropdown = widgets.Dropdown(
            options=dropdown_options,
            value=self.y_var,  # Use the trait value directly
            description='Y-axis:',
            style={'description_width': 'initial'}
        )
        
        print(f"Created dropdowns with x_value={x_dropdown.value}, y_value={y_dropdown.value}")
        
        # Store references to dropdowns
        self.x_dropdown = x_dropdown
        self.y_dropdown = y_dropdown
        
        # Direct update method that bypasses trait notifications during setup
        def update_plot_direct(change=None):
            if hasattr(self, 'ax') and self.ax is not None:
                try:
                    # Get values directly from dropdowns
                    x_key = self.x_dropdown.value
                    y_key = self.y_dropdown.value
                    
                    print(f"Dropdown change: x={x_key}, y={y_key}")
                    
                    # Only update trait values if they've changed
                    if self.x_var != x_key:
                        self.x_var = x_key
                    if self.y_var != y_key:
                        self.y_var = y_key
                        
                    # Update the plot directly
                    self.update_plot()
                except Exception as e:
                    print(f"Error in direct update: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Connect dropdowns to direct update method
        x_dropdown.observe(update_plot_direct, names='value')
        y_dropdown.observe(update_plot_direct, names='value')
        
        # Layout
        selectors = widgets.HBox([x_dropdown, y_dropdown])
        
        self.widget = widgets.VBox([selectors, canvas_widget])
        
        # Initialize the plot
        try:
            print("Calling initial update_plot from _create_widget")
            self.update_plot()
        except Exception as e:
            print(f"Error initializing plot: {e}")
            import traceback
            traceback.print_exc()
        
        return self.widget
    
    def _response_data_changed(self):
        """Handle changes in the response_data"""
        # Setup the options for plotting
        self.setup_options()
        
        # If widget is already created, update it
        if hasattr(self, 'widget') and self.widget is not None:
            # Update dropdown options
            if hasattr(self, 'x_dropdown') and self.x_dropdown is not None:
                self.x_dropdown.options = [(label, key) for label, key in self.x_var_options]
                
            if hasattr(self, 'y_dropdown') and self.y_dropdown is not None:
                self.y_dropdown.options = [(label, key) for label, key in self.y_var_options]
                
            # Update plot with new data
            self.update_plot()
    
    def _x_var_changed(self):
        """Handle changes in x-axis variable selection"""
        # Delay plot updates until widget is fully initialized
        # This prevents errors when traits are set during initialization
        if hasattr(self, 'widget') and self.widget is not None:
            self.update_plot()
    
    def _y_var_changed(self):
        """Handle changes in y-axis variable selection"""
        # Delay plot updates until widget is fully initialized
        if hasattr(self, 'widget') and self.widget is not None:
            self.update_plot()
    
    def __init__(self, response_data=None, **traits):
        # Don't create the figure here - wait until _create_widget
        self.fig = None
        self.ax = None
        
        super().__init__(**traits)
        
        if response_data is not None:
            self.response_data = response_data
            self.setup_options()
    
    def get_widget(self):
        """Return the ipywidget for display"""
        if not hasattr(self, 'widget') or self.widget is None:
            return self._create_widget()
        return self.widget
