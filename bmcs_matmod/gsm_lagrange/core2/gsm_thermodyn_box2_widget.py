"""
GSM Thermodynamic Box 2 Widget - Interactive IPywidgets Implementation

This module provides an interactive ipywidgets-based visualization for the thermodynamic
square with clickable buttons in a 3x3 grid layout. This approach focuses on didactic
value for teaching thermodynamics using the interface-based GSMThermodynBox2.

Key Features:
- 3x3 grid of clickable ipywidget buttons
- Simple text labels (no complex LaTeX rendering)
- Interactive: clicking buttons displays mathematical expressions below
- Educational focus: step-by-step exploration of thermodynamic relationships
- Uses IPython.display for high-quality mathematical rendering
- Visual indication: Initial state function button has darker background color
- Interface-based: Works with GSMThermodynBox2 and GSMStateFnIfc implementations

3x3 Grid Layout:
    ε   | F  | T
    U   | ☐  | G  
   -S   | H  |-σ

Where ☐ represents a center information button.

Visual Indicators:
- Blue buttons: Variables (ε, T, -S, -σ)
- Light pink buttons: State functions (F, G, U, H)
- Dark pink button: Initial state function (the starting point for transformations)
- Green button: Information center
"""

import ipywidgets as widgets
from IPython.display import display, Math, clear_output
import sympy as sp
from typing import Dict, Optional
from .gsm_thermodyn_box2 import GSMThermodynBox2
from .gsm_state_fn import StateFunction


class GSMThermodynBox2Widget:
    """
    Interactive 3x3 grid widget for thermodynamic square visualization using ipywidgets.
    
    This widget creates a 3x3 grid of clickable buttons representing variables and state
    functions. Clicking a button displays the corresponding mathematical expression below
    the grid using IPython's high-quality LaTeX rendering.
    
    Works with GSMThermodynBox2 which uses interface-based state function implementations.
    """
    
    def __init__(self, gsm_box: GSMThermodynBox2, title: str = "Interactive Thermodynamic Square"):
        """
        Initialize the interactive 3x3 grid widget.
        
        Args:
            gsm_box: GSMThermodynBox2 instance containing interface-based state functions
            title: Title for the widget
        """
        self.gsm_box = gsm_box
        self.title = title
        
        # Track the last clicked button for visual feedback
        self.last_clicked_button = None
        
        # Create output widget for displaying expressions
        self.output_area = widgets.Output()
        
        # Define the 3x3 grid layout (rotated 90° counter-clockwise as in widget2)
        self.grid_layout = [
            ['ε', 'F', 'T'],      # Row 0: ε | F | T
            ['U', 'Info', 'G'],   # Row 1: U | ☐ | G  
            ['-S', 'H', '-σ']     # Row 2: -S | H | -σ
        ]
        
        # Create the button grid
        self.buttons = {}
        self.grid_widget = self._create_button_grid()
        
        # Create the main container
        self.container = self._create_container()
    
    def _create_button_grid(self):
        """Create the 3x3 grid of buttons."""
        rows = []
        
        # Get the initial state function name for highlighting
        initial_state_name = self.gsm_box.current_state_fn.value
        
        for row_idx, row in enumerate(self.grid_layout):
            button_row = []
            for col_idx, label in enumerate(row):
                button = widgets.Button(
                    description=label,
                    layout=widgets.Layout(
                        width='80px',
                        height='60px',
                        border='2px solid #333'
                    ),
                    style=widgets.ButtonStyle(
                        font_size='16px',
                        font_weight='bold',
                        text_color='black'  # Ensure all buttons start with black text
                    )
                )
                
                # Set button colors based on type
                if label in ['ε', 'T', '-S', '-σ']:  # Variables
                    button.style.button_color = 'lightblue'
                elif label in ['F', 'G', 'U', 'H']:  # State functions
                    # Check if this is the initial state function - use more subtle highlighting
                    if label == initial_state_name:
                        # Slightly darker pink for the initial state function
                        button.style.button_color = '#FFB6C1'  # Light pink (more subtle than before)
                        button.layout.border = '2px solid #FF69B4'  # Slightly pink border
                    else:
                        button.style.button_color = 'lightpink'
                else:  # Info button
                    button.style.button_color = 'lightgreen'
                
                # Set click handler
                button.on_click(lambda b, lbl=label: self._on_button_click(lbl))
                
                self.buttons[label] = button
                button_row.append(button)
            
            rows.append(widgets.HBox(button_row, layout=widgets.Layout(justify_content='center')))
        
        return widgets.VBox(rows, layout=widgets.Layout(align_items='center'))
    
    def _create_container(self):
        """Create the main container with title, grid, and output area."""
        title_widget = widgets.HTML(
            value=f"<h3 style='text-align: center; margin: 10px;'>{self.title}</h3>"
        )
        
        return widgets.VBox([
            title_widget,
            self.grid_widget,
            widgets.HTML("<hr style='margin: 20px 0;'>"),
            widgets.HTML("<h4 style='text-align: center;'>Mathematical Expression:</h4>"),
            self.output_area
        ], layout=widgets.Layout(
            border='2px solid #666',
            padding='20px',
            margin='10px',
            align_items='center'
        ))
    
    def _update_button_styles(self, clicked_label: str):
        """Update button styles to show the clicked button and initial state function."""
        initial_state_name = self.gsm_box.current_state_fn.value
        
        for label, button in self.buttons.items():
            if label == clicked_label:
                # Clicked button: inverted style (dark background, white text)
                if label in ['ε', 'T', '-S', '-σ']:  # Variables
                    button.style.button_color = '#4682B4'  # Dark steel blue
                elif label in ['F', 'G', 'U', 'H']:  # State functions
                    button.style.button_color = '#C71585'  # Dark magenta
                else:  # Info button
                    button.style.button_color = '#228B22'  # Forest green
                # Set white text for selected button
                button.style.text_color = 'white'
                button.layout.border = '3px solid #000'
            else:
                # Reset to default colors
                if label in ['ε', 'T', '-S', '-σ']:  # Variables
                    button.style.button_color = 'lightblue'
                elif label in ['F', 'G', 'U', 'H']:  # State functions
                    if label == initial_state_name:
                        # Slightly darker pink for the initial state function
                        button.style.button_color = '#FFB6C1'  # Light pink (subtle highlighting)
                        button.layout.border = '2px solid #FF69B4'  # Slightly pink border
                    else:
                        button.style.button_color = 'lightpink'
                        button.layout.border = '2px solid #333'
                else:  # Info button
                    button.style.button_color = 'lightgreen'
                    button.layout.border = '2px solid #333'
                # Set black text for non-selected buttons
                button.style.text_color = 'black'
    
    def _on_button_click(self, label: str):
        """Handle button click events."""
        # Update visual feedback for clicked button
        self._update_button_styles(label)
        self.last_clicked_button = label
        
        with self.output_area:
            clear_output(wait=True)
            
            if label == 'Info':
                self._show_info()
            elif label in ['F', 'G', 'U', 'H']:
                self._show_state_function(label)
            elif label in ['ε', 'T', '-S', '-σ']:
                self._show_variable(label)
            else:
                display(Math(f"\\text{{Unknown: }} {label}"))
    
    def _show_state_function(self, func_name: str):
        """Display a state function expression."""

        print('XXXXXXXXXXXXXXXXXXXX')
        try:
            # Direct access - let the property handle everything
            state_fn_instance = getattr(self.gsm_box, func_name)
            
            # Display the expression
            expr = state_fn_instance.fn_expr
            latex_expr = sp.latex(expr)
            display(Math(f"{func_name} = {latex_expr}"))
            
            # # Add description
            # descriptions = {'F': 'Helmholtz Free Energy', 'G': 'Gibbs Free Energy', 
            #               'U': 'Internal Energy', 'H': 'Enthalpy'}
            # display(widgets.HTML(f"<p style='text-align: center; font-style: italic;'>{descriptions[func_name]}</p>"))
            
        except Exception as e:
            display(widgets.HTML(f"<p style='color: red;'>Error displaying {func_name}: {str(e)}</p>"))
    
    def _show_variable(self, var_name: str):
        """Display information about a variable."""
        var_info = {
            'ε': ('\\varepsilon', 'Strain', 'Mechanical deformation variable'),
            'T': ('T', 'Temperature', 'Thermal intensive variable'),
            '-S': ('-S', 'Negative Entropy', 'Thermal extensive variable (conjugate to T)'),
            '-σ': ('-\\sigma', 'Negative Stress', 'Mechanical intensive variable (conjugate to ε)')
        }
        
        if var_name in var_info:
            symbol, name, description = var_info[var_name]
            display(Math(f"\\text{{Variable: }} {symbol}"))
            display(widgets.HTML(f"<p style='text-align: center;'><strong>{name}</strong></p>"))
            display(widgets.HTML(f"<p style='text-align: center; font-style: italic;'>{description}</p>"))
    
    def _show_info(self):
        """Display general information about the thermodynamic square."""
        initial_state_name = self.gsm_box.current_state_fn.value
        info_text = f"""
        <div style='text-align: center;'>
        <h4>Thermodynamic Square</h4>
        <p>This 3×3 grid represents the fundamental thermodynamic relationships.</p>
        <ul style='text-align: left; display: inline-block;'>
        <li><strong>Blue buttons:</strong> Thermodynamic variables (ε, T, -S, -σ)</li>
        <li><strong>Light pink buttons:</strong> State functions (F, G, U, H)</li>
        <li><strong style='color: #FF69B4;'>Slightly darker button ({initial_state_name}):</strong> Initial state function</li>
        <li><strong style='color: #C71585;'>Dark inverted button:</strong> Currently selected</li>
        <li><strong>Green button:</strong> Information about the thermodynamic square</li>
        <li><strong>Relationships:</strong> State functions derived via Legendre transformations</li>
        </ul>
        <p><em>Click any button to explore the mathematical expressions!</em></p>
        </div>
        """
        display(widgets.HTML(info_text))
    
    def show(self):
        """Display the interactive widget."""
        display(self.container)
        
        # Display the initial state function by default instead of info
        initial_state_name = self.gsm_box.current_state_fn.value
        
        # Update visual styles first (without triggering click handler)
        self._update_button_styles(initial_state_name)
        self.last_clicked_button = initial_state_name
        
        # Then display the initial content directly without going through _on_button_click
        with self.output_area:
            clear_output(wait=True)
            self._show_state_function(initial_state_name)
    
    def get_expressions_summary(self) -> Dict:
        """Get a summary of all expressions in the GSM box."""
        summary = {'current_state': self.gsm_box.current_state_fn.value}
        
        # Get expressions from available state function instances
        for func_name, attr in [('F', 'F'), ('G', 'G'), ('U', 'U'), ('H', 'H')]:
            state_fn_instance = getattr(self.gsm_box, attr)
            if state_fn_instance is not None:
                summary[func_name] = str(state_fn_instance.fn_expr)
            else:
                summary[func_name] = 'Not computed'
                
        return summary


def create_interactive_widget(gsm_box: GSMThermodynBox2, 
                            title: str = "Interactive Thermodynamic Square") -> GSMThermodynBox2Widget:
    """
    Convenience function to create an interactive 3x3 grid widget.
    
    Args:
        gsm_box: GSMThermodynBox2 instance containing interface-based state functions
        title: Title for the widget
        
    Returns:
        GSMThermodynBox2Widget instance ready to display
    """
    return GSMThermodynBox2Widget(gsm_box, title)


def demo_interactive_widget():
    """Quick demonstration of the interactive widget."""
    print("Demo: Interactive Thermodynamic Square Widget")
    print("=" * 50)
    print("This widget creates an interactive 3x3 grid with clickable buttons.")
    print("Click any button to see high-quality mathematical expressions below.")
    print("Perfect for teaching thermodynamic relationships step by step!")
