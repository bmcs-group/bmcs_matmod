"""
Minimal test widget to debug duplicate rendering issue.

This creates a simplified version that mimics gsm_thermodyn_box2_widget.py structure
but uses a GSMThermodynBox2 instance with four individual buttons (F, G, H, U)
that directly display their corresponding state functions. No cycling -
each button shows its own function when clicked.
"""

import ipywidgets as widgets
import sympy as sp
from IPython.display import display, clear_output, Math
from .gsm_thermodyn_box2 import GSMThermodynBox2
from .gsm_state_fn import StateFunction


class TestWidget:
    """
    Test widget that mimics GSMThermodynBox2Widget structure for debugging.
    
    Uses a GSMThermodynBox2 instance with four individual buttons (F, G, H, U)
    that directly display their corresponding state functions. No cycling -
    each button shows its own function.
    """
    
    def __init__(self, gsm_box: GSMThermodynBox2, title: str = "Test Widget - State Function Buttons"):
        """Initialize the test widget with GSMThermodynBox2 support."""
        self.gsm_box = gsm_box
        self.title = title
        self.click_count = 0
        
        # Define the state function buttons
        self.state_function_map = {
            'F': StateFunction.HELMHOLTZ,
            'G': StateFunction.GIBBS,
            'H': StateFunction.ENTHALPY,
            'U': StateFunction.INTERNAL_ENERGY
        }
        
        # Create output widget for displaying expressions
        self.output_area = widgets.Output()
        
        # Track the last clicked button for visual feedback
        self.last_clicked_button = None
        
        # Create button grid (four buttons in a row)
        self.buttons = {}
        self.grid_widget = self._create_button_grid()
        
        # Create the main container
        self.container = self._create_container()
    
    def _create_button_grid(self):
        """Create four buttons in a row for F, G, H, U."""
        button_row = []
        
        for func_name in ['F', 'G', 'H', 'U']:
            button = widgets.Button(
                description=func_name,
                layout=widgets.Layout(
                    width='80px',
                    height='60px',
                    border='2px solid #333'
                ),
                style=widgets.ButtonStyle(
                    font_size='16px',
                    font_weight='bold',
                    text_color='black',
                    button_color='lightpink'
                )
            )
            
            # Set click handler
            button.on_click(lambda b, name=func_name: self._on_button_click(name))
            
            self.buttons[func_name] = button
            button_row.append(button)
        
        # Return as VBox containing HBox of buttons
        return widgets.VBox([
            widgets.HBox(button_row, layout=widgets.Layout(justify_content='center'))
        ], layout=widgets.Layout(align_items='center'))
    
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
        """Update button styles to show the clicked button."""
        for label, button in self.buttons.items():
            if label == clicked_label:
                # Clicked button: dark background, white text
                button.style.button_color = '#C71585'  # Dark magenta
                button.style.text_color = 'white'
            else:
                # Reset to default style
                button.style.button_color = 'lightpink'
                button.style.text_color = 'black'
    
    def _on_button_click(self, label: str):
        """Handle button click events."""
        self.click_count += 1
        
        print(f"DEBUG: _on_button_click called - button: {label}, count: {self.click_count}")
        
        # Update visual feedback for clicked button
        self._update_button_styles(label)
        self.last_clicked_button = label
        
        with self.output_area:
            clear_output(wait=True)
            
            if label in ['F', 'G', 'H', 'U']:
                self._show_state_function(label)
            else:
                display(widgets.HTML(f"<p style='color: red;'>Unknown button: {label}</p>"))
    
    def _show_state_function(self, func_name: str):
        """Display the state function expression for the given function name."""
        try:
            print(f"DEBUG: Showing state function {func_name}")
            
            # Get state function instance from the GSM box using property access
            state_fn_instance = getattr(self.gsm_box, func_name)
            
            # Display the expression using LaTeX
            expr = state_fn_instance.fn_expr
            latex_expr = sp.latex(expr)
            display(Math(f"{func_name} = {latex_expr}"))
            
        except Exception as e:
            display(widgets.HTML(f"<p style='color: red;'>Error displaying {func_name}: {str(e)}</p>"))
    
    
    def show(self):
        """Display the test widget."""
        display(self.container)
        
        # Display F function by default
        self._update_button_styles('F')
        self.last_clicked_button = 'F'
        
        # Display initial content
        with self.output_area:
            clear_output(wait=True)
            self._show_state_function('F')


def create_test_widget(gsm_box: GSMThermodynBox2, 
                      title: str = "Test Widget - State Function Buttons") -> TestWidget:
    """Create and return a test widget instance."""
    return TestWidget(gsm_box, title)
