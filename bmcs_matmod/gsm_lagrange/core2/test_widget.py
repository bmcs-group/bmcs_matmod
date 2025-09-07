"""
Minimal test widget to debug duplicate rendering issue.

This creates the simplest possible case: one button, one output area,
and a click counter to see if the issue is in the widget system itself.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output


class TestWidget:
    """Minimal widget with single button and output area to test duplicate rendering."""
    
    def __init__(self):
        self.click_count = 0
        
        # Create output widget
        self.output_area = widgets.Output()
        
        # Create single test button
        self.test_button = widgets.Button(
            description='Test Click',
            layout=widgets.Layout(width='120px', height='40px'),
            style=widgets.ButtonStyle(button_color='lightblue')
        )
        
        # Set click handler
        self.test_button.on_click(self._on_button_click)
        
        # Create container
        self.container = widgets.VBox([
            widgets.HTML("<h3>Minimal Test Widget</h3>"),
            self.test_button,
            widgets.HTML("<hr>"),
            widgets.HTML("<h4>Output Area:</h4>"),
            self.output_area
        ])
    
    def _on_button_click(self, button):
        """Handle button click - increment counter and display."""
        self.click_count += 1
        
        print(f"DEBUG: _on_button_click called - count: {self.click_count}")
        
        with self.output_area:
            clear_output(wait=True)
            print(f"Button clicked {self.click_count} times")
            print(f"Click timestamp: {self.click_count}")
    
    def show(self):
        """Display the test widget."""
        display(self.container)
        
        # Initial display
        with self.output_area:
            clear_output(wait=True)
            print("Ready - click the button to test!")


def create_test_widget():
    """Create and return a test widget instance."""
    return TestWidget()
