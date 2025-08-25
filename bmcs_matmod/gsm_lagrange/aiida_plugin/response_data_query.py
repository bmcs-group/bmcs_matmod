"""
ResponseDataQuery - Interactive query interface for ResponseDataNode instances

This module provides a widget-based interface for querying ResponseDataNode instances
from the AiiDA database with various filtering criteria.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any
import aiida
from aiida.orm import QueryBuilder, Node


class ResponseDataQuery:
    """
    Interactive query interface for ResponseDataNode instances with ipywidgets.
    
    Provides a user-friendly widget interface for filtering and retrieving
    ResponseDataNode instances from the AiiDA database.
    """
    
    def __init__(self):
        """Initialize the query interface with widgets."""
        self.results = []
        self.results_df = None
        
        # Create widgets
        self._create_widgets()
        self._setup_layout()
        
    def _create_widgets(self):
        """Create all the interface widgets."""
        
        # Date selection widgets - Set defaults to cover a much wider range
        self.date_from_widget = widgets.DatePicker(
            description='From Date:',
            value=date.today() - timedelta(days=365),  # Default to last year
            style={'description_width': 'initial'}
        )
        
        self.date_to_widget = widgets.DatePicker(
            description='To Date:',
            value=date.today(),
            style={'description_width': 'initial'}
        )
        
        # Time selection (optional refinement)
        self.time_from_widget = widgets.TimePicker(
            description='From Time:',
            value=datetime.now().time().replace(hour=0, minute=0, second=0, microsecond=0),
            style={'description_width': 'initial'}
        )
        
        self.time_to_widget = widgets.TimePicker(
            description='To Time:',
            value=datetime.now().time().replace(hour=23, minute=59, second=59, microsecond=0),
            style={'description_width': 'initial'}
        )
        
        # Node type selection - Default to ResponseDataNode since that's what's in the database
        self.node_type_widget = widgets.Dropdown(
            options=[
                ('ResponseDataNode', 'ResponseDataNode'),
                ('ArrayData (with response signature)', 'ArrayData'),
                ('All ArrayData', 'AllArrayData')
            ],
            value='ResponseDataNode',  # Start with ResponseDataNode
            description='Node Type:',
            style={'description_width': 'initial'}
        )
        
        # Limit results
        self.limit_widget = widgets.IntSlider(
            value=50,
            min=1,
            max=1000,
            step=1,
            description='Max Results:',
            style={'description_width': 'initial'}
        )
        
        # Query button
        self.query_button = widgets.Button(
            description='Query Database',
            button_style='primary',
            icon='search'
        )
        
        # Clear button
        self.clear_button = widgets.Button(
            description='Clear Results',
            button_style='warning',
            icon='trash'
        )
        
        # Output area for results
        self.output_area = widgets.Output()
        
        # Progress indicator
        self.progress_widget = widgets.HTML(value="")
        
        # Connect button events
        self.query_button.on_click(self._on_query_clicked)
        self.clear_button.on_click(self._on_clear_clicked)
        
    def _setup_layout(self):
        """Setup the widget layout."""
        
        # Date/Time selection box
        date_time_box = widgets.VBox([
            widgets.HTML(value="<h3>📅 Time Range Selection</h3>"),
            widgets.HBox([self.date_from_widget, self.date_to_widget]),
            widgets.HBox([self.time_from_widget, self.time_to_widget])
        ])
        
        # Query options box
        options_box = widgets.VBox([
            widgets.HTML(value="<h3>⚙️ Query Options</h3>"),
            self.node_type_widget,
            self.limit_widget
        ])
        
        # Control buttons
        button_box = widgets.HBox([
            self.query_button,
            self.clear_button
        ])
        
        # Main interface
        self.interface = widgets.VBox([
            widgets.HTML(value="<h2>🔍 ResponseDataNode Query Interface</h2>"),
            widgets.HBox([date_time_box, options_box]),
            button_box,
            self.progress_widget,
            self.output_area
        ])
        
    def _on_query_clicked(self, button):
        """Handle query button click."""
        with self.output_area:
            clear_output(wait=True)
            
        self.progress_widget.value = "⏳ Querying database..."
        
        try:
            # Perform the query
            self._execute_query()
            self.progress_widget.value = f"✅ Found {len(self.results)} results"
            
        except Exception as e:
            self.progress_widget.value = f"❌ Query failed: {str(e)}"
            with self.output_area:
                print(f"Error: {e}")
                
    def _on_clear_clicked(self, button):
        """Handle clear button click."""
        self.results = []
        self.results_df = None
        self.progress_widget.value = ""
        
        with self.output_area:
            clear_output()
            
    def _execute_query(self):
        """Execute the database query based on widget values."""
        
        # Combine date and time widgets to create datetime objects
        from_datetime = datetime.combine(
            self.date_from_widget.value,
            self.time_from_widget.value
        )
        to_datetime = datetime.combine(
            self.date_to_widget.value,
            self.time_to_widget.value
        )
        
        # Convert to timezone-aware datetime for AiiDA
        from aiida.common import timezone
        from_datetime = timezone.make_aware(from_datetime)
        to_datetime = timezone.make_aware(to_datetime)
        
        # Build the query
        qb = QueryBuilder()
        
        if self.node_type_widget.value == 'ResponseDataNode':
            # Query specific ResponseDataNode type
            try:
                from bmcs_matmod.gsm_lagrange.aiida_plugin.response_data_node import ResponseDataNode
                qb.append(ResponseDataNode, filters={
                    'ctime': {'>=': from_datetime, '<=': to_datetime}
                }, tag='nodes')
            except ImportError:
                # Fallback to all nodes if ResponseDataNode not available
                qb.append(Node, filters={
                    'ctime': {'>=': from_datetime, '<=': to_datetime}
                }, tag='nodes')
                
        elif self.node_type_widget.value == 'ArrayData':
            # ArrayData with response signature
            from aiida.orm import ArrayData
            qb.append(ArrayData, filters={
                'ctime': {'>=': from_datetime, '<=': to_datetime},
                'attributes.array_names': {'contains': ['time']}
            }, tag='nodes')
            
        else:  # AllArrayData
            from aiida.orm import ArrayData
            qb.append(ArrayData, filters={
                'ctime': {'>=': from_datetime, '<=': to_datetime}
            }, tag='nodes')
        
        # Apply limit
        qb.limit(self.limit_widget.value)
        
        # Order by creation time (newest first)
        qb.order_by({'nodes': {'ctime': 'desc'}})
        
        # Execute query
        self.results = qb.all(flat=True)
        
        # Convert to DataFrame for display
        self._create_results_dataframe()
        
        # Display results
        self._display_results()
        
    def _create_results_dataframe(self):
        """Convert query results to a pandas DataFrame."""
        
        if not self.results:
            self.results_df = pd.DataFrame(columns=['Node PK', 'UUID', 'Created', 'Type', 'Arrays'])
            return
            
        data = []
        for node in self.results:
            try:
                # Get array names if available
                if hasattr(node, 'get_arraynames'):
                    arrays = ', '.join(node.get_arraynames())
                else:
                    arrays = 'N/A'
                
                data.append({
                    'Node PK': node.pk,
                    'UUID': str(node.uuid)[:8] + '...',  # Shortened UUID
                    'Created': node.ctime.strftime('%Y-%m-%d %H:%M:%S'),
                    'Type': node.__class__.__name__,
                    'Arrays': arrays
                })
            except Exception as e:
                data.append({
                    'Node PK': getattr(node, 'pk', 'Unknown'),
                    'UUID': 'Error',
                    'Created': 'Error',
                    'Type': 'Error',
                    'Arrays': f'Error: {e}'
                })
        
        self.results_df = pd.DataFrame(data)
        
    def _display_results(self):
        """Display the query results in the output area."""
        
        with self.output_area:
            clear_output(wait=True)
            
            if self.results_df.empty:
                display(HTML("<h3>No results found</h3>"))
                return
                
            # Display summary
            display(HTML(f"<h3>📊 Query Results ({len(self.results)} nodes)</h3>"))
            
            # Display the DataFrame as HTML table
            display(HTML(self.results_df.to_html(index=False, escape=False)))
            
            # Add some basic statistics
            node_types = self.results_df['Type'].value_counts()
            display(HTML("<h4>Node Type Distribution:</h4>"))
            display(HTML(node_types.to_frame().to_html()))
            
    def display(self):
        """Display the query interface."""
        display(self.interface)
        
    def get_results(self) -> List[Node]:
        """Get the current query results as a list of AiiDA nodes."""
        return self.results
        
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get the current query results as a pandas DataFrame."""
        return self.results_df if self.results_df is not None else pd.DataFrame()
        
    def export_results(self, filename: str = 'query_results.csv'):
        """Export results to CSV file."""
        if self.results_df is not None and not self.results_df.empty:
            self.results_df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
        else:
            print("No results to export")


def create_query_interface() -> ResponseDataQuery:
    """
    Factory function to create and return a ResponseDataQuery interface.
    
    Returns
    -------
    ResponseDataQuery
        Configured query interface ready for use
    """
    return ResponseDataQuery()
