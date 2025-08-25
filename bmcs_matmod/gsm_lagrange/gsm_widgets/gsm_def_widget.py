"""
GSM Definition Widget

Interactive widget for visualizing GSM model attributes using ipytree and ipywidgets.
Provides a tree-based view of model components with detailed attribute inspection.
"""

import ipywidgets as widgets
from ipytree import Tree, Node
from IPython.display import display, HTML
import sympy as sp
from typing import Any, Dict, List, Optional, Union
import inspect


class GSMDefWidget:
    """
    Interactive widget for exploring GSM Definition classes.
    
    Provides a tree-based interface on the left for browsing model attributes
    and a detail panel on the right for examining selected items.
    """
    
    def __init__(self, gsm_def_class):
        """
        Initialize the GSM Definition widget.
        
        Parameters
        ----------
        gsm_def_class : GSMDef subclass
            The GSM definition class to visualize
        """
        self.gsm_def_class = gsm_def_class
        self.selected_item = None
        self.selected_value = None
        
        # Create the widget interface
        self._create_widgets()
        self._populate_tree()
        self._setup_layout()
        
    def _create_widgets(self):
        """Create the main widget components."""
        
        # Title
        self.title = widgets.HTML(
            value=f"<h2>🔬 GSM Model Explorer: {self.gsm_def_class.__name__}</h2>"
        )
        
        # Left panel: Tree view
        self.tree = Tree()
        self.tree.layout.height = '500px'
        self.tree.layout.width = '400px'
        
        # Right panel: Detail view
        self.detail_output = widgets.Output()
        self.detail_output.layout.height = '500px'
        self.detail_output.layout.width = '600px'
        
        # Status bar
        self.status = widgets.HTML(value="Select an item from the tree to view details")
        
    def _populate_tree(self):
        """Populate the tree with GSM model attributes."""
        
        # Root node
        root = Node(f"📊 {self.gsm_def_class.__name__}")
        self.tree.add_node(root)
        
        # Engines section
        engines_node = Node("🔧 Engines")
        root.add_node(engines_node)
        
        # F_engine (Helmholtz)
        if hasattr(self.gsm_def_class, 'F_engine') and self.gsm_def_class.F_engine:
            f_engine_node = Node("⚡ F_engine (Helmholtz)")
            engines_node.add_node(f_engine_node)
            self._add_engine_attributes(f_engine_node, self.gsm_def_class.F_engine, 'F_engine')
            
        # G_engine (Gibbs)
        if hasattr(self.gsm_def_class, 'G_engine') and self.gsm_def_class.G_engine:
            g_engine_node = Node("⚡ G_engine (Gibbs)")
            engines_node.add_node(g_engine_node)
            self._add_engine_attributes(g_engine_node, self.gsm_def_class.G_engine, 'G_engine')
        
        # Variables section
        variables_node = Node("📋 Variables")
        root.add_node(variables_node)
        
        # Add variable categories
        var_categories = [
            ('eps_vars', '🔵 Strain Variables', 'eps_codenames'),
            ('sig_vars', '🔴 Stress Variables', 'sig_codenames'),
            ('Eps_vars', '🟢 Internal Strain Variables', 'Eps_codenames'),
            ('Sig_vars', '🟡 Internal Stress Variables', 'Sig_codenames'),
            ('m_params', '⚙️ Material Parameters', 'param_codenames')
        ]
        
        for attr_name, display_name, codename_attr in var_categories:
            self._add_variable_category(variables_node, attr_name, display_name, codename_attr)
        
        # Expressions section
        expressions_node = Node("🧮 Expressions")
        root.add_node(expressions_node)
        
        # Add symbolic expressions
        expr_attrs = [
            ('F_expr', '📈 Free Energy F'),
            ('f_expr', '💫 Dissipation Potential f'),
            ('phi_ext_expr', '🌐 External Potential φ'),
            ('h_k', '🔒 Constraints h')
        ]
        
        for attr_name, display_name in expr_attrs:
            self._add_expression_attribute(expressions_node, attr_name, display_name)
        
        # Derived expressions
        derived_node = Node("🧬 Derived Expressions")
        root.add_node(derived_node)
        
        derived_attrs = [
            ('eps_a_', '🔄 Strain Substitutions'),
            ('dot_eps_a_', '📈 Strain Rate Substitutions'),
            ('sig_x_eps_', '✖️ Stress × Strain'),
            ('subs_eps_sig', '🔀 ε→σ Substitutions'),
            ('subs_dot_eps_sig', '🔀 ε̇→σ̇ Substitutions')
        ]
        
        for attr_name, display_name in derived_attrs:
            if hasattr(self.gsm_def_class, attr_name):
                node = Node(display_name)
                node.observe(self._on_node_selected, 'selected')
                node.metadata = {'type': 'derived_expr', 'attr': attr_name, 'value': getattr(self.gsm_def_class, attr_name)}
                derived_node.add_node(node)
        
        # Expand the root node by default
        root.opened = True
        engines_node.opened = True
        variables_node.opened = True
        
    def _add_engine_attributes(self, parent_node, engine, engine_name):
        """Add engine attributes to the tree."""
        
        engine_attrs = [
            ('name', '📛 Name'),
            ('eps_vars', '🔵 Strain Variables'),
            ('sig_vars', '🔴 Stress Variables'),
            ('T_var', '🌡️ Temperature Variable'),
            ('m_params', '⚙️ Material Parameters'),
            ('Eps_vars', '🟢 Internal Strain Variables'),
            ('Sig_vars', '🟡 Internal Stress Variables'),
            ('F_expr', '📈 Free Energy Expression'),
            ('f_expr', '💫 Dissipation Potential'),
            ('phi_ext_expr', '🌐 External Potential')
        ]
        
        for attr_name, display_name in engine_attrs:
            if hasattr(engine, attr_name):
                node = Node(display_name)
                node.observe(self._on_node_selected, 'selected')
                node.metadata = {
                    'type': 'engine_attr',
                    'engine': engine_name,
                    'attr': attr_name,
                    'value': getattr(engine, attr_name)
                }
                parent_node.add_node(node)
                
    def _add_variable_category(self, parent_node, attr_name, display_name, codename_attr):
        """Add a category of variables to the tree."""
        
        if not hasattr(self.gsm_def_class, 'F_engine') or not self.gsm_def_class.F_engine:
            return
            
        engine = self.gsm_def_class.F_engine
        if not hasattr(engine, attr_name):
            return
            
        variables = getattr(engine, attr_name)
        if not variables:
            return
            
        category_node = Node(display_name)
        parent_node.add_node(category_node)
        
        # Handle different variable formats (tuple, list, matrix)
        if isinstance(variables, (tuple, list)):
            var_list = variables
        elif hasattr(variables, 'shape') and len(variables.shape) > 0:
            # Matrix case - extract elements
            var_list = [variables[i, 0] for i in range(variables.shape[0])]
        else:
            var_list = [variables]
        
        for i, var in enumerate(var_list):
            # Handle matrix variables
            if hasattr(var, 'shape') and var.shape == (1, 1):
                actual_var = var[0, 0]
            else:
                actual_var = var
                
            # Get the display name - use codename from class attribute if available
            var_name = actual_var.name if hasattr(actual_var, 'name') else str(actual_var)
            
            # Try to get codename from the GSM class attribute
            codename = var_name
            if hasattr(actual_var, 'codename'):
                codename = actual_var.codename
                
            display_text = f"{var_name}"
            if codename != var_name:
                display_text += f" ({codename})"
                
            var_node = Node(display_text)
            var_node.observe(self._on_node_selected, 'selected')
            var_node.metadata = {
                'type': 'variable',
                'category': attr_name,
                'index': i,
                'symbol': actual_var,
                'codename': codename,
                'value': actual_var
            }
            category_node.add_node(var_node)
            
    def _add_expression_attribute(self, parent_node, attr_name, display_name):
        """Add expression attributes to the tree."""
        
        if not hasattr(self.gsm_def_class, 'F_engine') or not self.gsm_def_class.F_engine:
            return
            
        engine = self.gsm_def_class.F_engine
        if not hasattr(engine, attr_name):
            return
            
        expr = getattr(engine, attr_name)
        if expr is None:
            return
            
        node = Node(display_name)
        node.observe(self._on_node_selected, 'selected')
        node.metadata = {
            'type': 'expression',
            'attr': attr_name,
            'value': expr
        }
        parent_node.add_node(node)
        
    def _on_node_selected(self, change):
        """Handle node selection in the tree."""
        
        if change['name'] == 'selected' and change['new']:
            node = change['owner']
            self.selected_item = node.name
            
            if hasattr(node, 'metadata'):
                self.selected_value = node.metadata
                self._update_detail_view()
            else:
                self.selected_value = None
                self._clear_detail_view()
                
    def _update_detail_view(self):
        """Update the detail view with information about the selected item."""
        
        if not self.selected_value:
            return
            
        with self.detail_output:
            self.detail_output.clear_output()
            
            metadata = self.selected_value
            item_type = metadata.get('type', 'unknown')
            
            print(f"📋 Selected: {self.selected_item}")
            print(f"🏷️ Type: {item_type}")
            print("=" * 50)
            
            if item_type == 'variable':
                self._display_variable_details(metadata)
            elif item_type == 'expression':
                self._display_expression_details(metadata)
            elif item_type == 'engine_attr':
                self._display_engine_attribute_details(metadata)
            elif item_type == 'derived_expr':
                self._display_derived_expression_details(metadata)
            else:
                print(f"Value: {metadata.get('value', 'N/A')}")
                
        self.status.value = f"Selected: {self.selected_item}"
        
    def _display_variable_details(self, metadata):
        """Display details for a variable."""
        
        symbol = metadata['value']
        category = metadata['category']
        codename = metadata.get('codename', 'N/A')
        
        print(f"📊 Variable Details:")
        print(f"  Category: {category}")
        print(f"  Symbol: {symbol}")
        print(f"  Codename: {codename}")
        print(f"  Type: {type(symbol).__name__}")
        
        if hasattr(symbol, 'assumptions0'):
            assumptions = symbol.assumptions0
            if assumptions:
                print(f"  Assumptions: {assumptions}")
                
        # Show LaTeX representation
        if hasattr(symbol, '_latex'):
            print(f"\\nLaTeX representation:")
            display(HTML(f"$${sp.latex(symbol)}$$"))
            
    def _display_expression_details(self, metadata):
        """Display details for an expression."""
        
        expr = metadata['value']
        attr_name = metadata['attr']
        
        print(f"🧮 Expression Details:")
        print(f"  Attribute: {attr_name}")
        print(f"  Type: {type(expr).__name__}")
        
        if isinstance(expr, (list, tuple)):
            print(f"  Length: {len(expr)}")
            print(f"  Contents:")
            for i, item in enumerate(expr):
                print(f"    [{i}]: {item}")
                if i < len(expr) - 1 and isinstance(item, sp.Basic):
                    display(HTML(f"$${sp.latex(item)}$$"))
        elif isinstance(expr, sp.Basic):
            print(f"  Expression: {expr}")
            print(f"\\nLaTeX representation:")
            display(HTML(f"$${sp.latex(expr)}$$"))
        else:
            print(f"  Value: {expr}")
            
    def _display_engine_attribute_details(self, metadata):
        """Display details for an engine attribute."""
        
        value = metadata['value']
        attr_name = metadata['attr']
        engine_name = metadata['engine']
        
        print(f"⚡ Engine Attribute Details:")
        print(f"  Engine: {engine_name}")
        print(f"  Attribute: {attr_name}")
        print(f"  Type: {type(value).__name__}")
        
        if isinstance(value, str):
            print(f"  Value: {value}")
        elif isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            print(f"  Contents:")
            for i, item in enumerate(value[:5]):  # Show first 5 items
                print(f"    [{i}]: {item}")
                if isinstance(item, sp.Basic):
                    display(HTML(f"$${sp.latex(item)}$$"))
            if len(value) > 5:
                print(f"    ... and {len(value) - 5} more items")
        elif isinstance(value, sp.Basic):
            print(f"  Expression: {value}")
            print(f"\\nLaTeX representation:")
            display(HTML(f"$${sp.latex(value)}$$"))
        else:
            print(f"  Value: {value}")
            
    def _display_derived_expression_details(self, metadata):
        """Display details for a derived expression."""
        
        value = metadata['value']
        attr_name = metadata['attr']
        
        print(f"🧬 Derived Expression Details:")
        print(f"  Attribute: {attr_name}")
        print(f"  Type: {type(value).__name__}")
        
        if isinstance(value, dict):
            print(f"  Dictionary with {len(value)} entries:")
            for i, (key, val) in enumerate(list(value.items())[:3]):
                print(f"    {key} → {val}")
                if isinstance(key, sp.Basic) and isinstance(val, sp.Basic):
                    display(HTML(f"$${sp.latex(key)} \\rightarrow {sp.latex(val)}$$"))
            if len(value) > 3:
                print(f"    ... and {len(value) - 3} more entries")
        elif isinstance(value, sp.Matrix):
            print(f"  Matrix shape: {value.shape}")
            print(f"  Matrix:")
            display(HTML(f"$${sp.latex(value)}$$"))
        elif isinstance(value, sp.Basic):
            print(f"  Expression: {value}")
            print(f"\\nLaTeX representation:")
            display(HTML(f"$${sp.latex(value)}$$"))
        else:
            print(f"  Value: {value}")
            
    def _clear_detail_view(self):
        """Clear the detail view."""
        
        with self.detail_output:
            self.detail_output.clear_output()
            print("Select an item from the tree to view details")
            
        self.status.value = "Select an item from the tree to view details"
        
    def _setup_layout(self):
        """Setup the widget layout."""
        
        # Main content area
        main_box = widgets.HBox([
            self.tree,
            self.detail_output
        ])
        main_box.layout.border = '1px solid #ddd'
        
        # Full interface
        self.interface = widgets.VBox([
            self.title,
            main_box,
            self.status
        ])
        
        # Style the components
        self.tree.layout.border = '1px solid #ccc'
        self.tree.layout.padding = '10px'
        self.detail_output.layout.border = '1px solid #ccc'
        self.detail_output.layout.padding = '10px'
        self.status.layout.margin = '10px 0 0 0'
        
    def display(self):
        """Display the widget."""
        display(self.interface)
        
        # Show initial message
        with self.detail_output:
            print("🔬 GSM Model Explorer")
            print("=" * 30)
            print("Select an item from the tree on the left to explore")
            print("the GSM model structure and attributes.")
            print("")
            print("Available sections:")
            print("🔧 Engines - F_engine (Helmholtz) and G_engine (Gibbs)")
            print("📋 Variables - All symbolic variables with codenames")
            print("🧮 Expressions - Symbolic expressions and potentials")
            print("🧬 Derived - Computed substitutions and transforms")


def create_gsm_widget(gsm_def_class):
    """
    Factory function to create a GSM Definition widget.
    
    Parameters
    ----------
    gsm_def_class : GSMDef subclass
        The GSM definition class to visualize
        
    Returns
    -------
    GSMDefWidget
        Configured widget instance
    """
    return GSMDefWidget(gsm_def_class)
