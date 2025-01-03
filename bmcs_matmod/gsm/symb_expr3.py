import os
from traits.api import HasTraits, Str
import sympy as sp
import dill

# Directory to store the serialized symbolic instances
CACHE_DIR = '_symbolic_cache'

class SymbExpr(HasTraits):
    """Base class for symbolic expressions."""

    def save_to_disk(self):
        """Serialize this instance to disk."""
        filepath = os.path.join(CACHE_DIR, f"{self.name}.pkl")
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load_from_disk(name):
        """Deserialize an instance from disk."""
        filepath = os.path.join(CACHE_DIR, f"{name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No cached data found for {name}.")
        with open(filepath, 'rb') as file:
            return dill.load(file)

def initialize_symbolic_cache():
    """Initialize symbolic instances and serialize them to disk."""
    expr1 = SymbExpr('Expression1')
    x = expr1.add_symbol('x')
    y = expr1.add_symbol('y')
    expr1.add_expression('quadratic', x**2 + y**2)
    expr1.save_to_disk()

class Model:
    """Model class utilizing [re]loaded symbolic expressions."""
    def __init__(self, symbolic_key):
        try:
            self.symbolic_expr = SymbExpr.load_from_disk(symbolic_key)
        except FileNotFoundError as e:
            print(str(e))
            self.symbolic_expr = None

    def execute(self, expr_name, values):
        """Execute the lambdified function with given values."""
        if self.symbolic_expr:
            func = self.symbolic_expr.lambdify(expr_name)
            return func(*values)
        raise ValueError(f"No symbolic expression loaded for key: {self.symbolic_key}")

# Initialize the symbolic cache (should be done once, e.g., in a setup script or notebook)
initialize_symbolic_cache()

# Example usage
model = Model('Expression1')
result = model.execute('quadratic', [2, 3])  # Should compute 2^2 + 3^2
print("Result of quadratic:", result)