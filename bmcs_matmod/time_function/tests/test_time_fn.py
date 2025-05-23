import numpy as np
import pytest
import sympy as sp
from ..time_fn import (
    TimeFnBase, TimeFnStepLoading, TimeFnMonotonicAscending, 
    TimeFnCycleSinus, TimeFnCycleLinear, TimeFnCycleWithRamps,
    TimeFnPeriodic, TimeFnStepping, TimeFnOverlay, SymbolRegistry
)

# Set of time values for numerical testing
test_time = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])

class TestSymbolRegistry:
    """Tests for the SymbolRegistry class."""
    
    def test_symbol_reuse(self):
        """Test that the same symbol is returned for the same name and attributes."""
        sym1 = SymbolRegistry.get_symbol('x', real=True)
        sym2 = SymbolRegistry.get_symbol('x', real=True)
        assert sym1 is sym2, "Should return the same symbol object for identical parameters"
        
    def test_different_symbols(self):
        """Test that different symbols are created for different names or attributes."""
        sym1 = SymbolRegistry.get_symbol('x', real=True)
        sym2 = SymbolRegistry.get_symbol('y', real=True)
        sym3 = SymbolRegistry.get_symbol('x', real=False)
        assert sym1 is not sym2, "Different names should produce different symbols"
        assert sym1 is not sym3, "Different attributes should produce different symbols"

class TestTimeFnBase:
    """Tests for the TimeFnBase class and its core functionality."""
    
    def test_symb_vars_creation(self):
        """Test that symbolic variables are properly created for instances."""
        # Create an instance to check if symb_vars is created correctly
        step_fn = TimeFnStepLoading(t_s=2.0, val=3.0)
        assert hasattr(step_fn, 'symb_vars'), "Instance should have symb_vars attribute"
        assert len(step_fn.symb_vars) == 1, "Should have one symbolic variable (t)"
        assert str(step_fn.symb_vars[0]) == 't', "Variable should be named 't'"
        
    def test_symb_params_creation(self):
        """Test that symbolic parameters are properly created for instances."""
        # Create an instance to check if symb_params is created correctly
        step_fn = TimeFnStepLoading(t_s=2.0, val=3.0)
        assert hasattr(step_fn, 'symb_params'), "Instance should have symb_params attribute"
        
        # Verify that symbolic traits with _sym suffix exist for each parameter
        assert hasattr(step_fn, 't_s_sym'), "Symbol trait for t_s missing"
        assert hasattr(step_fn, 'val_sym'), "Symbol trait for val missing"
        
        # Check that the symbolic traits contain sympy.Symbol instances
        assert isinstance(step_fn.t_s_sym, sp.Symbol), "t_s_sym should be a sympy Symbol"
        assert isinstance(step_fn.val_sym, sp.Symbol), "val_sym should be a sympy Symbol"
        
class TestTimeFnStepLoading:
    """Tests for the TimeFnStepLoading class."""
    
    def test_symb_expr(self):
        """Test that the symbolic expression is correctly defined."""
        step_fn = TimeFnStepLoading(t_s=2.0, val=3.0)
        expr = step_fn.symb_expr
        assert expr is not None, "Symbolic expression should be defined"
        assert isinstance(expr, sp.Piecewise), "Expression should be a Piecewise function"
        
    def test_call(self):
        """Test the numerical evaluation of the function."""
        step_fn = TimeFnStepLoading(t_s=2.0, val=3.0)
        result = step_fn(test_time)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0])
        np.testing.assert_allclose(result, expected)

class TestTimeFnMonotonicAscending:
    """Tests for the TimeFnMonotonicAscending class."""
    
    def test_symb_expr(self):
        """Test that the symbolic expression is correctly defined."""
        monotonic_fn = TimeFnMonotonicAscending(rate=1.5)
        expr = monotonic_fn.symb_expr
        assert expr is not None, "Symbolic expression should be defined"
        
    def test_call(self):
        """Test the numerical evaluation of the function."""
        rate = 1.5
        monotonic_fn = TimeFnMonotonicAscending(rate=rate)
        result = monotonic_fn(test_time)
        expected = rate * test_time
        np.testing.assert_allclose(result, expected)

class TestTimeFnCycleSinus:
    """Tests for the TimeFnCycleSinus class."""
    
    def test_symb_expr(self):
        """Test that the symbolic expression is correctly defined."""
        sin_fn = TimeFnCycleSinus()
        expr = sin_fn.symb_expr
        assert expr is not None, "Symbolic expression should be defined"
        assert isinstance(expr, sp.Piecewise), "Expression should be a Piecewise function"
        
    def test_call(self):
        """Test the numerical evaluation of the function."""
        sin_fn = TimeFnCycleSinus()
        result = sin_fn(np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]))
        expected = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

class TestTimeFnPeriodic:
    """Tests for the TimeFnPeriodic class."""
    
    def test_symb_expr(self):
        """Test that the symbolic expression is correctly defined."""
        cycle = TimeFnCycleSinus()
        periodic_fn = TimeFnPeriodic(mean_value=2.0, amplitude=1.5, period=3.0, time_fn_cycle=cycle)
        expr = periodic_fn.symb_expr
        assert expr is not None, "Symbolic expression should be defined"
                
class TestTimeFnStepping:
    """Tests for the TimeFnStepping class (numerical approach)."""
    
    def test_call(self):
        """Test the numerical evaluation of the function."""
        step_times = np.array([0.0, 2.0, 4.0])
        step_values = np.array([1.0, 2.0, 3.0])
        stepping_fn = TimeFnStepping(step_times=step_times, step_values=step_values)
        
        result = stepping_fn(test_time)
        expected = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
        np.testing.assert_array_equal(result, expected)

class TestTimeFnOverlay:
    """Tests for the TimeFnOverlay class (hybrid approach)."""
    
    def test_call(self):
        """Test the numerical evaluation of the function."""
        fn1 = TimeFnStepLoading(t_s=2.0, val=2.0)
        fn2 = TimeFnMonotonicAscending(rate=0.5)
        overlay_fn = TimeFnOverlay(fn1=fn1, fn2=fn2)
        
        result = overlay_fn(test_time)
        # Expected: fn1(t) * fn2(t)
        fn1_vals = fn1(test_time)
        fn2_vals = fn2(test_time)
        expected = fn1_vals * fn2_vals
        
        np.testing.assert_allclose(result, expected)
    
    def test_symb_expr(self):
        """Test the symbolic expression when both functions have symbolic expressions."""
        fn1 = TimeFnStepLoading(t_s=2.0, val=2.0)
        fn2 = TimeFnMonotonicAscending(rate=0.5)
        overlay_fn = TimeFnOverlay(fn1=fn1, fn2=fn2)
        
        # The symbolic expression should be the product of the two expressions
        expr = overlay_fn.symb_expr
        assert expr is not None, "Symbolic expression should be defined"
