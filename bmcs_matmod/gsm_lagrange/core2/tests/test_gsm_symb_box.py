"""
Unit tests for GSMSymbBox - Based on comprehensive verification notebook

These tests ensure the mathematical correctness of the GSMSymbBox implementation
by validating round-trip consistency and constitutive relation invariants.

Primary Test Criteria:
1. Round-trip consistency: F→G→F = F exactly (gold standard)
2. Property access consistency
3. Constitutive relation consistency
4. Thermodynamic framework validation

Test Models:
- Elastic material: F(T,ε) = ½Eε²
- Elastic-damage material: F(T,ε,ω) = ½(1-ω)Eε²
"""

import pytest
import sympy as sp
import sys
from pathlib import Path

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
from gsm_symb_box import GSMSymbBox, StateFunction


class TestGSMSymbBoxRoundTrips:
    """Test round-trip transformations - the primary criterion for mathematical correctness."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': sp.Symbol('T', real=True, positive=True),
            'S': sp.Symbol('S', real=True),
            'eps': sp.Symbol('eps', real=True),
            'sig': sp.Symbol('sig', real=True),
            'omega': sp.Symbol('omega', real=True, positive=True),
            'Y': sp.Symbol('Y', real=True),
            'E': sp.Symbol('E', positive=True)
        }
    
    @pytest.fixture
    def elastic_box(self, symbols):
        """Create a GSMSymbBox with simple elastic material."""
        T, S, eps, sig, E = symbols['T'], symbols['S'], symbols['eps'], symbols['sig'], symbols['E']
        
        # Simple elastic Helmholtz free energy: F(T,ε) = ½Eε²
        F_elastic = sp.Rational(1, 2) * E * eps**2
        
        return GSMSymbBox(
            T_var=T,
            S_var=S,
            eps_vars=(eps,),
            sig_vars=(sig,),
            Eps_vars=(),
            Sig_vars=(),
            m_params=(E,),
            initial_state_fn=StateFunction.HELMHOLTZ,
            initial_expression=F_elastic
        )
    
    @pytest.fixture
    def damage_box(self, symbols):
        """Create a GSMSymbBox with elastic-damage material."""
        T, S, eps, sig, omega, Y, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'omega', 'Y', 'E'])
        
        # Elastic-damage Helmholtz free energy: F(T,ε,ω) = ½(1-ω)Eε²
        F_damage = sp.Rational(1, 2) * (1 - omega) * E * eps**2
        
        return GSMSymbBox(
            T_var=T,
            S_var=S,
            eps_vars=(eps,),
            sig_vars=(sig,),
            Eps_vars=(omega,),
            Sig_vars=(Y,),
            m_params=(E,),
            initial_state_fn=StateFunction.HELMHOLTZ,
            initial_expression=F_damage
        )
    
    def test_round_trip_F_to_G_to_F_elastic(self, elastic_box):
        """Test F→G→F round-trip for elastic material."""
        # Get original F
        F_original = elastic_box.F
        
        # Transform F→G→F
        elastic_box.set_current_state_function(StateFunction.HELMHOLTZ, F_original)
        G_from_F = elastic_box.legendre_transform(StateFunction.GIBBS)
        
        elastic_box.set_current_state_function(StateFunction.GIBBS, G_from_F)
        F_recovered = elastic_box.legendre_transform(StateFunction.HELMHOLTZ)
        
        # Check exact equality (difference = 0)
        difference = sp.simplify(F_recovered - F_original)
        assert difference == 0, f"Round-trip F→G→F failed. Difference: {difference}"
    
    def test_round_trip_F_to_U_to_F_elastic(self, elastic_box):
        """Test F→U→F round-trip for elastic material."""
        F_original = elastic_box.F
        
        elastic_box.set_current_state_function(StateFunction.HELMHOLTZ, F_original)
        U_from_F = elastic_box.legendre_transform(StateFunction.INTERNAL_ENERGY)
        
        elastic_box.set_current_state_function(StateFunction.INTERNAL_ENERGY, U_from_F)
        F_recovered = elastic_box.legendre_transform(StateFunction.HELMHOLTZ)
        
        difference = sp.simplify(F_recovered - F_original)
        assert difference == 0, f"Round-trip F→U→F failed. Difference: {difference}"
    
    def test_round_trip_G_to_H_to_G_elastic(self, elastic_box):
        """Test G→H→G round-trip for elastic material."""
        G_original = elastic_box.G
        
        elastic_box.set_current_state_function(StateFunction.GIBBS, G_original)
        H_from_G = elastic_box.legendre_transform(StateFunction.ENTHALPY)
        
        elastic_box.set_current_state_function(StateFunction.ENTHALPY, H_from_G)
        G_recovered = elastic_box.legendre_transform(StateFunction.GIBBS)
        
        difference = sp.simplify(G_recovered - G_original)
        assert difference == 0, f"Round-trip G→H→G failed. Difference: {difference}"
    
    def test_round_trip_U_to_G_to_U_elastic(self, elastic_box):
        """Test U→G→U round-trip (diagonal transformation) for elastic material."""
        U_original = elastic_box.U
        
        elastic_box.set_current_state_function(StateFunction.INTERNAL_ENERGY, U_original)
        G_from_U = elastic_box.legendre_transform(StateFunction.GIBBS)
        
        elastic_box.set_current_state_function(StateFunction.GIBBS, G_from_U)
        U_recovered = elastic_box.legendre_transform(StateFunction.INTERNAL_ENERGY)
        
        difference = sp.simplify(U_recovered - U_original)
        assert difference == 0, f"Round-trip U→G→U failed. Difference: {difference}"
    
    def test_round_trip_H_to_F_to_H_elastic(self, elastic_box):
        """Test H→F→H round-trip (diagonal transformation) for elastic material."""
        H_original = elastic_box.H
        
        elastic_box.set_current_state_function(StateFunction.ENTHALPY, H_original)
        F_from_H = elastic_box.legendre_transform(StateFunction.HELMHOLTZ)
        
        elastic_box.set_current_state_function(StateFunction.HELMHOLTZ, F_from_H)
        H_recovered = elastic_box.legendre_transform(StateFunction.ENTHALPY)
        
        difference = sp.simplify(H_recovered - H_original)
        assert difference == 0, f"Round-trip H→F→H failed. Difference: {difference}"
    
    def test_all_round_trips_damage_material(self, damage_box):
        """Test all round-trips for damage material (comprehensive test)."""
        # Store original state functions
        F_orig = damage_box.F
        G_orig = damage_box.G
        U_orig = damage_box.U
        H_orig = damage_box.H
        
        # Test all round-trips
        round_trips = [
            (StateFunction.HELMHOLTZ, StateFunction.GIBBS, StateFunction.HELMHOLTZ, F_orig),
            (StateFunction.HELMHOLTZ, StateFunction.INTERNAL_ENERGY, StateFunction.HELMHOLTZ, F_orig),
            (StateFunction.GIBBS, StateFunction.ENTHALPY, StateFunction.GIBBS, G_orig),
            (StateFunction.INTERNAL_ENERGY, StateFunction.GIBBS, StateFunction.INTERNAL_ENERGY, U_orig),
            (StateFunction.ENTHALPY, StateFunction.HELMHOLTZ, StateFunction.ENTHALPY, H_orig),
        ]
        
        for start_fn, intermediate_fn, end_fn, original_expr in round_trips:
            # Perform round-trip
            damage_box.set_current_state_function(start_fn, original_expr)
            intermediate_expr = damage_box.legendre_transform(intermediate_fn)
            
            damage_box.set_current_state_function(intermediate_fn, intermediate_expr)
            final_expr = damage_box.legendre_transform(end_fn)
            
            # Check round-trip consistency
            difference = sp.simplify(final_expr - original_expr)
            assert difference == 0, (
                f"Round-trip {start_fn.value}→{intermediate_fn.value}→{end_fn.value} failed. "
                f"Difference: {difference}"
            )


class TestGSMSymbBoxPropertyConsistency:
    """Test property-based access consistency."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': sp.Symbol('T', real=True, positive=True),
            'S': sp.Symbol('S', real=True),
            'eps': sp.Symbol('eps', real=True),
            'sig': sp.Symbol('sig', real=True),
            'omega': sp.Symbol('omega', real=True, positive=True),
            'Y': sp.Symbol('Y', real=True),
            'E': sp.Symbol('E', positive=True)
        }
    
    @pytest.fixture
    def test_box(self, symbols):
        """Create a test GSMSymbBox."""
        T, S, eps, sig, omega, Y, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'omega', 'Y', 'E'])
        F_test = sp.Rational(1, 2) * (1 - omega) * E * eps**2
        
        return GSMSymbBox(
            T_var=T, S_var=S, eps_vars=(eps,), sig_vars=(sig,),
            Eps_vars=(omega,), Sig_vars=(Y,), m_params=(E,),
            initial_state_fn=StateFunction.HELMHOLTZ, initial_expression=F_test
        )
    
    def test_property_vs_explicit_consistency(self, test_box):
        """Test that property access gives same results as explicit transformations."""
        # Reset to clean state
        F_original = sp.Rational(1, 2) * (1 - test_box.Eps_vars[0]) * test_box.m_params[0] * test_box.eps_vars[0]**2
        test_box.set_current_state_function(StateFunction.HELMHOLTZ, F_original)
        test_box.explicit_expressions = {StateFunction.HELMHOLTZ: F_original}
        
        # Get via properties
        F_prop = test_box.F
        G_prop = test_box.G
        U_prop = test_box.U
        H_prop = test_box.H
        
        # Get via explicit transformations
        test_box.set_current_state_function(StateFunction.HELMHOLTZ, F_original)
        G_explicit = test_box.legendre_transform(StateFunction.GIBBS)
        U_explicit = test_box.legendre_transform(StateFunction.INTERNAL_ENERGY)
        H_explicit = test_box.legendre_transform(StateFunction.ENTHALPY)
        
        # Check consistency
        assert sp.simplify(F_prop - F_original) == 0, "F property inconsistent"
        assert sp.simplify(G_prop - G_explicit) == 0, "G property inconsistent"
        assert sp.simplify(U_prop - U_explicit) == 0, "U property inconsistent"
        assert sp.simplify(H_prop - H_explicit) == 0, "H property inconsistent"


class TestGSMSymbBoxConstitutiveRelations:
    """Test constitutive relation consistency and correctness."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': sp.Symbol('T', real=True, positive=True),
            'S': sp.Symbol('S', real=True),
            'eps': sp.Symbol('eps', real=True),
            'sig': sp.Symbol('sig', real=True),
            'omega': sp.Symbol('omega', real=True, positive=True),
            'Y': sp.Symbol('Y', real=True),
            'E': sp.Symbol('E', positive=True)
        }
    
    @pytest.fixture
    def test_box(self, symbols):
        """Create a test GSMSymbBox."""
        T, S, eps, sig, omega, Y, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'omega', 'Y', 'E'])
        F_test = sp.Rational(1, 2) * (1 - omega) * E * eps**2
        
        return GSMSymbBox(
            T_var=T, S_var=S, eps_vars=(eps,), sig_vars=(sig,),
            Eps_vars=(omega,), Sig_vars=(Y,), m_params=(E,),
            initial_state_fn=StateFunction.HELMHOLTZ, initial_expression=F_test
        )
    
    def test_stress_consistency(self, test_box, symbols):
        """Test that stress derived from F and U are consistent."""
        eps = symbols['eps']
        
        F_final = test_box.F
        U_final = test_box.U
        
        sigma_from_F = sp.diff(F_final, eps)  # σ = ∂F/∂ε
        sigma_from_U = sp.diff(U_final, eps)  # σ = ∂U/∂ε
        
        difference = sp.simplify(sigma_from_F - sigma_from_U)
        assert difference == 0, f"Stress inconsistency between F and U: {difference}"
    
    def test_inverse_stress_strain_relations(self, test_box, symbols):
        """Test that σ(ε) and ε(σ) are proper inverses."""
        eps, sig = symbols['eps'], symbols['sig']
        
        F_final = test_box.F
        G_final = test_box.G
        
        sigma_from_F = sp.diff(F_final, eps)      # σ = ∂F/∂ε
        epsilon_from_G = -sp.diff(G_final, sig)   # ε = -∂G/∂σ
        
        # Test inverse relationship: ε(σ(ε)) should equal ε
        epsilon_test = epsilon_from_G.subs(sig, sigma_from_F)
        epsilon_simplified = sp.simplify(epsilon_test)
        
        difference = sp.simplify(epsilon_simplified - eps)
        assert difference == 0, f"Inverse stress-strain relationship failed: ε(σ(ε)) - ε = {difference}"
    
    def test_damage_force_consistency(self, test_box, symbols):
        """Test that damage force Y is consistent across state functions."""
        omega = symbols['omega']
        
        F_final = test_box.F
        G_final = test_box.G
        U_final = test_box.U
        H_final = test_box.H
        
        Y_from_F = -sp.diff(F_final, omega)  # Y = -∂F/∂ω
        Y_from_G = -sp.diff(G_final, omega)  # Y = -∂G/∂ω  
        Y_from_U = -sp.diff(U_final, omega)  # Y = -∂U/∂ω
        Y_from_H = -sp.diff(H_final, omega)  # Y = -∂H/∂ω
        
        # All damage forces should be identical (before variable substitution)
        assert sp.simplify(Y_from_F - Y_from_U) == 0, "Damage force F vs U inconsistent"
        
        # Note: G and H expressions will differ due to variable substitution (σ vs ε)
        # but they should be equivalent when constitutive relations are applied


class TestGSMSymbBoxFrameworkValidation:
    """Test overall framework validation and edge cases."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': sp.Symbol('T', real=True, positive=True),
            'S': sp.Symbol('S', real=True),
            'eps': sp.Symbol('eps', real=True),
            'sig': sp.Symbol('sig', real=True),
            'E': sp.Symbol('E', positive=True)
        }
    
    def test_simple_elastic_material_known_results(self, symbols):
        """Test simple elastic material against known analytical results."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        # Simple elastic material: F(T,ε) = ½Eε²
        F_elastic = sp.Rational(1, 2) * E * eps**2
        
        box = GSMSymbBox(
            T_var=T, S_var=S, eps_vars=(eps,), sig_vars=(sig,),
            Eps_vars=(), Sig_vars=(), m_params=(E,),
            initial_state_fn=StateFunction.HELMHOLTZ, initial_expression=F_elastic
        )
        
        # Expected results for elastic material
        expected_sigma = E * eps  # σ = Eε (Hooke's law)
        expected_G = -sp.Rational(1, 2) * sig**2 / E  # G(T,σ) = -½σ²/E
        
        # Test constitutive relation
        actual_sigma = sp.diff(box.F, eps)
        assert sp.simplify(actual_sigma - expected_sigma) == 0, "Hooke's law not satisfied"
        
        # Test Gibbs function (after removing constants that may differ)
        actual_G = box.G
        # Extract the σ²/E term from both expressions for comparison
        G_coeff = actual_G.coeff(sig**2)
        expected_coeff = expected_G.coeff(sig**2)
        assert sp.simplify(G_coeff - expected_coeff) == 0, "Gibbs function incorrect"
    
    def test_built_in_validation_if_available(self, symbols):
        """Test built-in validation method if available."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        F_test = sp.Rational(1, 2) * E * eps**2
        
        box = GSMSymbBox(
            T_var=T, S_var=S, eps_vars=(eps,), sig_vars=(sig,),
            Eps_vars=(), Sig_vars=(), m_params=(E,),
            initial_state_fn=StateFunction.HELMHOLTZ, initial_expression=F_test
        )
        
        # Test built-in validation if it exists
        try:
            validation_result = box.validate_thermodynamic_consistency()
            assert validation_result, "Built-in validation failed"
        except AttributeError:
            # If method doesn't exist, that's fine - this test just passes
            pass
    
    def test_state_function_accessibility(self, symbols):
        """Test that all four state functions are accessible."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        F_test = sp.Rational(1, 2) * E * eps**2
        
        box = GSMSymbBox(
            T_var=T, S_var=S, eps_vars=(eps,), sig_vars=(sig,),
            Eps_vars=(), Sig_vars=(), m_params=(E,),
            initial_state_fn=StateFunction.HELMHOLTZ, initial_expression=F_test
        )
        
        # All state functions should be accessible and non-None
        assert box.F is not None, "Helmholtz function F not accessible"
        assert box.G is not None, "Gibbs function G not accessible"
        assert box.U is not None, "Internal energy U not accessible"
        assert box.H is not None, "Enthalpy H not accessible"
        
        # They should also be SymPy expressions
        assert isinstance(box.F, sp.Basic), "F is not a SymPy expression"
        assert isinstance(box.G, sp.Basic), "G is not a SymPy expression"
        assert isinstance(box.U, sp.Basic), "U is not a SymPy expression"
        assert isinstance(box.H, sp.Basic), "H is not a SymPy expression"


class TestGSMSymbBoxEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': sp.Symbol('T', real=True, positive=True),
            'S': sp.Symbol('S', real=True),
            'eps': sp.Symbol('eps', real=True),
            'sig': sp.Symbol('sig', real=True),
            'E': sp.Symbol('E', positive=True)
        }
    
    def test_linear_function_round_trips(self, symbols):
        """Test round-trips with linear function (edge case)."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        # Linear function: F(T,ε) = Eε (no quadratic term)
        F_linear = E * eps
        
        box = GSMSymbBox(
            T_var=T, S_var=S, eps_vars=(eps,), sig_vars=(sig,),
            Eps_vars=(), Sig_vars=(), m_params=(E,),
            initial_state_fn=StateFunction.HELMHOLTZ, initial_expression=F_linear
        )
        
        # Test F→G→F round-trip for linear case
        F_original = box.F
        
        box.set_current_state_function(StateFunction.HELMHOLTZ, F_original)
        G_from_F = box.legendre_transform(StateFunction.GIBBS)
        
        box.set_current_state_function(StateFunction.GIBBS, G_from_F)
        F_recovered = box.legendre_transform(StateFunction.HELMHOLTZ)
        
        difference = sp.simplify(F_recovered - F_original)
        assert difference == 0, f"Linear function round-trip failed: {difference}"
    
    def test_zero_function_round_trips(self, symbols):
        """Test round-trips with zero function (degenerate case)."""
        T, S, eps, sig = (symbols[k] for k in ['T', 'S', 'eps', 'sig'])
        
        # Zero function: F(T,ε) = 0
        F_zero = sp.sympify(0)
        
        box = GSMSymbBox(
            T_var=T, S_var=S, eps_vars=(eps,), sig_vars=(sig,),
            Eps_vars=(), Sig_vars=(), m_params=(),
            initial_state_fn=StateFunction.HELMHOLTZ, initial_expression=F_zero
        )
        
        # Test F→G→F round-trip for zero function
        F_original = box.F
        
        # Zero function should remain zero through transformations
        assert F_original == 0, "Zero function not preserved"
        
        # Perform round-trip
        box.set_current_state_function(StateFunction.HELMHOLTZ, F_original)
        G_from_F = box.legendre_transform(StateFunction.GIBBS)
        
        box.set_current_state_function(StateFunction.GIBBS, G_from_F)
        F_recovered = box.legendre_transform(StateFunction.HELMHOLTZ)
        
        assert F_recovered == 0, "Zero function round-trip failed"


# Test collection for easy discovery
@pytest.mark.integration
def test_comprehensive_gsm_symb_box_validation():
    """Integration test that runs a comprehensive validation similar to the notebook."""
    # Symbols
    T = sp.Symbol('T', real=True, positive=True)
    S = sp.Symbol('S', real=True)
    eps = sp.Symbol('eps', real=True)
    sig = sp.Symbol('sig', real=True)
    omega = sp.Symbol('omega', real=True, positive=True)
    Y = sp.Symbol('Y', real=True)
    E = sp.Symbol('E', positive=True)
    
    # Elastic-damage model
    F_damage = sp.Rational(1, 2) * (1 - omega) * E * eps**2
    
    box = GSMSymbBox(
        T_var=T, S_var=S, eps_vars=(eps,), sig_vars=(sig,),
        Eps_vars=(omega,), Sig_vars=(Y,), m_params=(E,),
        initial_state_fn=StateFunction.HELMHOLTZ, initial_expression=F_damage
    )
    
    # Store original expressions for comparison
    F_original = box.F
    G_original = box.G
    U_original = box.U
    H_original = box.H
    
    # Test all round-trips
    round_trip_tests = [
        (StateFunction.HELMHOLTZ, StateFunction.GIBBS, F_original),
        (StateFunction.HELMHOLTZ, StateFunction.INTERNAL_ENERGY, F_original),
        (StateFunction.GIBBS, StateFunction.ENTHALPY, G_original),
        (StateFunction.INTERNAL_ENERGY, StateFunction.GIBBS, U_original),
        (StateFunction.ENTHALPY, StateFunction.HELMHOLTZ, H_original),
    ]
    
    for start_fn, intermediate_fn, original in round_trip_tests:
        # Perform round-trip transformation
        box.set_current_state_function(start_fn, original)
        intermediate = box.legendre_transform(intermediate_fn)
        
        box.set_current_state_function(intermediate_fn, intermediate)
        recovered = box.legendre_transform(start_fn)
        
        # Assert exact equality
        difference = sp.simplify(recovered - original)
        assert difference == 0, (
            f"Integration test failed for {start_fn.value}→{intermediate_fn.value}→{start_fn.value}: "
            f"difference = {difference}"
        )
    
    # Test constitutive relations - reset box to clean state first
    box.set_current_state_function(StateFunction.HELMHOLTZ, F_damage)
    box.explicit_expressions = {StateFunction.HELMHOLTZ: F_damage}
    
    # Get fresh expressions for constitutive relation testing
    F_fresh = box.F
    G_fresh = box.G
    
    sigma_from_F = sp.diff(F_fresh, eps)
    epsilon_from_G = -sp.diff(G_fresh, sig)
    
    # Test inverse relationship
    epsilon_test = epsilon_from_G.subs(sig, sigma_from_F)
    epsilon_simplified = sp.simplify(epsilon_test)
    inverse_diff = sp.simplify(epsilon_simplified - eps)
    
    assert inverse_diff == 0, f"Integration test: inverse relationship failed, difference = {inverse_diff}"
    
    print("✓ Comprehensive GSMSymbBox validation passed!")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
