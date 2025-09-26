"""
Unit tests for GSMThermodynBox - Interface-based Implementation

These tests ensure the mathematical correctness of the GSMThermodynBox implementation
by validating round-trip consistency and constitutive relation invariants.

Primary Test Criteria:
1. Round-trip consistency: F→G→F = F exactly (gold standard)
2. Property access consistency
3. Constitutive relation consistency
4. Thermodynamic framework validation
5. Interface-based state function management

Test Models:
- Elastic material: F(T,ε) = ½Eε²
- Elastic-damage material: F(T,ε,ω) = ½(1-ω)Eε²
- Thermal expansion material: F(T,ε) = ½Eε² + αT²
"""

import pytest
import sympy as sp
import sys
from pathlib import Path

# Import the modules under test using full import paths
from bmcs_matmod.gsm_lagrange.core2.gsm_thermodyn_box import GSMThermodynBox
from bmcs_matmod.gsm_lagrange.core2.gsm_state_fn import GSMStateFn, StateFunctionType
from bmcs_matmod.gsm_lagrange.core2.gsm_vars import Scalar


class TestGSMThermodynBoxRoundTrips:
    """Test round-trip transformations - the primary criterion for mathematical correctness."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': Scalar(r'\vartheta', codename='T', real=True, positive=True),
            'S': Scalar('S', codename='S', real=True),
            'eps': Scalar(r'\varepsilon', codename='eps', real=True),
            'sig': Scalar(r'\sigma', codename='sig', real=True),
            'omega': Scalar(r'\omega', codename='omega', real=True, positive=True),
            'Y': Scalar('Y', codename='Y', real=True),
            'E': Scalar('E', codename='E', positive=True)
        }
    
    @pytest.fixture
    def elastic_box(self, symbols):
        """Create a GSMThermodynBox with simple elastic material."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        # Simple elastic Helmholtz free energy: F(T,ε) = ½Eε²
        F_elastic = sp.Rational(1, 2) * E * eps**2
        
        # Create the initial state function instance
        F_state_fn = GSMStateFn(
            fn_expr=F_elastic,
            th_x_var=T,      # Temperature is natural variable
            th_y_var=S,      # Entropy is conjugate variable
            mc_x_var=eps,    # Strain is natural variable
            mc_y_var=sig,    # Stress is conjugate variable
            Eps_var=sp.Symbol('eps_dummy'),  # Dummy internal variable
            Sig_var=sp.Symbol('sig_dummy'),  # Dummy internal conjugate
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        return GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
    
    @pytest.fixture
    def damage_box(self, symbols):
        """Create a GSMThermodynBox with elastic-damage material."""
        T, S, eps, sig, omega, Y, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'omega', 'Y', 'E'])
        
        # Elastic-damage Helmholtz free energy: F(T,ε,ω) = ½(1-ω)Eε²
        F_damage = sp.Rational(1, 2) * (1 - omega) * E * eps**2
        
        # Create the initial state function instance
        F_state_fn = GSMStateFn(
            fn_expr=F_damage,
            th_x_var=T,      # Temperature is natural variable
            th_y_var=S,      # Entropy is conjugate variable
            mc_x_var=eps,    # Strain is natural variable
            mc_y_var=sig,    # Stress is conjugate variable
            Eps_var=omega,   # Internal natural variable (damage)
            Sig_var=Y,       # Internal conjugate variable (stored energy)
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        return GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
    
    @pytest.fixture
    def thermal_expansion_box(self, symbols):
        """Create a GSMThermodynBox with thermal expansion material."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        alpha = Scalar(r'\alpha', codename='alpha', real=True)
        
        # Thermal expansion Helmholtz free energy: F(T,ε) = ½Eε² + αT²
        F_thermal = sp.Rational(1, 2) * E * eps**2 + alpha * T**2
        
        # Create the initial state function instance
        F_state_fn = GSMStateFn(
            fn_expr=F_thermal,
            th_x_var=T,      # Temperature is natural variable
            th_y_var=S,      # Entropy is conjugate variable
            mc_x_var=eps,    # Strain is natural variable
            mc_y_var=sig,    # Stress is conjugate variable
            Eps_var=alpha,   # Thermal parameter as internal variable
            Sig_var=sp.Symbol('alpha_conj'),  # Dummy conjugate
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        return GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
    
    def test_round_trip_F_to_G_to_F_elastic(self, elastic_box):
        """Test F→G→F round-trip for elastic material."""
        # Get original F
        F_original = elastic_box.get_current_state_function()
        F_original_expr = F_original.fn_expr
        
        # Transform F→G→F
        G_instance = elastic_box.legendre_transform(StateFunctionType.GIBBS)
        elastic_box.set_current_state_function(StateFunctionType.GIBBS)
        
        F_recovered_instance = elastic_box.legendre_transform(StateFunctionType.HELMHOLTZ)
        F_recovered_expr = F_recovered_instance.fn_expr
        
        # Check exact equality (difference = 0)
        difference = sp.simplify(F_recovered_expr - F_original_expr)
        assert difference == 0, f"Round-trip F→G→F failed. Difference: {difference}"
    
    def test_round_trip_F_to_U_to_F_elastic(self, elastic_box):
        """Test F→U→F round-trip for elastic material."""
        F_original = elastic_box.get_current_state_function()
        F_original_expr = F_original.fn_expr
        
        U_instance = elastic_box.legendre_transform(StateFunctionType.INTERNAL_ENERGY)
        elastic_box.set_current_state_function(StateFunctionType.INTERNAL_ENERGY)
        
        F_recovered_instance = elastic_box.legendre_transform(StateFunctionType.HELMHOLTZ)
        F_recovered_expr = F_recovered_instance.fn_expr
        
        difference = sp.simplify(F_recovered_expr - F_original_expr)
        assert difference == 0, f"Round-trip F→U→F failed. Difference: {difference}"
    
    def test_round_trip_G_to_H_to_G_elastic(self, elastic_box):
        """Test G→H→G round-trip for elastic material."""
        # First get G
        G_instance = elastic_box.legendre_transform(StateFunctionType.GIBBS)
        G_original_expr = G_instance.fn_expr
        elastic_box.set_current_state_function(StateFunctionType.GIBBS)
        
        H_instance = elastic_box.legendre_transform(StateFunctionType.ENTHALPY)
        elastic_box.set_current_state_function(StateFunctionType.ENTHALPY)
        
        G_recovered_instance = elastic_box.legendre_transform(StateFunctionType.GIBBS)
        G_recovered_expr = G_recovered_instance.fn_expr
        
        difference = sp.simplify(G_recovered_expr - G_original_expr)
        assert difference == 0, f"Round-trip G→H→G failed. Difference: {difference}"
    
    def test_round_trip_U_to_G_to_U_elastic(self, elastic_box):
        """Test U→G→U round-trip (diagonal transformation) for elastic material."""
        # First get U
        U_instance = elastic_box.legendre_transform(StateFunctionType.INTERNAL_ENERGY)
        U_original_expr = U_instance.fn_expr
        elastic_box.set_current_state_function(StateFunctionType.INTERNAL_ENERGY)
        
        G_instance = elastic_box.legendre_transform(StateFunctionType.GIBBS)
        elastic_box.set_current_state_function(StateFunctionType.GIBBS)
        
        U_recovered_instance = elastic_box.legendre_transform(StateFunctionType.INTERNAL_ENERGY)
        U_recovered_expr = U_recovered_instance.fn_expr
        
        difference = sp.simplify(U_recovered_expr - U_original_expr)
        assert difference == 0, f"Round-trip U→G→U failed. Difference: {difference}"
    
    def test_round_trip_H_to_F_to_H_elastic(self, elastic_box):
        """Test H→F→H round-trip (diagonal transformation) for elastic material."""
        # First get H
        H_instance = elastic_box.legendre_transform(StateFunctionType.ENTHALPY)
        H_original_expr = H_instance.fn_expr
        elastic_box.set_current_state_function(StateFunctionType.ENTHALPY)
        
        F_instance = elastic_box.legendre_transform(StateFunctionType.HELMHOLTZ)
        elastic_box.set_current_state_function(StateFunctionType.HELMHOLTZ)
        
        H_recovered_instance = elastic_box.legendre_transform(StateFunctionType.ENTHALPY)
        H_recovered_expr = H_recovered_instance.fn_expr
        
        difference = sp.simplify(H_recovered_expr - H_original_expr)
        assert difference == 0, f"Round-trip H→F→H failed. Difference: {difference}"
    
    def test_all_round_trips_damage_material(self, damage_box):
        """Test all round-trips for damage material (comprehensive test)."""
        # Get all state functions first
        F_orig_instance = damage_box.get_current_state_function()  # Start with F
        F_orig_expr = F_orig_instance.fn_expr
        
        # Get other state functions
        G_orig_instance = damage_box.legendre_transform(StateFunctionType.GIBBS)
        U_orig_instance = damage_box.legendre_transform(StateFunctionType.INTERNAL_ENERGY)
        H_orig_instance = damage_box.legendre_transform(StateFunctionType.ENTHALPY)
        
        G_orig_expr = G_orig_instance.fn_expr
        U_orig_expr = U_orig_instance.fn_expr
        H_orig_expr = H_orig_instance.fn_expr
        
        # Test key round-trips
        round_trips = [
            (StateFunctionType.HELMHOLTZ, StateFunctionType.GIBBS, StateFunctionType.HELMHOLTZ, F_orig_expr, F_orig_instance),
            (StateFunctionType.HELMHOLTZ, StateFunctionType.INTERNAL_ENERGY, StateFunctionType.HELMHOLTZ, F_orig_expr, F_orig_instance),
            (StateFunctionType.GIBBS, StateFunctionType.ENTHALPY, StateFunctionType.GIBBS, G_orig_expr, G_orig_instance),
        ]
        
        for start_fn, intermediate_fn, end_fn, original_expr, start_instance in round_trips:
            # Reset to start state
            damage_box.set_current_state_function(start_fn)
            
            # Perform round-trip
            intermediate_instance = damage_box.legendre_transform(intermediate_fn)
            damage_box.set_current_state_function(intermediate_fn)
            
            recovered_instance = damage_box.legendre_transform(end_fn)
            recovered_expr = recovered_instance.fn_expr
            
            # Check round-trip consistency
            difference = sp.simplify(recovered_expr - original_expr)
            assert difference == 0, f"Round-trip {start_fn.value}→{intermediate_fn.value}→{end_fn.value} failed. Difference: {difference}"
    
    def test_thermal_expansion_round_trips(self, thermal_expansion_box):
        """Test round-trips for thermal expansion material with temperature dependence."""
        F_original = thermal_expansion_box.get_current_state_function()
        F_original_expr = F_original.fn_expr
        
        # Test F→U→F round-trip (thermal transformation)
        U_instance = thermal_expansion_box.legendre_transform(StateFunctionType.INTERNAL_ENERGY)
        thermal_expansion_box.set_current_state_function(StateFunctionType.INTERNAL_ENERGY)
        
        F_recovered_instance = thermal_expansion_box.legendre_transform(StateFunctionType.HELMHOLTZ)
        F_recovered_expr = F_recovered_instance.fn_expr
        
        difference = sp.simplify(F_recovered_expr - F_original_expr)
        assert difference == 0, f"Thermal expansion F→U→F round-trip failed. Difference: {difference}"


class TestGSMThermodynBoxPropertyAccess:
    """Test property-based access and state function management."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': Scalar(r'\vartheta', codename='T', real=True, positive=True),
            'S': Scalar('S', codename='S', real=True),
            'eps': Scalar(r'\varepsilon', codename='eps', real=True),
            'sig': Scalar(r'\sigma', codename='sig', real=True),
            'omega': Scalar(r'\omega', codename='omega', real=True, positive=True),
            'Y': Scalar('Y', codename='Y', real=True),
            'E': Scalar('E', codename='E', positive=True)
        }
    
    @pytest.fixture
    def test_box(self, symbols):
        """Create a test box with damage material."""
        T, S, eps, sig, omega, Y, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'omega', 'Y', 'E'])
        
        F_damage = sp.Rational(1, 2) * (1 - omega) * E * eps**2
        
        F_state_fn = GSMStateFn(
            fn_expr=F_damage,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=omega, Sig_var=Y,
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        return GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
    
    def test_property_access_consistency(self, test_box):
        """Test that property access gives same results as explicit access."""
        # Get via properties
        F_prop = test_box.F
        G_prop = test_box.G
        U_prop = test_box.U
        H_prop = test_box.H
        
        # Get via explicit access
        F_explicit = test_box.get_state_function(StateFunctionType.HELMHOLTZ) or test_box.legendre_transform(StateFunctionType.HELMHOLTZ)
        G_explicit = test_box.get_state_function(StateFunctionType.GIBBS) or test_box.legendre_transform(StateFunctionType.GIBBS)
        U_explicit = test_box.get_state_function(StateFunctionType.INTERNAL_ENERGY) or test_box.legendre_transform(StateFunctionType.INTERNAL_ENERGY)
        H_explicit = test_box.get_state_function(StateFunctionType.ENTHALPY) or test_box.legendre_transform(StateFunctionType.ENTHALPY)
        
        # Check consistency
        assert sp.simplify(F_prop.fn_expr - F_explicit.fn_expr) == 0, "F property inconsistent"
        assert sp.simplify(G_prop.fn_expr - G_explicit.fn_expr) == 0, "G property inconsistent"
        assert sp.simplify(U_prop.fn_expr - U_explicit.fn_expr) == 0, "U property inconsistent"
        assert sp.simplify(H_prop.fn_expr - H_explicit.fn_expr) == 0, "H property inconsistent"
    
    def test_state_function_storage_and_retrieval(self, test_box):
        """Test that state functions are properly stored and retrievable."""
        # Initially only F should be available
        initial_functions = test_box.get_available_state_functions()
        assert StateFunctionType.HELMHOLTZ in initial_functions, "Initial F not available"
        
        # Transform to G and check it's stored
        G_instance = test_box.legendre_transform(StateFunctionType.GIBBS)
        available_after_G = test_box.get_available_state_functions()
        assert StateFunctionType.GIBBS in available_after_G, "G not stored after transformation"
        
        # Retrieve G and check it's the same instance
        G_retrieved = test_box.get_state_function(StateFunctionType.GIBBS)
        assert G_retrieved is G_instance, "Retrieved G is not the same instance"
    
    def test_current_state_function_management(self, test_box):
        """Test current state function setting and getting."""
        # Initially should be F (HELMHOLTZ)
        assert test_box.current_state_fn == StateFunctionType.HELMHOLTZ, "Initial current state function incorrect"
        
        # Change current state function
        test_box.legendre_transform(StateFunctionType.GIBBS)
        test_box.set_current_state_function(StateFunctionType.GIBBS)
        assert test_box.current_state_fn == StateFunctionType.GIBBS, "Current state function not updated to G"
        
        # Get current state function instance
        current_instance = test_box.get_current_state_function()
        G_instance = test_box.get_state_function(StateFunctionType.GIBBS)
        assert current_instance is G_instance, "Current state function instance incorrect"


class TestGSMThermodynBoxConstitutiveRelations:
    """Test constitutive relation consistency and correctness."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': Scalar(r'\vartheta', codename='T', real=True, positive=True),
            'S': Scalar('S', codename='S', real=True),
            'eps': Scalar(r'\varepsilon', codename='eps', real=True),
            'sig': Scalar(r'\sigma', codename='sig', real=True),
            'omega': Scalar(r'\omega', codename='omega', real=True, positive=True),
            'Y': Scalar('Y', codename='Y', real=True),
            'E': Scalar('E', codename='E', positive=True)
        }
    
    @pytest.fixture
    def test_box(self, symbols):
        """Create a test box with damage material."""
        T, S, eps, sig, omega, Y, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'omega', 'Y', 'E'])
        
        F_damage = sp.Rational(1, 2) * (1 - omega) * E * eps**2
        
        F_state_fn = GSMStateFn(
            fn_expr=F_damage,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=omega, Sig_var=Y,
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        return GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
    
    def test_stress_consistency_across_functions(self, test_box, symbols):
        """Test that stress derived from different state functions is consistent."""
        eps = symbols['eps']
        
        F_instance = test_box.F
        U_instance = test_box.U
        
        # Stress should be ∂F/∂ε = ∂U/∂ε
        sigma_from_F = sp.diff(F_instance.fn_expr, eps)
        sigma_from_U = sp.diff(U_instance.fn_expr, eps)
        
        difference = sp.simplify(sigma_from_F - sigma_from_U)
        assert difference == 0, f"Stress inconsistency between F and U: {difference}"
    
    def test_damage_force_consistency(self, test_box, symbols):
        """Test that damage force Y is consistent across state functions."""
        omega = symbols['omega']
        
        F_instance = test_box.F
        G_instance = test_box.G
        U_instance = test_box.U
        H_instance = test_box.H
        
        # Damage force: Y = -∂/∂ω for all state functions
        Y_from_F = -sp.diff(F_instance.fn_expr, omega)
        Y_from_U = -sp.diff(U_instance.fn_expr, omega)
        
        # Before variable substitution, damage forces should be identical
        assert sp.simplify(Y_from_F - Y_from_U) == 0, "Damage force F vs U inconsistent"
    
    def test_constitutive_relations_via_interface(self, test_box, symbols):
        """Test constitutive relations through the GSMStateFn interface."""
        F_instance = test_box.F
        
        # Get constitutive relations through the interface
        thermal_relation = F_instance.get_thermal_constitutive_relation()
        mechanical_relations = F_instance.get_mechanical_constitutive_relations()
        internal_relations = F_instance.get_internal_constitutive_relations()
        
        # Check that relations are returned (structure test)
        assert isinstance(thermal_relation, tuple), "Thermal relation not returned as tuple"
        assert len(thermal_relation) == 2, "Thermal relation tuple should have 2 elements"
        assert isinstance(thermal_relation[1], sp.Expr), "Thermal relation expression should be sympy expression"
        
        assert isinstance(mechanical_relations, list), "Mechanical relations not returned as list"
        assert len(mechanical_relations) > 0, "Mechanical relations list should not be empty"
        assert isinstance(mechanical_relations[0], tuple), "Mechanical relations should contain tuples"
        assert isinstance(mechanical_relations[0][1], sp.Expr), "Mechanical relation expression should be sympy expression"
        
        assert isinstance(internal_relations, list), "Internal relations not returned as list"
        assert len(internal_relations) > 0, "Internal relations list should not be empty"
        assert isinstance(internal_relations[0], tuple), "Internal relations should contain tuples"
        assert isinstance(internal_relations[0][1], sp.Expr), "Internal relation expression should be sympy expression"


class TestGSMThermodynBoxFrameworkValidation:
    """Test overall framework validation and edge cases."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': Scalar(r'\vartheta', codename='T', real=True, positive=True),
            'S': Scalar('S', codename='S', real=True),
            'eps': Scalar(r'\varepsilon', codename='eps', real=True),
            'sig': Scalar(r'\sigma', codename='sig', real=True),
            'E': Scalar('E', codename='E', positive=True)
        }
    
    def test_simple_elastic_material_known_results(self, symbols):
        """Test simple elastic material against known analytical results."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        # Simple elastic material: F(T,ε) = ½Eε²
        F_elastic = sp.Rational(1, 2) * E * eps**2
        
        F_state_fn = GSMStateFn(
            fn_expr=F_elastic,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        box = GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
        
        # Expected results for elastic material
        expected_sigma = E * eps  # σ = Eε (Hooke's law)
        expected_G_coeff = sp.Rational(-1, 2) / E  # G should have -½σ²/E term
        
        # Test constitutive relation
        actual_sigma = sp.diff(box.F.fn_expr, eps)
        assert sp.simplify(actual_sigma - expected_sigma) == 0, "Hooke's law not satisfied"
        
        # Test Gibbs function coefficient
        actual_G = box.G.fn_expr
        G_sigma2_coeff = actual_G.coeff(sig**2)
        if G_sigma2_coeff is not None:
            assert sp.simplify(G_sigma2_coeff - expected_G_coeff) == 0, "Gibbs function σ² coefficient incorrect"
    
    def test_compute_all_state_functions(self, symbols):
        """Test computing all four state functions."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        F_test = sp.Rational(1, 2) * E * eps**2
        F_state_fn = GSMStateFn(
            fn_expr=F_test,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        box = GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
        
        # Compute all state functions
        all_functions = box.compute_all_state_functions()
        
        # Check that all four functions are available
        expected_functions = {StateFunctionType.INTERNAL_ENERGY, StateFunctionType.HELMHOLTZ, 
                            StateFunctionType.ENTHALPY, StateFunctionType.GIBBS}
        actual_functions = set(all_functions.keys())
        
        assert actual_functions == expected_functions, f"Not all state functions computed. Got: {actual_functions}"
        
        # Check that all are valid instances
        for state_fn, instance in all_functions.items():
            assert hasattr(instance, 'fn_expr'), f"State function {state_fn} missing fn_expr"
            assert hasattr(instance, 'get_natural_variables'), f"State function {state_fn} missing interface methods"
    
    def test_state_function_accessibility(self, symbols):
        """Test that all four state functions are accessible."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        F_test = sp.Rational(1, 2) * E * eps**2
        F_state_fn = GSMStateFn(
            fn_expr=F_test,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        box = GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
        
        # All state functions should be accessible and return valid instances
        F_instance = box.F
        G_instance = box.G
        U_instance = box.U
        H_instance = box.H
        
        assert F_instance is not None, "Helmholtz function F not accessible"
        assert G_instance is not None, "Gibbs function G not accessible"
        assert U_instance is not None, "Internal energy U not accessible"
        assert H_instance is not None, "Enthalpy H not accessible"
        
        # They should be GSMStateFnIfc instances
        assert hasattr(F_instance, 'fn_expr'), "F is not a proper state function instance"
        assert hasattr(G_instance, 'fn_expr'), "G is not a proper state function instance"
        assert hasattr(U_instance, 'fn_expr'), "U is not a proper state function instance"
        assert hasattr(H_instance, 'fn_expr'), "H is not a proper state function instance"
    
    def test_transformation_graph(self, symbols):
        """Test transformation graph generation."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        F_test = sp.Rational(1, 2) * E * eps**2
        F_state_fn = GSMStateFn(
            fn_expr=F_test,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        box = GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
        
        # Get transformation graph
        graph = box.get_transformation_graph()
        
        # Check structure
        assert isinstance(graph, dict), "Transformation graph not returned as dict"
        for state_fn in StateFunctionType:
            assert state_fn in graph, f"State function {state_fn} not in graph"
            assert isinstance(graph[state_fn], list), f"Graph entry for {state_fn} not a list"
    
    def test_validation_methods(self, symbols):
        """Test built-in validation methods."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        F_test = sp.Rational(1, 2) * E * eps**2
        F_state_fn = GSMStateFn(
            fn_expr=F_test,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        box = GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
        
        # Test validation method
        validation_result = box.validate_thermodynamic_consistency()
        assert isinstance(validation_result, bool), "Validation should return boolean"
        assert validation_result == True, "Validation should pass for valid box"


class TestGSMThermodynBoxEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def symbols(self):
        """Provide standard symbols for all tests."""
        return {
            'T': Scalar(r'\vartheta', codename='T', real=True, positive=True),
            'S': Scalar('S', codename='S', real=True),
            'eps': Scalar(r'\varepsilon', codename='eps', real=True),
            'sig': Scalar(r'\sigma', codename='sig', real=True),
            'E': Scalar('E', codename='E', positive=True)
        }
    
    def test_linear_function_round_trips(self, symbols):
        """Test round-trips with linear function (edge case)."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        # Linear function: F(T,ε) = Eε (no quadratic term)
        F_linear = E * eps
        
        F_state_fn = GSMStateFn(
            fn_expr=F_linear,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        box = GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
        
        # Test F→G→F round-trip for linear case
        F_original = box.F.fn_expr
        
        G_instance = box.legendre_transform(StateFunctionType.GIBBS)
        box.set_current_state_function(StateFunctionType.GIBBS)
        
        F_recovered_instance = box.legendre_transform(StateFunctionType.HELMHOLTZ)
        F_recovered = F_recovered_instance.fn_expr
        
        difference = sp.simplify(F_recovered - F_original)
        assert difference == 0, f"Linear function round-trip failed: {difference}"
    
    def test_zero_function_round_trips(self, symbols):
        """Test round-trips with zero function (degenerate case)."""
        T, S, eps, sig = (symbols[k] for k in ['T', 'S', 'eps', 'sig'])
        
        # Zero function: F(T,ε) = 0
        F_zero = sp.sympify(0)
        
        F_state_fn = GSMStateFn(
            fn_expr=F_zero,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        box = GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn)
        
        # Test F→G→F round-trip for zero function
        F_original = box.F.fn_expr
        
        # Zero function should remain zero through transformations
        assert F_original == 0, "Zero function not preserved"
        
        # Perform round-trip
        G_instance = box.legendre_transform(StateFunctionType.GIBBS)
        box.set_current_state_function(StateFunctionType.GIBBS)
        
        F_recovered_instance = box.legendre_transform(StateFunctionType.HELMHOLTZ)
        F_recovered = F_recovered_instance.fn_expr
        
        assert F_recovered == 0, "Zero function round-trip failed"
    
    def test_invalid_state_function_setting(self, symbols):
        """Test error handling for invalid state function operations."""
        T, S, eps, sig, E = (symbols[k] for k in ['T', 'S', 'eps', 'sig', 'E'])
        
        F_test = sp.Rational(1, 2) * E * eps**2
        F_state_fn = GSMStateFn(
            fn_expr=F_test,
            th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
            Eps_var=sp.Symbol('eps_dummy'), Sig_var=sp.Symbol('sig_dummy'),
            state_function_type=StateFunctionType.HELMHOLTZ
        )
        
        box = GSMThermodynBox(
            initial_state_fn=StateFunctionType.HELMHOLTZ,
            initial_state_instance=F_state_fn
        )
        
        # Try to set current to unavailable state function
        with pytest.raises(ValueError, match="not available"):
            box.set_current_state_function(StateFunctionType.GIBBS)  # G not computed yet


@pytest.mark.integration
def test_comprehensive_gsm_thermodyn_box2_validation():
    """Integration test that runs a comprehensive validation."""
    # Import here to ensure proper module loading
    from bmcs_matmod.gsm_lagrange.core2.gsm_thermodyn_box import GSMThermodynBox
    from bmcs_matmod.gsm_lagrange.core2.gsm_state_fn import GSMStateFn, StateFunctionType
    from bmcs_matmod.gsm_lagrange.core2.gsm_vars import Scalar
    
    # Symbols
    T = Scalar(r'\vartheta', codename='T', real=True, positive=True)
    S = Scalar('S', codename='S', real=True)
    eps = Scalar(r'\varepsilon', codename='eps', real=True)
    sig = Scalar(r'\sigma', codename='sig', real=True)
    omega = Scalar(r'\omega', codename='omega', real=True, positive=True)
    Y = Scalar('Y', codename='Y', real=True)
    E = Scalar('E', codename='E', positive=True)
    
    # Elastic-damage model
    F_damage = sp.Rational(1, 2) * (1 - omega) * E * eps**2
    
    F_state_fn = GSMStateFn(
        fn_expr=F_damage,
        th_x_var=T, th_y_var=S, mc_x_var=eps, mc_y_var=sig,
        Eps_var=omega, Sig_var=Y,
        state_function_type=StateFunctionType.HELMHOLTZ
    )
    
    box = GSMThermodynBox(
        initial_state_fn=StateFunctionType.HELMHOLTZ,
        initial_state_instance=F_state_fn
    )
    
    # Get all state functions
    F_original = box.F.fn_expr
    G_original = box.G.fn_expr
    U_original = box.U.fn_expr
    H_original = box.H.fn_expr
    
    # Test key round-trips
    round_trip_tests = [
        (StateFunctionType.HELMHOLTZ, StateFunctionType.GIBBS, F_original),
        (StateFunctionType.HELMHOLTZ, StateFunctionType.INTERNAL_ENERGY, F_original),
        (StateFunctionType.GIBBS, StateFunctionType.ENTHALPY, G_original),
    ]
    
    for start_fn, intermediate_fn, original in round_trip_tests:
        # Set to start function
        box.set_current_state_function(start_fn)
        
        # Transform to intermediate
        intermediate_instance = box.legendre_transform(intermediate_fn)
        box.set_current_state_function(intermediate_fn)
        
        # Transform back
        recovered_instance = box.legendre_transform(start_fn)
        recovered = recovered_instance.fn_expr
        
        # Check round-trip
        difference = sp.simplify(recovered - original)
        assert difference == 0, f"Integration test: round-trip {start_fn.value}→{intermediate_fn.value}→{start_fn.value} failed, difference = {difference}"
    
    # Test constitutive relations
    box.set_current_state_function(StateFunctionType.HELMHOLTZ)  # Reset to clean state
    
    F_fresh = box.F.fn_expr
    G_fresh = box.G.fn_expr
    
    sigma_from_F = sp.diff(F_fresh, eps)
    epsilon_from_G = -sp.diff(G_fresh, sig)
    
    # Test inverse relationship
    epsilon_test = epsilon_from_G.subs(sig, sigma_from_F)
    epsilon_simplified = sp.simplify(epsilon_test)
    inverse_diff = sp.simplify(epsilon_simplified - eps)
    
    assert inverse_diff == 0, f"Integration test: inverse relationship failed, difference = {inverse_diff}"
    
    print("✓ Comprehensive GSMThermodynBox validation passed!")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
