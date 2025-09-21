"""
Basic Unit Database for GSM Variables

This module provides a lookup table mapping common codenames used in 
thermodynamic and mechanical modeling to their base SI units.

The database covers:
- Thermodynamic variables (temperature, entropy, heat capacity, etc.)
- Mechanical variables (stress, strain, displacement, force, etc.) 
- Material properties (elastic modulus, thermal expansion, etc.)
- Damage and internal variables
- Geometric properties

Base SI Units Used:
- Length: m (meter)
- Mass: kg (kilogram)
- Time: s (second)
- Temperature: K (Kelvin)
- Amount of substance: mol (mole)
- Electric current: A (ampere)
- Luminous intensity: cd (candela)

Derived units are expressed in terms of these base units.
"""

from typing import Dict

# Base SI unit lookup table
BASIC_UNIT_DB: Dict[str, str] = {
    # Thermodynamic variables
    'T': 'K',                    # Temperature
    'temperature': 'K',          # Temperature  
    'S': 'J⋅K⁻¹',               # Entropy
    'entropy': 'J⋅K⁻¹',         # Entropy
    'U': 'J',                    # Internal energy
    'internal_energy': 'J',      # Internal energy
    'H': 'J',                    # Enthalpy
    'enthalpy': 'J',             # Enthalpy
    'F': 'J',                    # Helmholtz free energy
    'helmholtz': 'J',            # Helmholtz free energy
    'G': 'J',                    # Gibbs free energy
    'gibbs': 'J',                # Gibbs free energy
    'C_v': 'J⋅K⁻¹',             # Heat capacity at constant volume
    'C_p': 'J⋅K⁻¹',             # Heat capacity at constant pressure
    'C_eps': 'J⋅K⁻¹',           # Heat capacity
    'heat_capacity': 'J⋅K⁻¹',    # Heat capacity
    'Q': 'J',                    # Heat
    'heat': 'J',                 # Heat
    
    # Mechanical variables - stress and strain
    'sig': 'Pa',                 # Stress (Pascal = N⋅m⁻² = kg⋅m⁻¹⋅s⁻²)
    'sigma': 'Pa',               # Stress
    'stress': 'Pa',              # Stress
    'tau': 'Pa',                 # Shear stress
    'shear_stress': 'Pa',        # Shear stress
    'eps': '1',                  # Strain (dimensionless)
    'epsilon': '1',              # Strain
    'strain': '1',               # Strain
    'gamma': '1',                # Shear strain
    'shear_strain': '1',         # Shear strain
    
    # Mechanical variables - displacement and force
    'u': 'm',                    # Displacement
    'displacement': 'm',         # Displacement
    'v': 'm⋅s⁻¹',               # Velocity
    'velocity': 'm⋅s⁻¹',         # Velocity
    'a': 'm⋅s⁻²',               # Acceleration
    'acceleration': 'm⋅s⁻²',     # Acceleration
    'F': 'N',                    # Force (Newton = kg⋅m⋅s⁻²)
    'force': 'N',                # Force
    'P': 'N',                    # Load/Force
    'load': 'N',                 # Load
    
    # Material properties - elastic
    'E': 'Pa',                   # Young's modulus
    'youngs_modulus': 'Pa',      # Young's modulus
    'G': 'Pa',                   # Shear modulus
    'shear_modulus': 'Pa',       # Shear modulus
    'K': 'Pa',                   # Bulk modulus
    'bulk_modulus': 'Pa',        # Bulk modulus
    'nu': '1',                   # Poisson's ratio (dimensionless)
    'poisson_ratio': '1',        # Poisson's ratio
    'lambda': 'Pa',              # Lamé parameter
    'lame_parameter': 'Pa',      # Lamé parameter
    'mu': 'Pa',                  # Second Lamé parameter (shear modulus)
    'kappa': 'Pa',               # Bulk modulus alternative notation
    
    # Material properties - thermal
    'alpha': 'K⁻¹',              # Thermal expansion coefficient
    'thermal_expansion': 'K⁻¹',  # Thermal expansion coefficient
    'k': 'W⋅m⁻¹⋅K⁻¹',          # Thermal conductivity
    'thermal_conductivity': 'W⋅m⁻¹⋅K⁻¹',  # Thermal conductivity
    'rho': 'kg⋅m⁻³',            # Density
    'density': 'kg⋅m⁻³',         # Density
    'c': 'J⋅kg⁻¹⋅K⁻¹',          # Specific heat capacity
    'specific_heat': 'J⋅kg⁻¹⋅K⁻¹',  # Specific heat capacity
    
    # Damage and internal variables
    'omega': '1',                # Damage parameter (dimensionless)
    'damage': '1',               # Damage parameter
    'D': '1',                    # Damage variable
    'kappa': '1',                # Hardening variable
    'hardening': '1',            # Hardening variable
    'Y': 'J⋅m⁻³',               # Damage driving force (energy density)
    'damage_force': 'J⋅m⁻³',     # Damage driving force
    'R': 'Pa',                   # Resistance/back-stress
    'resistance': 'Pa',          # Resistance
    'X': 'Pa',                   # Back-stress
    'back_stress': 'Pa',         # Back-stress
    
    # Geometric properties
    'A': 'm²',                   # Area
    'area': 'm²',                # Area
    'V': 'm³',                   # Volume
    'volume': 'm³',              # Volume
    'L': 'm',                    # Length
    'length': 'm',               # Length
    'h': 'm',                    # Height/thickness
    'height': 'm',               # Height
    'thickness': 'm',            # Thickness
    'R': 'm',                    # Radius
    'radius': 'm',               # Radius
    'I': 'm⁴',                   # Second moment of area
    'moment_of_inertia': 'm⁴',   # Second moment of area
    
    # Time and reference values
    't': 's',                    # Time
    'time': 's',                 # Time
    'T_0': 'K',                  # Reference temperature
    'T_ref': 'K',                # Reference temperature
    'eps_0': '1',                # Reference strain
    'sig_0': 'Pa',               # Reference stress
    
    # Dimensionless parameters
    'phi': '1',                  # Phase field variable
    'phase': '1',                # Phase variable
    'xi': '1',                   # Normalized coordinate
    'eta': '1',                  # Normalized coordinate
    'zeta': '1',                 # Normalized coordinate
    
    # Energy densities
    'psi': 'J⋅m⁻³',             # Helmholtz free energy density
    'w': 'J⋅m⁻³',               # Strain energy density
    'strain_energy_density': 'J⋅m⁻³',  # Strain energy density
}


def get_base_unit(codename: str) -> str:
    """
    Get the base SI unit for a given codename.
    
    Args:
        codename: Variable codename to look up
        
    Returns:
        str: Base SI unit string, or '1' (dimensionless) if not found
        
    Example:
        >>> get_base_unit('sig')
        'Pa'
        >>> get_base_unit('eps')  
        '1'
        >>> get_base_unit('unknown')
        '1'
    """
    return BASIC_UNIT_DB.get(codename, '1')


def add_unit_mapping(codename: str, unit: str) -> None:
    """
    Add a new codename-unit mapping to the database.
    
    Args:
        codename: Variable codename
        unit: SI unit string
        
    Example:
        >>> add_unit_mapping('my_var', 'kg⋅s⁻¹')
    """
    BASIC_UNIT_DB[codename] = unit


def get_all_units() -> Dict[str, str]:
    """
    Get a copy of the entire unit database.
    
    Returns:
        Dict[str, str]: Copy of the unit lookup table
    """
    return BASIC_UNIT_DB.copy()


def list_available_codenames() -> list:
    """
    Get list of all available codenames in the database.
    
    Returns:
        list: Sorted list of available codenames
    """
    return sorted(BASIC_UNIT_DB.keys())