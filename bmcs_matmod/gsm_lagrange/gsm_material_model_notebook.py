# To convert this file to a notebook, use:
# jupyter nbconvert --to notebook --execute gsm_material_model_notebook.py

# %%
"""
# Generalized Standard Material Model API Demonstration

This notebook demonstrates the usage of the `GSMMaterialModel` class, which provides a bridge between 
symbolic material definitions (using `GSMBase`) and executable numerical models with concrete parameter values.

We'll use the elastic-damage model (`GSM1DED`) as an example to show how simple it is to create, parametrize, and 
visualize material responses using this framework.
"""

# %%
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import traits.api as tr
import bmcs_utils.api as bu

# Import the GSM framework
from bmcs_matmod.gsm_lagrange.gsm_base import GSMBase
from bmcs_matmod.gsm_lagrange.gsm_mpdp import GSMMPDP
from bmcs_matmod.gsm_lagrange.gsm_material_model import GSMMaterialModel

# Import the specific model we'll use for demonstration
from bmcs_matmod.gsm_lagrange.gsm1d_ed import GSM1DED

# For nicer plot display
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
try:
    plt.style.use('bmcs')
except:
    pass

# For LaTeX rendering in plots
plt.rcParams['text.usetex'] = False

# %%
"""
## 1. Examining the Elastic-Damage Model

First, let's take a look at the `GSM1DED` class to understand its structure. This is a 1D elastic damage model implemented using the GSM framework.
"""

# %%
# Examine the GSM1DED class
gsm1d_ed = GSM1DED()
gsm1d_ed.print_potentials()

# %%
"""
## 2. Creating a Material Model from GSM1DED

Now, let's use the `GSMMaterialModel` class to create an executable model with specific parameter values. The `GSMMaterialModel` class automatically:

1. Analyzes the symbolic model structure
2. Creates traits for all parameters  
3. Provides methods for numerical simulation

This allows us to work with material models in a more intuitive way, defining parameter values directly and running simulations without managing the symbolic-to-numerical conversion manually.
"""

# %%
# Create a material model with specific parameter values
ed_material = GSMMaterialModel(
    gsm_model_type=GSM1DED,
    E=20000.0,      # Young's modulus (MPa)
    omega_0=0.01,   # Damage threshold strain
    omega_1=0.2,    # Ultimate strain
    kappa_0=0.0,    # Initial internal variable
)

# Examine the traits that were automatically created
print("Available parameters:")
for param_sym, name in ed_material.trait_model_params.items():
    value = getattr(ed_material, name)
    print(f"  {name} = {value} ({param_sym})")

# %%
"""
## 3. Monotonic Tension Test

Let's simulate a monotonic tension test to see how our material behaves. This will demonstrate the strain-softening behavior characteristic of damage mechanics.
"""

# %%
# Define a monotonic tensile strain history
n_steps = 100
strain_max = 0.006  # Maximum strain
strain = np.linspace(0, strain_max, n_steps)
time = np.linspace(0, 1.0, n_steps)

# Run the simulation with our material model
response = ed_material.get_response(strain, time)

# Unpack the results
t_t, eps_ta, sig_ta, Eps_t, Sig_t, iter_t, lam_t, (d_t_t, d_eps_ta) = response

# Extract and reshape for easier plotting
eps = eps_ta[:, 0]
sig = sig_ta[:, 0]
kappa = Eps_t[:, 0]  # First internal variable (damage variable)

# %%
# Plot the stress-strain curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Stress-strain curve
ax1.plot(eps, sig, 'b-', linewidth=2)
ax1.set_xlabel('Strain $\\varepsilon$')
ax1.set_ylabel('Stress $\\sigma$ (MPa)')
ax1.set_title('Stress-Strain Curve (Tensile Test)')
ax1.grid(True)

# Add markers for key points
ax1.axvline(x=ed_material.omega_0, color='r', linestyle='--', 
            label=f'Damage threshold: {ed_material.omega_0}')
ax1.axvline(x=ed_material.omega_1, color='g', linestyle='--', 
            label=f'Ultimate strain: {ed_material.omega_1}')
ax1.legend()

# Evolution of internal variable (kappa - damage)
ax2.plot(eps, kappa, 'g-', linewidth=2)
ax2.set_xlabel('Strain $\\varepsilon$')
ax2.set_ylabel('Internal variable $\\kappa$')
ax2.set_title('Evolution of Damage Variable')
ax2.grid(True)
ax2.axvline(x=ed_material.omega_0, color='r', linestyle='--')
ax2.axvline(x=ed_material.omega_1, color='g', linestyle='--')

plt.tight_layout()
plt.show()

# %%
"""
## 4. Computing the Damage Function

The elastic-damage model typically includes a damage function that reduces the stiffness based on the internal variable. Let's calculate and visualize this function.
"""

# %%
# Calculate the damage function omega(kappa)
kappa_range = np.linspace(0, 0.25, 100)
omega_values = np.zeros_like(kappa_range)

# For each kappa value, calculate omega based on the model's parameters
for i, k in enumerate(kappa_range):
    if k <= ed_material.omega_0:
        omega_values[i] = 0.0
    elif k >= ed_material.omega_1:
        omega_values[i] = 1.0
    else:
        # Linear damage evolution between omega_0 and omega_1
        omega_values[i] = (k - ed_material.omega_0) / (ed_material.omega_1 - ed_material.omega_0)

# Plot the damage function
plt.figure(figsize=(10, 6))
plt.plot(kappa_range, omega_values, 'r-', linewidth=2)
plt.xlabel('Internal variable $\\kappa$')
plt.ylabel('Damage $\\omega$')
plt.title('Damage Function $\\omega(\\kappa)$')
plt.grid(True)
plt.axvline(x=ed_material.omega_0, color='b', linestyle='--', 
            label=f'Damage threshold: {ed_material.omega_0}')
plt.axvline(x=ed_material.omega_1, color='g', linestyle='--', 
            label=f'Ultimate damage: {ed_material.omega_1}')
plt.legend()
plt.tight_layout()
plt.show()

# %%
"""
## 5. Parameter Sensitivity Analysis

One of the key advantages of the `GSMMaterialModel` approach is the ease of creating models with different parameter values. Let's leverage this to perform a parameter sensitivity analysis.
"""

# %%
# Create several material models with different ultimate strain values
omega_1_values = [0.1, 0.2, 0.3]
ed_materials = [
    GSMMaterialModel(
        gsm_model_type=GSM1DED,
        E=20000.0,
        omega_0=0.01,
        omega_1=omega_1,
        kappa_0=0.0
    )
    for omega_1 in omega_1_values
]

# Compute responses for each material model
responses = []
for material in ed_materials:
    response = material.get_response(strain, time)
    responses.append(response)

# Plot the stress-strain curves for comparison
plt.figure(figsize=(12, 8))

for i, response in enumerate(responses):
    _, eps_ta, sig_ta, _, _, _, _, _ = response
    eps = eps_ta[:, 0]
    sig = sig_ta[:, 0]
    plt.plot(eps, sig, linewidth=2, 
             label=f'$\\omega_1 = {omega_1_values[i]}$')

plt.xlabel('Strain $\\varepsilon$')
plt.ylabel('Stress $\\sigma$ (MPa)')
plt.title('Effect of Ultimate Strain on Material Response')
plt.grid(True)
plt.legend()
plt.axvline(x=ed_material.omega_0, color='r', linestyle='--', 
            label=f'Damage threshold: {ed_material.omega_0}')
plt.tight_layout()
plt.show()

# %%
"""
## 6. Cyclic Loading Test

Now let's test the model's behavior under cyclic loading to see how damage accumulates during load-unload cycles.
"""

# %%
# Define a cyclic strain history
n_cycles = 3
n_points = 200
time_cyclic = np.linspace(0, n_cycles, n_points)

# Create a saw-tooth strain pattern that goes from 0 to strain_max
strain_max = 0.005
strain_cyclic = np.zeros(n_points)

# For each cycle, increase the strain to a higher peak
cycle_duration = n_points // n_cycles
for i in range(n_cycles):
    cycle_strain_max = (i + 1) * strain_max / n_cycles
    cycle_start = i * cycle_duration
    cycle_mid = cycle_start + cycle_duration // 2
    cycle_end = (i + 1) * cycle_duration if i < n_cycles - 1 else n_points
    
    # Loading phase
    strain_cyclic[cycle_start:cycle_mid] = np.linspace(0, cycle_strain_max, cycle_mid - cycle_start)
    
    # Unloading phase
    strain_cyclic[cycle_mid:cycle_end] = np.linspace(cycle_strain_max, 0, cycle_end - cycle_mid)

# Reset our material model
ed_material = GSMMaterialModel(
    gsm_model_type=GSM1DED,
    E=20000.0,
    omega_0=0.01,
    omega_1=0.2,
    kappa_0=0.0
)

# Compute the response
response_cyclic = ed_material.get_response(strain_cyclic, time_cyclic)
t_t, eps_ta, sig_ta, Eps_t, Sig_t, iter_t, lam_t, (d_t_t, d_eps_ta) = response_cyclic

# Extract data for plotting
eps_cyclic = eps_ta[:, 0]
sig_cyclic = sig_ta[:, 0]
kappa_cyclic = Eps_t[:, 0]

# %%
# Plot the cyclic response
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Stress-strain hysteresis curve
ax1.plot(eps_cyclic, sig_cyclic, 'b-', linewidth=2)
ax1.set_xlabel('Strain $\\varepsilon$')
ax1.set_ylabel('Stress $\\sigma$ (MPa)')
ax1.set_title('Stress-Strain Hysteresis Loops')
ax1.grid(True)
ax1.axvline(x=ed_material.omega_0, color='r', linestyle='--', 
            label=f'Damage threshold: {ed_material.omega_0}')
ax1.legend()

# Evolution of strain and stress over time
ax2.plot(time_cyclic, strain_cyclic, 'g-', label='Strain', linewidth=2)
ax2.set_xlabel('Time $t$')
ax2.set_ylabel('Strain $\\varepsilon$')
ax2.set_title('Evolution of Strain and Stress Over Time')
ax2.grid(True)

# Add stress on a secondary y-axis
ax2b = ax2.twinx()
ax2b.plot(time_cyclic, sig_cyclic, 'r-', label='Stress', linewidth=2)
ax2b.set_ylabel('Stress $\\sigma$ (MPa)')

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Evolution of damage variable over time
ax3.plot(time_cyclic, kappa_cyclic, 'k-', linewidth=2)
ax3.set_xlabel('Time $t$')
ax3.set_ylabel('Internal Variable $\\kappa$')
ax3.set_title('Evolution of Damage Variable')
ax3.grid(True)

plt.tight_layout()
plt.show()

# %%
"""
## 7. Modulus Degradation

In damage mechanics, one of the key aspects is the degradation of the elastic modulus due to damage. Let's visualize how the effective modulus changes with increasing strain.
"""

# %%
# Calculate the effective modulus as a function of strain
effective_modulus = np.zeros_like(eps)
undamaged_modulus = ed_material.E

for i in range(len(eps)):
    # The effective modulus is E * (1-ω)
    # where ω is the damage variable, which is related to κ
    if kappa[i] <= ed_material.omega_0:
        effective_modulus[i] = undamaged_modulus
    elif kappa[i] >= ed_material.omega_1:
        effective_modulus[i] = 0.0
    else:
        damage = (kappa[i] - ed_material.omega_0) / (ed_material.omega_1 - ed_material.omega_0)
        effective_modulus[i] = undamaged_modulus * (1 - damage)

# Plot the degradation of elastic modulus
plt.figure(figsize=(10, 6))
plt.plot(eps, effective_modulus, 'b-', linewidth=2)
plt.xlabel('Strain $\\varepsilon$')
plt.ylabel('Effective Modulus $E_{\\text{eff}}$ (MPa)')
plt.title('Degradation of Elastic Modulus with Increasing Strain')
plt.grid(True)
plt.axvline(x=ed_material.omega_0, color='r', linestyle='--', 
           label=f'Damage threshold: {ed_material.omega_0}')
plt.axhline(y=undamaged_modulus, color='g', linestyle='--',
           label=f'Initial modulus: {undamaged_modulus} MPa')
plt.legend()
plt.tight_layout()
plt.show()

# %%
"""
## 8. Summary and Advantages of the GSMMaterialModel API

The `GSMMaterialModel` class provides several key advantages:

1. **Seamless Bridge Between Symbolic and Numerical**:
   - No need to manage the symbolic-to-numerical conversion manually
   - Clear separation between model definition (GSMBase subclass) and parameter values

2. **Trait-Based Parameter System**:
   - All parameters are automatically converted to traits
   - Validation, documentation, and UI integration come for free

3. **Simplified Parameter Studies**:
   - Easy creation of models with different parameter values
   - Straightforward comparison of material responses
   - Support for visualization and analysis

4. **Compatibility with Existing Models**:
   - Works with any properly defined GSMBase subclass
   - No need to modify existing model implementations

This approach makes it significantly easier for researchers and engineers to create, parametrize, and explore material models defined within the Generalized Standard Material framework.
"""

# %%
"""
## 9. Using the GSMMaterialModel in a Larger System

In practice, the GSMMaterialModel can be integrated into larger simulations or parameter identification systems. Here's a simple example of how it might be used as a component in a structural simulation.
"""

# %%
# Define a simple structural model that uses the material model
class SimpleBeam:
    def __init__(self, material_model, length=1.0, area=1.0):
        self.material = material_model
        self.length = length
        self.area = area
        
    def compute_extension(self, force, n_increments=10):
        """Calculate the extension of the beam under force"""
        # Convert force to stress
        stress_max = force / self.area
        
        # Create a stress history with incremental loading
        stress = np.linspace(0, stress_max, n_increments)
        time = np.linspace(0, 1.0, n_increments)
        
        # Use Gibbs formulation (stress-driven) to get strain
        try:
            t_t, sig_ta, eps_ta, Eps_t, Sig_t, iter_t, lam_t, (d_t_t, d_sig_ta) = (
                self.material.get_G_response(stress, time)
            )
            
            # Get the final strain
            final_strain = eps_ta[-1, 0]
            
            # Calculate extension
            extension = final_strain * self.length
            
            return extension, eps_ta, sig_ta
        except Exception as e:
            print(f"Error in beam calculation: {e}")
            return 0, np.zeros_like(stress), stress
        
# Create a beam with our material model
beam = SimpleBeam(
    material_model=ed_material,
    length=1.0,  # meters
    area=0.01    # square meters
)

# Calculate extension for different forces
forces = np.linspace(0, 120, 10)  # kN
extensions = []
strain_histories = []
stress_histories = []

for force in forces:
    extension, strain_history, stress_history = beam.compute_extension(force)
    extensions.append(extension)
    strain_histories.append(strain_history)
    stress_histories.append(stress_history)

# Plot force-extension curve
plt.figure(figsize=(10, 6))
plt.plot(extensions, forces, 'bo-', linewidth=2)
plt.xlabel('Extension (m)')
plt.ylabel('Force (kN)')
plt.title('Force-Extension Behavior of Beam with Damage Material')
plt.grid(True)
plt.tight_layout()
plt.show()
