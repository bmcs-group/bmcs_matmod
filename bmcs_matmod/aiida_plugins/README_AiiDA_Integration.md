# GSM AiiDA Integration

This document describes the AiiDA integration for BMCS material modeling with GSM (Generalized Standard Material) definitions.

## Overview

The AiiDA integration provides:
- **Calculation plugins** for running GSM simulations through AiiDA
- **Data types** for material parameters and loading histories
- **Workchains** for complex material characterization workflows
- **Parsers** for processing simulation results
- **Export tools** for data analysis and visualization

## Installation

### Prerequisites

- AiiDA core >= 2.6.0
- bmcs_matmod package with GSM CLI
- Python >= 3.8

### Setup

```bash
# Install with AiiDA support
pip install bmcs_matmod[aiida]

# Or install AiiDA separately
pip install aiida-core>=2.6.0 aiida-shell

# Set up AiiDA profile (if not already done)
verdi presto

# Verify plugin installation
verdi plugin list aiida.calculations | grep gsm
verdi plugin list aiida.workflows | grep gsm
```

## Entry Points

The package provides the following AiiDA entry points:

### Calculations

- `gsm.simulation`: Core GSM simulation calculation

### Parsers

- `gsm.parser`: Parser for GSM simulation outputs

### Workchains

- `gsm.monotonic`: Monotonic loading characterization
- `gsm.fatigue`: Single-level fatigue testing
- `gsm.sn_curve`: S-N curve construction workflow

### Data Types

- `gsm.material`: Material parameter data with validation
- `gsm.loading`: Loading history data

## Architecture

### GSM Calculation
The `GSMSimulationCalculation` provides the core interface to the GSM CLI:

```python
from aiida.plugins import CalculationFactory
GSMCalculation = CalculationFactory('gsm.simulation')

# Set up calculation
inputs = {
    'code': gsm_code,
    'gsm_model': orm.Str('GSM1D_ED'),
    'formulation': orm.Str('F'),
    'material_parameters': orm.Dict(dict=material_params),
    'loading_data': orm.Dict(dict=loading_data)
}

# Submit calculation
calc = engine.submit(GSMCalculation, **inputs)
```

### Workchains

#### Monotonic Loading (`gsm_monotonic`)
Performs monotonic loading characterization:
- Generates strain-controlled loading history
- Executes GSM simulation
- Extracts stress-strain response

```python
from aiida.plugins import WorkflowFactory
MonotonicWC = WorkflowFactory('gsm.monotonic')

inputs = {
    'gsm_code': code,
    'gsm_model': orm.Str('GSM1D_ED'),
    'formulation': orm.Str('F'), 
    'material_parameters': orm.Dict(dict=params),
    'max_strain': orm.Float(0.01),
    'num_steps': orm.Int(100)
}

workchain = engine.submit(MonotonicWC, **inputs)
```

#### Fatigue Testing (`gsm_fatigue`)
Performs stress-controlled fatigue testing:
- Generates cyclic stress history
- Monitors for failure criteria
- Determines cycles to failure

```python
FatigueWC = WorkflowFactory('gsm.fatigue')

inputs = {
    'gsm_code': code,
    'gsm_model': orm.Str('GSM1D_VED'),
    'material_parameters': orm.Dict(dict=params),
    'stress_amplitude': orm.Float(150.0),
    'stress_mean': orm.Float(50.0),
    'max_cycles': orm.Int(10000),
    'failure_strain': orm.Float(0.05)
}

workchain = engine.submit(FatigueWC, **inputs)
```

#### S-N Curve Construction (`gsm_sn_curve`)
Constructs complete S-N curves:
- Runs multiple fatigue tests in parallel
- Collects cycles-to-failure data
- Constructs S-N relationship

```python
SNCurveWC = WorkflowFactory('gsm.sn_curve')

stress_levels = [200, 175, 150, 125, 100, 75, 50]
inputs = {
    'gsm_code': code,
    'gsm_model': orm.Str('GSM1D_VED'),
    'material_parameters': orm.Dict(dict=params),
    'stress_levels': orm.List(list=stress_levels),
    'max_cycles': orm.Int(10000)
}

workchain = engine.submit(SNCurveWC, **inputs)
```

## Data Management

### Material Parameters
```python
# Create material parameter data with validation
material_data = GSMMaterialData(
    parameters={'E': 30000, 'S': 1.0, 'c': 2.0},
    model='GSM1D_ED'
)
material_data.validate_parameters()
```

### Loading Histories
```python
# Create loading history data
import numpy as np

time_array = np.linspace(0, 1, 100)
strain_history = np.linspace(0, 0.01, 100)

loading_data = GSMLoadingData({
    'time_array': time_array,
    'strain_history': strain_history,
    'loading_type': 'monotonic_tension'
})
```

## Code Setup

### Setting up GSM CLI Code
```python
# Create computer (if needed)
computer = orm.Computer(
    label='localhost',
    hostname='localhost',
    transport_type='core.local',
    scheduler_type='core.direct'
).store()

# Create GSM CLI code
code = orm.Code(
    input_plugin_name='gsm.simulation',
    remote_computer_uuid=computer.uuid,
    remote_absolute_path='/usr/local/bin/gsm-cli',
    label='gsm-cli'
).store()
```

### Remote Execution
For remote execution, configure appropriate transport and scheduler:

```python
# For SSH remote execution
computer = orm.Computer(
    label='hpc_cluster',
    hostname='hpc.example.com',
    transport_type='core.ssh',
    scheduler_type='core.slurm'  # or appropriate scheduler
)

# Configure SSH settings
computer.configure(
    username='your_username',
    port=22,
    look_for_keys=True
)
```

## Monitoring and Analysis

### Monitoring Workchains
```bash
# List running processes
verdi process list

# Show process details
verdi process show <pk>

# Monitor workchain progress
verdi process watch <pk>
```

### Analyzing Results
```python
# Load completed workchain
from aiida import orm
workchain = orm.load_node(<pk>)

# Access results
if workchain.is_finished_ok:
    results = workchain.outputs.monotonic_results
    print(results.get_dict())
    
    # Access array data
    if 'stress_strain_curve' in workchain.outputs:
        arrays = workchain.outputs.stress_strain_curve
        stress = arrays.get_array('stress')
        strain = arrays.get_array('strain')
```

### Data Export
```python
from bmcs_matmod.aiida_plugins.exporters import GSMJSONExporter

# Export results to JSON
GSMJSONExporter.export_simulation_results(
    results_node, 
    'simulation_results.json'
)

# Export S-N curve
GSMJSONExporter.export_sn_curve(
    sn_curve_node, 
    'sn_curve.csv', 
    format='csv'
)
```

## Integration with External Tools

### Jupyter Widgets
The AiiDA workchains can be integrated with Jupyter widgets for interactive material characterization:

```python
# In Jupyter notebook
from ipywidgets import interact, FloatSlider
from aiida import engine

@interact(max_strain=FloatSlider(min=0.001, max=0.1, step=0.001, value=0.01))
def run_monotonic_test(max_strain):
    inputs = {
        'gsm_code': code,
        'gsm_model': orm.Str('GSM1D_ED'),
        'material_parameters': orm.Dict(dict=current_params),
        'max_strain': orm.Float(max_strain)
    }
    
    workchain = engine.submit(MonotonicWC, **inputs)
    print(f"Submitted workchain: {workchain.pk}")
```

### Database Integration
Results are automatically stored in the AiiDA database with full provenance:

```python
# Query all fatigue tests for a material
qb = orm.QueryBuilder()
qb.append(GSMFatigueWorkChain, tag='fatigue')
qb.append(orm.Dict, with_incoming='fatigue', filters={
    'attributes.material_parameters.E': 30000
})

for workchain, results in qb.all():
    print(f"Fatigue test {workchain.pk}: {results.get_dict()}")
```

## Best Practices

1. **Parameter Validation**: Always validate material parameters before submission
2. **Resource Management**: Set appropriate computational resources for each calculation
3. **Parallel Execution**: Use S-N curve workchains for efficient parallel fatigue testing
4. **Data Organization**: Use meaningful labels and descriptions for workchains
5. **Result Caching**: Leverage AiiDA's caching to avoid duplicate calculations
6. **Error Handling**: Monitor workchains for failures and implement retry logic

## Troubleshooting

### Common Issues

1. **Code not found**: Ensure GSM CLI is properly installed and code path is correct
2. **Parameter validation errors**: Check material parameters against GSM model specifications
3. **Memory issues**: Adjust computational resources for large simulations
4. **Network timeouts**: Configure appropriate timeouts for remote execution

### Debugging
```bash
# Check calculation details
verdi calcjob gotocomputer <pk>

# View calculation files
verdi calcjob outputcat <pk>

# Check parser output
verdi calcjob res <pk>
```

## Recent Fixes and Known Issues

### Version 2.6+ Compatibility Updates

The AiiDA integration has been updated for compatibility with AiiDA 2.6+. If you encounter issues with older installations:

1. **Plugin Loading Errors**: Make sure you're using the correct plugin names with dots (e.g., `gsm.simulation` not `gsm_simulation`)

2. **Deprecated API Warnings**: The data types now use the modern `base.attributes.set/get` API instead of deprecated `set_attribute/get_attribute`

3. **Import Errors in CLI**: The GSM CLI now uses relative imports. If you see import errors, reinstall in development mode:
   ```bash
   pip uninstall bmcs_matmod
   pip install -e .[aiida]
   ```

4. **Entry Point Discovery**: The validation script now uses modern Python packaging APIs. If you encounter `pkg_resources` warnings, this is expected and handled automatically.

### Validation and Testing

Always run the validation script after installation or updates:

```bash
cd bmcs_matmod/aiida_plugins/
python validate_aiida_installation.py
```

For detailed explanations of fixes and troubleshooting, see `FIXES_EXPLAINED.md` in this directory.

## Testing the AiiDA Integration

### Interactive Testing with Jupyter Notebook

A comprehensive test notebook is provided to demonstrate and test all AiiDA integration features:

```bash
# Navigate to the aiida_plugins directory
cd bmcs_matmod/bmcs_matmod/aiida_plugins/

# Open the test notebook in VS Code
code test_gsm_aiida_integration.ipynb

# Or open with Jupyter
jupyter notebook test_gsm_aiida_integration.ipynb
```

The test notebook includes:

- **Setup verification**: Check AiiDA profile and plugin loading
- **Mock demonstrations**: Run examples even without full AiiDA setup
- **Computer and code setup**: Configure GSM CLI for AiiDA execution
- **Workchain testing**: Test monotonic, fatigue, and S-N curve workchains
- **Result visualization**: Plot stress-strain curves and S-N diagrams
- **Data export**: Export results in JSON and CSV formats
- **Monitoring examples**: Show how to track workchain progress

### Running the Test Notebook

1. **Prerequisites Setup**:
   ```bash
   # Install AiiDA with bmcs_matmod
   pip install bmcs_matmod[aiida]
   
   # Set up AiiDA profile
   verdi quicksetup
   ```

2. **Open in VS Code**:
   - Open VS Code in the project directory
   - Navigate to `bmcs_matmod/aiida_plugins/test_gsm_aiida_integration.ipynb`
   - Select Python kernel with AiiDA installed
   - Run cells sequentially

3. **Expected Outputs**:
   - Plugin availability check
   - Mock workchain submissions and results
   - Stress-strain curve plots
   - S-N curve construction and fitting
   - Data export files in `gsm_aiida_exports/` directory

### Manual Plugin Verification

```bash
# Check plugin registration
verdi plugin list aiida.calculations | grep gsm
verdi plugin list aiida.workflows | grep gsm
verdi plugin list aiida.parsers | grep gsm

# Test code setup
verdi code test gsm-cli@localhost

# List available computers
verdi computer list
```

### Automated Validation

Use the provided validation script to check your installation:

```bash
# Run comprehensive validation
cd bmcs_matmod/aiida_plugins/
python validate_aiida_installation.py

# Run with debug output
python validate_aiida_installation.py --debug
```

The validation script checks:
- AiiDA core installation and profile
- Plugin registration and entry points
- Plugin loading functionality
- GSM CLI accessibility
- Custom data type creation

### Example Test Execution

```python
# In Python/Jupyter environment
from aiida import orm, engine, load_profile
from aiida.plugins import WorkflowFactory

# Load profile and plugins
load_profile()
GSMMonotonicWorkChain = WorkflowFactory('gsm.monotonic')

# Set up and submit test workchain
inputs = {
    'gsm_code': orm.Code.collection.get(label='gsm-cli'),
    'gsm_model': orm.Str('GSM1D_ED'),
    'formulation': orm.Str('F'),
    'material_parameters': orm.Dict(dict={
        'E': 30000.0, 'S': 1.0, 'c': 2.0, 'r': 0.9
    }),
    'max_strain': orm.Float(0.01),
    'num_steps': orm.Int(100)
}

workchain = engine.submit(GSMMonotonicWorkChain, **inputs)
print(f"Submitted test workchain: {workchain.pk}")
```

### Troubleshooting Test Issues

1. **Plugin not found errors**:
   ```bash
   # Reinstall with entry points
   pip uninstall bmcs_matmod
   pip install -e .[aiida]
   
   # Check entry points
   python -c "import pkg_resources; print(list(pkg_resources.iter_entry_points('aiida.workflows')))"
   ```

2. **Code setup issues**:
   ```bash
   # Verify GSM CLI is accessible
   which gsm-cli
   gsm-cli --list-models
   
   # Update code path in AiiDA
   verdi code show gsm-cli@localhost
   ```

3. **Database connection issues**:
   ```bash
   # Check AiiDA status
   verdi status
   
   # Reset database if needed
   verdi database migrate
   ```

### Performance Testing

For performance testing with larger workloads:

```python
# Test S-N curve with more stress levels
stress_levels = [300, 275, 250, 225, 200, 175, 150, 125, 100, 75, 50]
max_cycles = 50000

# Test parallel execution
inputs = {
    'stress_levels': orm.List(list=stress_levels),
    'max_cycles': orm.Int(max_cycles),
    # ... other inputs
}

sn_workchain = engine.submit(GSMSNCurveWorkChain, **inputs)
```

## Future Enhancements
