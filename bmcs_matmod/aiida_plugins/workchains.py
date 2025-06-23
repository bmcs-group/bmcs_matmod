"""
AiiDA workchains for GSM material characterization workflows.

This module provides workchains for:
- Monotonic loading simulations
- Fatigue characterization workflows
- S-N curve construction
"""

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_, while_, calcfunction
from aiida.plugins import CalculationFactory, DataFactory
import numpy as np
import json


# Load plugins
GSMSimulationCalculation = CalculationFactory('gsm.simulation')


@calcfunction
def prepare_monotonic_loading_data(max_strain: orm.Float, num_steps: orm.Int) -> orm.Dict:
    """Prepare monotonic loading history data"""
    max_strain_val = max_strain.value
    num_steps_val = num_steps.value
    
    # Create monotonic strain history
    time_array = np.linspace(0, 1, num_steps_val)
    strain_history = np.linspace(0, max_strain_val, num_steps_val)
    
    loading_data = {
        'time_array': time_array.tolist(),
        'strain_history': strain_history.tolist(),
        'loading_type': 'monotonic_tension',
        'max_strain': max_strain_val,
        'num_steps': num_steps_val
    }
    
    return orm.Dict(dict=loading_data)


@calcfunction
def prepare_fatigue_loading_data(stress_amplitude: orm.Float, stress_mean: orm.Float, 
                                max_cycles: orm.Int) -> orm.Dict:
    """Prepare cyclic loading history data"""
    stress_amp = stress_amplitude.value
    stress_mean_val = stress_mean.value
    max_cycles_val = max_cycles.value
    
    # Create cyclic stress history
    points_per_cycle = 20
    total_points = max_cycles_val * points_per_cycle
    
    time_array = np.linspace(0, max_cycles_val, total_points)
    stress_history = stress_mean_val + stress_amp * np.sin(2 * np.pi * time_array)
    
    loading_data = {
        'time_array': time_array.tolist(),
        'stress_history': stress_history.tolist(),
        'loading_type': 'cyclic_stress',
        'cycles': max_cycles_val,
        'stress_amplitude': stress_amp,
        'stress_mean': stress_mean_val
    }
    
    return orm.Dict(dict=loading_data)


class GSMMonotonicWorkChain(WorkChain):
    """Workchain for monotonic loading characterization"""

    @classmethod
    def define(cls, spec):
        """Define the workchain specification"""
        super().define(spec)
        
        # Inputs
        spec.input('gsm_code', valid_type=orm.Code, help='GSM CLI code')
        spec.input('gsm_model', valid_type=orm.Str, help='GSM model identifier')
        spec.input('formulation', valid_type=orm.Str, default=orm.Str('F'), help='Model formulation')
        spec.input('material_parameters', valid_type=orm.Dict, help='Material parameters')
        spec.input('max_strain', valid_type=orm.Float, help='Maximum strain level')
        spec.input('num_steps', valid_type=orm.Int, default=orm.Int(100), help='Number of loading steps')
        
        # Outputs
        spec.output('monotonic_results', valid_type=orm.Dict, help='Monotonic simulation results')
        spec.output('stress_strain_curve', valid_type=orm.ArrayData, required=False, help='Stress-strain response')
        
        # Exit codes
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED', message='Sub-process failed')
        spec.exit_code(401, 'ERROR_INVALID_PARAMETERS', message='Invalid material parameters')
        
        # Workflow outline
        spec.outline(
            cls.prepare_loading,
            cls.run_simulation,
            cls.extract_results
        )

    def prepare_loading(self):
        """Prepare monotonic loading history"""
        self.ctx.loading_data = prepare_monotonic_loading_data(
            self.inputs.max_strain, 
            self.inputs.num_steps
        )

    def run_simulation(self):
        """Run the monotonic simulation"""
        
        inputs = {
            'code': self.inputs.gsm_code,
            'gsm_model': self.inputs.gsm_model,
            'formulation': self.inputs.formulation,
            'material_parameters': self.inputs.material_parameters,
            'loading_data': self.ctx.loading_data,
            'metadata': {
                'label': 'GSM Monotonic Loading',
                'description': f'Monotonic loading to {self.inputs.max_strain.value} strain',
                'options': {
                    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 1},
                    'max_wallclock_seconds': 1800
                }
            }
        }
        
        running = self.submit(GSMSimulationCalculation, **inputs)
        return ToContext(monotonic_calc=running)

    def extract_results(self):
        """Extract and organize results"""
        calc = self.ctx.monotonic_calc
        
        if not calc.is_finished_ok:
            self.report(f"Simulation calculation failed with exit status: {calc.exit_status}")
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        
        # Pass through main results
        self.out('monotonic_results', calc.outputs.results)
        
        # Extract stress-strain curve if available
        if 'response_data' in calc.outputs:
            self.out('stress_strain_curve', calc.outputs.response_data)


class GSMFatigueWorkChain(WorkChain):
    """Workchain for fatigue characterization at fixed stress level"""

    @classmethod
    def define(cls, spec):
        """Define the workchain specification"""
        super().define(spec)
        
        # Inputs
        spec.input('gsm_code', valid_type=orm.Code, help='GSM CLI code')
        spec.input('gsm_model', valid_type=orm.Str, help='GSM model identifier')
        spec.input('formulation', valid_type=orm.Str, help='Model formulation')
        spec.input('material_parameters', valid_type=orm.Dict, help='Material parameters')
        spec.input('stress_amplitude', valid_type=orm.Float, help='Fatigue stress amplitude')
        spec.input('stress_mean', valid_type=orm.Float, default=orm.Float(0.0), help='Mean stress')
        spec.input('max_cycles', valid_type=orm.Int, default=orm.Int(10000), help='Maximum cycles')
        spec.input('failure_strain', valid_type=orm.Float, default=orm.Float(0.1), 
                  help='Failure strain criterion')
        
        # Outputs
        spec.output('fatigue_results', valid_type=orm.Dict, help='Fatigue simulation results')
        spec.output('cycles_to_failure', valid_type=orm.Int, help='Number of cycles to failure')
        spec.output('fatigue_response', valid_type=orm.ArrayData, required=False, 
                   help='Fatigue response data')
        
        # Workflow outline
        spec.outline(
            cls.prepare_fatigue_loading,
            cls.run_fatigue_simulation,
            cls.analyze_fatigue_results
        )

    def prepare_fatigue_loading(self):
        """Prepare cyclic loading history"""
        stress_amplitude = self.inputs.stress_amplitude.value
        stress_mean = self.inputs.stress_mean.value
        max_cycles = self.inputs.max_cycles.value
        
        # Create cyclic stress history
        points_per_cycle = 20
        total_points = max_cycles * points_per_cycle
        
        time_array = np.linspace(0, max_cycles, total_points)
        stress_history = stress_mean + stress_amplitude * np.sin(2 * np.pi * time_array)
        
        loading_data = {
            'time_array': time_array.tolist(),
            'stress_history': stress_history.tolist(),
            'loading_type': 'cyclic_stress',
            'cycles': max_cycles,
            'stress_amplitude': stress_amplitude,
            'stress_mean': stress_mean
        }
        
        self.ctx.loading_data = orm.Dict(dict=loading_data)

    def run_fatigue_simulation(self):
        """Run the fatigue simulation"""
        
        inputs = {
            'code': self.inputs.gsm_code,
            'gsm_model': self.inputs.gsm_model,
            'formulation': self.inputs.formulation,
            'material_parameters': self.inputs.material_parameters,
            'loading_data': self.ctx.loading_data,
            'metadata': {
                'label': 'GSM Fatigue Loading',
                'description': f'Fatigue at {self.inputs.stress_amplitude.value} MPa amplitude'
            }
        }
        
        running = self.submit(GSMSimulationCalculation, **inputs)
        return ToContext(fatigue_calc=running)

    def analyze_fatigue_results(self):
        """Analyze fatigue results and determine failure"""
        calc = self.ctx.fatigue_calc
        
        if not calc.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        
        # Extract results
        results = calc.outputs.results.get_dict()
        
        # Analyze for failure (simplified - could be more sophisticated)
        failure_strain = self.inputs.failure_strain.value
        max_strain = results.get('max_strain', 0.0)
        
        if max_strain >= failure_strain:
            # Estimate cycles to failure (simplified)
            cycles_to_failure = int(results.get('final_time', self.inputs.max_cycles.value))
        else:
            cycles_to_failure = self.inputs.max_cycles.value  # Runout
        
        # Create comprehensive results
        fatigue_results = {
            **results,
            'stress_amplitude': self.inputs.stress_amplitude.value,
            'stress_mean': self.inputs.stress_mean.value,
            'failed': max_strain >= failure_strain,
            'max_strain_reached': max_strain,
            'failure_criterion': failure_strain
        }
        
        self.out('fatigue_results', orm.Dict(dict=fatigue_results))
        self.out('cycles_to_failure', orm.Int(cycles_to_failure))
        
        if 'response_data' in calc.outputs:
            self.out('fatigue_response', calc.outputs.response_data)


class GSMSNCurveWorkChain(WorkChain):
    """Workchain for constructing S-N fatigue curves"""

    @classmethod
    def define(cls, spec):
        """Define the workchain specification"""
        super().define(spec)
        
        # Inputs
        spec.input('gsm_code', valid_type=orm.Code, help='GSM CLI code')
        spec.input('gsm_model', valid_type=orm.Str, help='GSM model identifier')
        spec.input('formulation', valid_type=orm.Str, help='Model formulation')
        spec.input('material_parameters', valid_type=orm.Dict, help='Material parameters')
        spec.input('stress_levels', valid_type=orm.List, help='List of stress amplitudes to test')
        spec.input('max_cycles', valid_type=orm.Int, default=orm.Int(10000), help='Maximum cycles')
        spec.input('failure_strain', valid_type=orm.Float, default=orm.Float(0.1), 
                  help='Failure strain criterion')
        
        # Outputs
        spec.output('sn_curve_data', valid_type=orm.ArrayData, help='S-N curve data points')
        spec.output('fatigue_database', valid_type=orm.Dict, help='Complete fatigue results database')
        
        # Workflow outline
        spec.outline(
            cls.initialize_sn_curve,
            cls.run_fatigue_tests,
            cls.construct_sn_curve
        )

    def initialize_sn_curve(self):
        """Initialize S-N curve construction"""
        stress_levels = self.inputs.stress_levels.get_list()
        self.ctx.stress_levels = stress_levels
        self.ctx.fatigue_results = {}
        self.ctx.current_stress_index = 0

    def run_fatigue_tests(self):
        """Run fatigue tests for all stress levels"""
        
        # Submit all fatigue simulations in parallel
        stress_levels = self.ctx.stress_levels
        calculations = {}
        
        for i, stress_amplitude in enumerate(stress_levels):
            inputs = {
                'gsm_code': self.inputs.gsm_code,
                'gsm_model': self.inputs.gsm_model,
                'formulation': self.inputs.formulation,
                'material_parameters': self.inputs.material_parameters,
                'stress_amplitude': orm.Float(stress_amplitude),
                'max_cycles': self.inputs.max_cycles,
                'failure_strain': self.inputs.failure_strain,
                'metadata': {
                    'label': f'GSM Fatigue S-N Point {i+1}',
                    'description': f'Fatigue test at {stress_amplitude} MPa'
                }
            }
            
            future = self.submit(GSMFatigueWorkChain, **inputs)
            calculations[f'fatigue_{i}'] = future
        
        return ToContext(**calculations)

    def construct_sn_curve(self):
        """Construct S-N curve from fatigue test results"""
        
        stress_levels = self.ctx.stress_levels
        sn_data = []
        fatigue_database = {}
        
        for i, stress_amplitude in enumerate(stress_levels):
            calc_key = f'fatigue_{i}'
            if calc_key in self.ctx:
                calc = getattr(self.ctx, calc_key)
                
                if calc.is_finished_ok:
                    cycles = calc.outputs.cycles_to_failure.value
                    results = calc.outputs.fatigue_results.get_dict()
                    
                    sn_data.append([stress_amplitude, cycles])
                    fatigue_database[f'stress_{stress_amplitude}'] = {
                        'stress_amplitude': stress_amplitude,
                        'cycles_to_failure': cycles,
                        'results': results
                    }
        
        # Create S-N curve array data
        sn_array = np.array(sn_data)
        array_data = orm.ArrayData()
        array_data.set_array('stress_amplitude', sn_array[:, 0])
        array_data.set_array('cycles_to_failure', sn_array[:, 1])
        array_data.set_attribute('description', 'S-N curve data')
        array_data.set_attribute('stress_units', 'MPa')
        array_data.set_attribute('cycles_units', 'cycles')
        
        self.out('sn_curve_data', array_data)
        self.out('fatigue_database', orm.Dict(dict=fatigue_database))
