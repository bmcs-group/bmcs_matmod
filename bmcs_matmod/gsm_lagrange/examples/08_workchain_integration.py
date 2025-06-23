#!/usr/bin/env python3
"""
Workchain Integration Example for GSM CLI

This script demonstrates how to integrate GSM CLI with workflow management systems
like AiiDA for automated material characterization workflows.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class WorkflowStep:
    """Represents a single step in a material characterization workflow"""
    step_id: str
    description: str
    model: str
    formulation: str
    parameters: Dict[str, Any]
    loading: Dict[str, Any]
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class GSMWorkchainManager:
    """Manages GSM workchains for material characterization"""
    
    def __init__(self, cli_path: str = "../cli_gsm.py"):
        self.cli_path = cli_path
        self.workflow_steps = {}
        self.results = {}
        self.execution_log = []
    
    def add_workflow_step(self, step: WorkflowStep):
        """Add a step to the workflow"""
        self.workflow_steps[step.step_id] = step
        print(f"Added workflow step: {step.step_id} - {step.description}")
    
    def validate_workflow(self) -> List[str]:
        """Validate the workflow for dependency cycles and missing dependencies"""
        issues = []
        
        # Check for missing dependencies
        for step_id, step in self.workflow_steps.items():
            for dep in step.dependencies:
                if dep not in self.workflow_steps:
                    issues.append(f"Step {step_id} depends on missing step {dep}")
        
        # Check for circular dependencies (simple check)
        for step_id, step in self.workflow_steps.items():
            visited = set()
            if self._has_circular_dependency(step_id, visited):
                issues.append(f"Circular dependency detected involving step {step_id}")
        
        return issues
    
    def _has_circular_dependency(self, step_id: str, visited: set) -> bool:
        """Check for circular dependencies starting from a step"""
        if step_id in visited:
            return True
        
        visited.add(step_id)
        step = self.workflow_steps.get(step_id)
        if step:
            for dep in step.dependencies:
                if self._has_circular_dependency(dep, visited.copy()):
                    return True
        
        return False
    
    def get_execution_order(self) -> List[str]:
        """Get the execution order for workflow steps"""
        ordered_steps = []
        remaining_steps = set(self.workflow_steps.keys())
        
        while remaining_steps:
            # Find steps with no unresolved dependencies
            ready_steps = []
            for step_id in remaining_steps:
                step = self.workflow_steps[step_id]
                if all(dep in ordered_steps for dep in step.dependencies):
                    ready_steps.append(step_id)
            
            if not ready_steps:
                # Circular dependency or other issue
                break
            
            # Add ready steps to execution order
            for step_id in ready_steps:
                ordered_steps.append(step_id)
                remaining_steps.remove(step_id)
        
        return ordered_steps
    
    def execute_step(self, step_id: str) -> Dict[str, Any]:
        """Execute a single workflow step"""
        if step_id not in self.workflow_steps:
            return {"status": "error", "error": f"Step {step_id} not found"}
        
        step = self.workflow_steps[step_id]
        print(f"Executing step: {step.step_id} - {step.description}")
        
        # Check dependencies
        for dep in step.dependencies:
            if dep not in self.results or self.results[dep]["status"] != "success":
                return {"status": "error", "error": f"Dependency {dep} not satisfied"}
        
        # Build CLI command
        params_json = json.dumps(step.parameters)
        loading_json = json.dumps(step.loading)
        
        cmd = [
            "python", self.cli_path,
            "--model", step.model,
            "--formulation", step.formulation,
            "--params-inline", params_json,
            "--loading-inline", loading_json,
            "--json-output"
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                simulation_result = json.loads(result.stdout)
                simulation_result["step_id"] = step_id
                simulation_result["execution_time"] = execution_time
                
                # Log execution
                self.execution_log.append({
                    "step_id": step_id,
                    "timestamp": time.time(),
                    "execution_time": execution_time,
                    "status": "success"
                })
                
                print(f"  ✅ Completed in {execution_time:.3f}s")
                return simulation_result
            else:
                error_result = {
                    "status": "error",
                    "error": result.stderr,
                    "step_id": step_id
                }
                
                self.execution_log.append({
                    "step_id": step_id,
                    "timestamp": time.time(),
                    "status": "error",
                    "error": result.stderr
                })
                
                print(f"  ❌ Failed: {result.stderr}")
                return error_result
                
        except subprocess.TimeoutExpired:
            error_result = {
                "status": "error",
                "error": "Execution timeout",
                "step_id": step_id
            }
            self.execution_log.append({
                "step_id": step_id,
                "timestamp": time.time(),
                "status": "error",
                "error": "timeout"
            })
            print(f"  ❌ Timeout")
            return error_result
        
        except json.JSONDecodeError as e:
            error_result = {
                "status": "error",
                "error": f"JSON decode error: {e}",
                "step_id": step_id
            }
            print(f"  ❌ JSON decode error: {e}")
            return error_result
    
    def execute_workflow(self) -> Dict[str, Any]:
        """Execute the complete workflow"""
        print("Starting GSM Workchain Execution")
        print("=" * 50)
        
        # Validate workflow
        issues = self.validate_workflow()
        if issues:
            print("Workflow validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return {"status": "error", "error": "Workflow validation failed", "issues": issues}
        
        # Get execution order
        execution_order = self.get_execution_order()
        if len(execution_order) != len(self.workflow_steps):
            return {"status": "error", "error": "Could not determine execution order"}
        
        print(f"Execution order: {' → '.join(execution_order)}")
        print()
        
        # Execute steps in order
        workflow_start_time = time.time()
        
        for step_id in execution_order:
            result = self.execute_step(step_id)
            self.results[step_id] = result
            
            if result["status"] != "success":
                print(f"Workflow failed at step {step_id}")
                break
        
        workflow_time = time.time() - workflow_start_time
        
        # Workflow summary
        successful_steps = sum(1 for r in self.results.values() if r["status"] == "success")
        total_steps = len(self.workflow_steps)
        
        summary = {
            "status": "success" if successful_steps == total_steps else "partial_failure",
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "workflow_execution_time": workflow_time,
            "results": self.results,
            "execution_log": self.execution_log
        }
        
        print(f"\nWorkflow Summary:")
        print(f"  Total steps: {total_steps}")
        print(f"  Successful: {successful_steps}")
        print(f"  Success rate: {successful_steps/total_steps:.1%}")
        print(f"  Total time: {workflow_time:.3f}s")
        
        return summary
    
    def save_workflow_results(self, filename: str):
        """Save workflow results to file"""
        output_data = {
            "workflow_steps": {k: {
                "step_id": v.step_id,
                "description": v.description,
                "model": v.model,
                "formulation": v.formulation,
                "parameters": v.parameters,
                "loading": v.loading,
                "dependencies": v.dependencies
            } for k, v in self.workflow_steps.items()},
            "results": self.results,
            "execution_log": self.execution_log
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Workflow results saved to {filename}")


def example_material_characterization_workflow():
    """Example: Complete material characterization workflow"""
    print("Material Characterization Workflow Example")
    print("=" * 60)
    
    manager = GSMWorkchainManager()
    
    # Step 1: Basic elastic response
    elastic_step = WorkflowStep(
        step_id="elastic_response",
        description="Determine elastic modulus and Poisson's ratio",
        model="GSM1D_ED",
        formulation="F",
        parameters={
            "E": 30000.0,
            "S": 100.0,  # High value to prevent damage
            "c": 10.0,
            "r": 0.1,
            "eps_0": 0.0001
        },
        loading={
            "time_array": [0.0, 1.0],
            "strain_history": [0.0, 0.001],  # Small strain for elastic response
            "loading_type": "strain_controlled"
        }
    )
    
    # Step 2: Damage initiation
    damage_initiation_step = WorkflowStep(
        step_id="damage_initiation",
        description="Determine damage initiation threshold",
        model="GSM1D_ED",
        formulation="F",
        parameters={
            "E": 30000.0,
            "S": 1.0,
            "c": 2.0,
            "r": 0.5,
            "eps_0": 0.001
        },
        loading={
            "time_array": [0.0, 0.5, 1.0],
            "strain_history": [0.0, 0.002, 0.004],
            "loading_type": "strain_controlled"
        },
        dependencies=["elastic_response"]
    )
    
    # Step 3: Post-peak behavior
    post_peak_step = WorkflowStep(
        step_id="post_peak",
        description="Characterize post-peak softening behavior",
        model="GSM1D_ED",
        formulation="F",
        parameters={
            "E": 30000.0,
            "S": 1.0,
            "c": 2.0,
            "r": 0.5,
            "eps_0": 0.001
        },
        loading={
            "time_array": [0.0, 0.3, 0.6, 1.0],
            "strain_history": [0.0, 0.005, 0.01, 0.015],
            "loading_type": "strain_controlled"
        },
        dependencies=["damage_initiation"]
    )
    
    # Step 4: Cyclic behavior
    cyclic_step = WorkflowStep(
        step_id="cyclic_behavior",
        description="Analyze cyclic loading response",
        model="GSM1D_ED",
        formulation="F",
        parameters={
            "E": 30000.0,
            "S": 1.0,
            "c": 2.0,
            "r": 0.5,
            "eps_0": 0.001
        },
        loading={
            "time_array": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
            "strain_history": [0.0, 0.005, 0.0, -0.003, 0.0, 0.005, 0.01],
            "loading_type": "strain_controlled"
        },
        dependencies=["post_peak"]
    )
    
    # Add steps to workflow
    for step in [elastic_step, damage_initiation_step, post_peak_step, cyclic_step]:
        manager.add_workflow_step(step)
    
    # Execute workflow
    workflow_result = manager.execute_workflow()
    
    # Save results
    manager.save_workflow_results("material_characterization_workflow.json")
    
    return workflow_result


def example_parameter_sensitivity_workflow():
    """Example: Parameter sensitivity analysis workflow"""
    print("\nParameter Sensitivity Analysis Workflow")
    print("=" * 60)
    
    manager = GSMWorkchainManager()
    
    # Base parameters
    base_params = {
        "E": 30000.0,
        "S": 1.0,
        "c": 2.0,
        "r": 0.5,
        "eps_0": 0.001
    }
    
    # Standard loading
    standard_loading = {
        "time_array": [0.0, 0.5, 1.0],
        "strain_history": [0.0, 0.005, 0.01],
        "loading_type": "strain_controlled"
    }
    
    # Vary elastic modulus
    e_values = [25000.0, 30000.0, 35000.0]
    for i, e_val in enumerate(e_values):
        params = base_params.copy()
        params["E"] = e_val
        
        step = WorkflowStep(
            step_id=f"E_sensitivity_{i}",
            description=f"Test with E = {e_val} MPa",
            model="GSM1D_ED",
            formulation="F",
            parameters=params,
            loading=standard_loading
        )
        manager.add_workflow_step(step)
    
    # Vary damage parameter S
    s_values = [0.5, 1.0, 1.5]
    for i, s_val in enumerate(s_values):
        params = base_params.copy()
        params["S"] = s_val
        
        step = WorkflowStep(
            step_id=f"S_sensitivity_{i}",
            description=f"Test with S = {s_val} MPa",
            model="GSM1D_ED",
            formulation="F",
            parameters=params,
            loading=standard_loading
        )
        manager.add_workflow_step(step)
    
    # Execute workflow
    workflow_result = manager.execute_workflow()
    
    # Save results
    manager.save_workflow_results("parameter_sensitivity_workflow.json")
    
    return workflow_result


def main():
    """Run workchain integration examples"""
    print("GSM CLI Workchain Integration Examples")
    print("=" * 70)
    
    try:
        # Run material characterization workflow
        char_result = example_material_characterization_workflow()
        
        # Run parameter sensitivity workflow
        sens_result = example_parameter_sensitivity_workflow()
        
        print("\n" + "=" * 70)
        print("Workchain Examples Completed!")
        print("\nGenerated files:")
        print("  - material_characterization_workflow.json")
        print("  - parameter_sensitivity_workflow.json")
        print("\nThese examples demonstrate how to:")
        print("  - Create complex multi-step workflows")
        print("  - Handle step dependencies")
        print("  - Validate workflow integrity")
        print("  - Execute workflows with error handling")
        print("  - Save and analyze workflow results")
        
    except KeyboardInterrupt:
        print("\nWorkflow execution interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
