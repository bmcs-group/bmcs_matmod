# Running Tests for Time Function Module

This directory contains unit tests for the `time_fn.py` module using the pytest framework.

## Running Tests in VS Code

1. **Install Required Packages**:
   ```
   pip install pytest numpy sympy
   ```

2. **Run Tests Using VS Code Test Explorer**:
   - Open VS Code
   - Navigate to the Testing panel (flask icon in the sidebar)
   - Click on the refresh icon to discover tests
   - Run all tests by clicking the play button, or run individual tests by clicking the play button next to each test

3. **Run Tests from Terminal**:
   ```
   # Navigate to the project root
   cd /home/rch/Coding/bmcs_matmod
   
   # Run all tests
   python -m pytest
   
   # Run specific test file
   python -m pytest bmcs_matmod/time_fn/tests/test_time_fn.py
   
   # Run a specific test class
   python -m pytest bmcs_matmod/time_fn/tests/test_time_fn.py::TestTimeFnStepLoading
   
   # Run a specific test
   python -m pytest bmcs_matmod/time_fn/tests/test_time_fn.py::TestTimeFnStepLoading::test_call
   ```

4. **Run with Verbose Output**:
   ```
   python -m pytest -v
   ```

5. **Generate Coverage Report**:
   ```
   pip install pytest-cov
   python -m pytest --cov=bmcs_matmod.time_fn
   ```

## Test Structure

The tests are organized by class, following the structure of the `time_fn.py` module:

- `TestSymbolRegistry`: Tests for the symbol registry functionality
- `TestTimeFnBase`: Tests for the base class and its core functionality
- `TestTimeFnStepLoading`: Tests for the step loading function
- `TestTimeFnMonotonicAscending`: Tests for the monotonic ascending function
- `TestTimeFnCycleSinus`: Tests for the sinusoidal cycle function
- `TestTimeFnPeriodic`: Tests for the periodic function
- `TestTimeFnStepping`: Tests for the stepping function (numerical approach)
- `TestTimeFnOverlay`: Tests for the overlay function (hybrid approach)

Each test class contains tests for both the symbolic expression (`test_symb_expr`) and the numerical evaluation (`test_call`), where applicable.
