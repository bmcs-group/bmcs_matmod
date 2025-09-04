# GSMSymbBox Unit Test Suite

This directory contains comprehensive unit tests for the GSMSymbBox (GSM Thermodynamic State Function Box) implementation, based on the verification notebook `gsm_symb_box_03.ipynb`.

## Overview

The test suite ensures the mathematical correctness of the GSMSymbBox implementation by validating key invariants:

1. **Round-trip consistency** (primary criterion): F→G→F = F exactly
2. **Property access consistency**: Properties vs explicit transformations
3. **Constitutive relation consistency**: Stress-strain and damage relations
4. **Framework validation**: Overall thermodynamic consistency

## Test Structure

### Core Test Classes

- **`TestGSMSymbBoxRoundTrips`**: Tests round-trip transformations (the gold standard)
  - F→G→F, F→U→F, G→H→G, U→G→U, H→F→H
  - Tests both elastic and damage materials
  - Ensures exact mathematical consistency (difference = 0)

- **`TestGSMSymbBoxPropertyConsistency`**: Tests property-based access
  - Validates that `box.F`, `box.G`, `box.U`, `box.H` give same results as explicit transformations
  - Ensures seamless property access

- **`TestGSMSymbBoxConstitutiveRelations`**: Tests constitutive relations
  - Stress consistency: ∂F/∂ε = ∂U/∂ε
  - Inverse relations: ε(σ(ε)) = ε
  - Damage force consistency across state functions

- **`TestGSMSymbBoxFrameworkValidation`**: Framework and edge case tests
  - Tests against known analytical results
  - Built-in validation methods
  - State function accessibility

- **`TestGSMSymbBoxEdgeCases`**: Edge cases and degenerate conditions
  - Linear functions
  - Zero functions
  - Error handling

### Integration Test

- **`test_comprehensive_gsm_symb_box_validation`**: Comprehensive validation similar to notebook
  - Tests all round-trips for damage material
  - Validates constitutive relations
  - Marked with `@pytest.mark.integration`

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific test categories  
python run_tests.py --roundtrip-only
python run_tests.py --constitutive-only

# Skip slow tests
python run_tests.py --fast

# Run with coverage
python run_tests.py --coverage
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run specific test class
pytest tests/test_gsm_symb_box.py::TestGSMSymbBoxRoundTrips

# Run specific test
pytest tests/test_gsm_symb_box.py::TestGSMSymbBoxRoundTrips::test_round_trip_F_to_G_to_F_elastic

# Use markers
pytest -m roundtrip          # Only round-trip tests
pytest -m "not integration"  # Skip integration tests
pytest -m constitutive       # Only constitutive tests

# Verbose output
pytest -v tests/

# Stop on first failure
pytest -x tests/
```

### Test Markers

- `integration`: Comprehensive integration tests (slower)
- `roundtrip`: Round-trip consistency tests (primary criterion)
- `constitutive`: Constitutive relation tests
- `slow`: Slow tests that can be skipped for quick validation

## Test Materials

The tests use two primary material models:

1. **Simple Elastic**: F(T,ε) = ½Eε²
   - Good for basic validation
   - Known analytical results

2. **Elastic-Damage**: F(T,ε,ω) = ½(1-ω)Eε²  
   - More complex model with internal variables
   - Tests full GSMSymbBox capabilities

## Test Fixtures

Common fixtures provide:
- Standard symbol definitions (T, S, ε, σ, ω, Y, E)
- Pre-configured GSMSymbBox instances
- Test material models

## Validation Philosophy

### Primary Criterion: Round-Trip Consistency

The tests prioritize **round-trip consistency** as the gold standard for mathematical correctness. If all round-trip transformations return exactly the original expression (difference = 0), then the implementation is mathematically sound.

This is based on the fundamental property of Legendre transformations: they should be reversible with perfect precision.

### Secondary Validations

- Property access should be consistent with explicit transformations
- Constitutive relations should satisfy thermodynamic principles
- Known analytical results should be reproduced

### Test Reliability

- All comparisons use `sp.simplify(difference) == 0` for exact equality
- Tests use realistic material models that exercise all code paths
- Edge cases and degenerate conditions are included
- Tests are isolated and use fresh GSMSymbBox instances

## Integration with Development Workflow

### Continuous Validation

```bash
# Quick validation during development
python run_tests.py --fast

# Full validation before commits
python run_tests.py

# Coverage analysis
python run_tests.py --coverage
```

### Test-Driven Development

1. When adding new features, first add tests that define expected behavior
2. Run tests to ensure they fail initially: `pytest -x`
3. Implement feature until tests pass
4. Run full test suite to ensure no regressions

### Debugging Failed Tests

```bash
# Run with maximum verbosity
pytest -vvv tests/

# Drop into debugger on failure
pytest --pdb tests/

# Run only failed tests from last run
pytest --lf tests/
```

## Test Maintenance

### Adding New Tests

1. Follow existing naming conventions: `test_*`
2. Use appropriate test class based on functionality
3. Add relevant markers for categorization
4. Include docstrings explaining the test purpose
5. Use fixtures for common setup

### Updating Tests

When GSMSymbBox functionality changes:
1. Update relevant test fixtures
2. Modify expected results if behavior changes
3. Add new tests for new functionality
4. Ensure all round-trip tests still pass (critical invariant)

## Dependencies

The test suite requires:
- `pytest`: Test framework
- `sympy`: Symbolic mathematics (same as GSMSymbBox)
- Standard library modules: `sys`, `pathlib`

Optional dependencies:
- `pytest-cov`: For coverage reports
- `pytest-xdist`: For parallel test execution

## Performance

- **Fast tests** (~1-5 seconds): Basic round-trips, simple materials
- **Integration tests** (~10-30 seconds): Comprehensive validation
- **Total suite** (~30-60 seconds): All tests including edge cases

Use `--fast` flag for quick validation during development.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the correct directory
2. **SymPy version conflicts**: Ensure compatible SymPy version
3. **Test failures**: Check if GSMSymbBox implementation has changed

### Getting Help

1. Run `python run_tests.py --markers` to see available test categories
2. Use `pytest --collect-only` to see all available tests
3. Check test output for specific failure details

### Contributing

When contributing new tests:
1. Follow the existing test structure and naming
2. Ensure tests are focused and test one concept each
3. Add appropriate docstrings and comments
4. Test your tests by making them fail first
5. Update this README if adding new test categories
