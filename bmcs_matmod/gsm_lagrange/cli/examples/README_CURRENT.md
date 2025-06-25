# GSM CLI Examples

## Status: Examples Need Updating

The example files in this directory are from an earlier version of the GSM Lagrange framework and need updating to work with the current architecture.

## Current Working Examples

For current, working examples of the GSM Lagrange framework, please see:

- **[GSM Lagrange AiiDA Demo Notebook](../../../notebooks/gsm_lagrange_aiida_demo.ipynb)** - Complete workflow demonstration
- **[GSM Model Notebook](../../notebooks/gsm_model.ipynb)** - Basic GSM model usage

## Architecture Changes

The current GSM Lagrange framework uses:
- `GSMModel` class for material model setup
- Direct parameter setting with `.set_params()`
- Optional AiiDA integration via the `aiida_plugin` package
- Modern entry point registration

## CLI Usage

The current CLI interface (`cli_gsm.py`) supports:
```bash
# List available models
python -m bmcs_matmod.gsm_lagrange.cli --list-models

# Run a simulation
python -m bmcs_matmod.gsm_lagrange.cli --model GSM1D_ED --params '{\"E\": 30000, \"S\": 0.0001}'

# Get parameter info for a model
python -m bmcs_matmod.gsm_lagrange.cli --model GSM1D_ED --info
```

## Contributing

If you'd like to update these examples for the current architecture:
1. Use `GSMModel` instead of direct parameter files
2. Follow the patterns in the working notebooks
3. Ensure compatibility with the current import structure
4. Test with both AiiDA and non-AiiDA environments
