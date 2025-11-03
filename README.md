# FRESH-CARE Fusion Toolkit

Python utilities for Work Package 1 (WP1) of the FRESH-CARE project, focused on fusing gridded geophysical fields using spatially weighted local linear regression. The toolkit targets workflows where a coarse Level-3 (L3) product is sharpened with a higher-resolution template while preserving quantitative traceability.

## Overview
- Core numerical routines (`py_fusion.fusion`) implement a NaN-aware 2D/3D local regression with configurable boundary handling and inverse-distance kernels.
- An xarray wrapper (`py_fusion.fusion_xr`) keeps coordinates/attributes intact and streamlines fusion for NetCDF datasets.
- A command-line interface (`python -m py_fusion.cli_fusion`) orchestrates multi-file NetCDF input, dimension remapping, coordinate validation, and NetCDF output with optional compression.

## Requirements
- Python 3.9 or newer (tested with CPython; PyPy not yet validated).
- Python packages: `numpy`, `scipy`, `xarray`, `netCDF4` (or another engine supported by xarray for NetCDF writing).
- Optional: configure logging to surface debug information (`--verbose` flag in the CLI).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/FRESH-CARE.git
   cd FRESH-CARE/code_FRESH_CARE
   ```
2. (Optional) Create and activate a virtual environment.
3. Install the Python dependencies:
   ```bash
   pip install numpy scipy xarray netCDF4
   ```
   Add other project-specific packages as needed for your data-processing pipeline.

## Quickstart
### Run from the command line
```bash
python -m py_code.cli_fusion \
    --l3 /path/to/L3.nc --l3-var L3_variable_name \
    --template /path/to/template.nc --template-var template_variable_name \
    --width 20 --exponent 2 \
    --mask-mode L3 --log-mode none --boundary reflect \
    --output fused_output.nc --verbose
```
Key behaviour:
- Accepts one or more files per input via shell globs (e.g., `"data/L3/*.nc"`).
- Verifies that L3 and template share the same rectilinear grid (no regridding is performed).
- Supports optional dimension remapping flags (`--l3-x-dim`, `--t-y-dim`, …) when datasets use non-standard names.
- Writes a NetCDF file containing the fused field (`L4`) plus diagnostic layers (`a`, `b`, `rho`, `err`).
  
## Development notes
- Enable CLI verbosity (`--verbose`) to expose logging from both the CLI and the core fusion routines.
- Automated tests are not yet included; create case-specific notebooks or scripts to validate outputs for your datasets.
- Contributions are welcome—please open an issue or pull request describing proposed enhancements or bug fixes.
