# Contributing to RIOMAR / healpix_regrid

Thank you for your interest in contributing! This document explains how to get
started.

## Getting started

1. **Fork** the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/softwares_RIOMAR.git
   cd softwares_RIOMAR
   ```

2. **Create the conda environment** and install the package in editable mode:

   ```bash
   conda env create -f notebook/environment.yml
   conda activate riomar
   pip install -e ".[test]"
   ```

3. **Create a branch** for your changes:

   ```bash
   git checkout -b my-feature
   ```

## Project layout

Reusable logic lives in the `healpix_regrid/` Python package. The `bin/`
scripts and `notebook/` Jupyter notebooks consume it.

When adding new functionality:
- If it is reusable across datasets, add it to `healpix_regrid/` with tests.
- If it is specific to a particular run or experiment, a notebook or script in
  `bin/` is fine.

## Running tests

```bash
python -m pytest tests/ -v
```

All new code in `healpix_regrid/` should have corresponding tests in `tests/`.
Tests should run without access to HPC filesystems or large datasets — use
small synthetic data (NumPy arrays, in-memory xarray Datasets).

## Code style

- Follow existing conventions in the codebase.
- Use type hints for function signatures.
- Keep docstrings in NumPy style (Parameters / Returns sections).

## Submitting changes

1. Make sure all tests pass locally.
2. Push your branch and open a Pull Request against `main`.
3. Describe **what** your change does and **why** in the PR description.

## Reporting issues

Open an issue at
[github.com/FAIR2Adapt/softwares_RIOMAR/issues](https://github.com/FAIR2Adapt/softwares_RIOMAR/issues)
with:
- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Your environment (OS, Python version, conda env)
