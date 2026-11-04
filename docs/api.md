# API Reference

Reference documentation generated from the `famex` package docstrings.

## Quick Reference

Most usage only needs {py:class}`~famex.core.explorer.Explorer` (and, for advanced cases,
{py:class}`~famex.io.geometry.Geometry` and {py:class}`~famex.io.path_manager.PathManager`).
Everything listed here can be imported from `famex` and is documented in full further down this page.

```{list-table}
:class: fit-table nowrap-col2
:header-rows: 1

* - Name
  - Category
  - Description
* - {py:class}`~famex.core.explorer.Explorer`
  - Core
  - Main entry point: run minima/TS/path optimizations
* - {py:class}`~famex.io.geometry.Geometry`
  - I/O
  - Structure container returned by `read_geometry`
* - {py:func}`~famex.io.geometry.read_geometry`, {py:func}`~famex.io.geometry.write_geometry`
  - I/O
  - Read and write structure files
* - {py:func}`~famex.embed.api.smiles_to_atoms`, {py:func}`~famex.io.pubchem.fetch_pubchem`
  - I/O
  - Build 3D structures from SMILES or PubChem
* - {py:class}`~famex.io.path_manager.PathManager`
  - I/O
  - Reaction paths: interpolation, TS guesses, duplicate removal
* - {py:class}`~famex.analysis.frequency.FrequencyAnalysis`
  - Analysis
  - Vibrational frequencies and thermodynamic properties
* - {py:class}`~famex.analysis.hessian.HessianCalculator`
  - Analysis
  - Numerical Hessian construction
* - {py:class}`~famex.analysis.thermodynamics.ThermodynamicProperties`
  - Analysis
  - Thermochemistry from vibrational frequencies
* - {py:func}`~famex.backends.registry.create_calculator`, {py:data}`~famex.backends.registry.calculator_registry`
  - Backends
  - Build/look up an ASE calculator for a given backend
* - {py:func}`~famex.backends.availability.get_available_backends`, {py:func}`~famex.backends.availability.is_backend_available`
  - Backends
  - Check which backends are installed
* - {py:class}`~famex.utils.validation.FAMEXError`, {py:class}`~famex.utils.validation.BackendError`, {py:class}`~famex.utils.validation.DependencyError`
  - Errors
  - Exception types raised by FAMEX
```

## Top-Level Package

```{eval-rst}
.. automodule:: famex
   :no-members:
```

## Core

### Explorer

```{eval-rst}
.. automodule:: famex.core.explorer
   :members:
```

### Strategy Base Classes

```{eval-rst}
.. automodule:: famex.core.base_strategy
   :members:
```

### Strategy Exceptions

```{eval-rst}
.. automodule:: famex.core.exceptions
   :members:
   :show-inheritance:
```

## Structures and I/O

### Geometry

```{eval-rst}
.. automodule:: famex.io.geometry
   :members:
```

### XYZ Files

```{eval-rst}
.. automodule:: famex.io.xyz_io
   :members:
```

### SMILES Embedding

```{eval-rst}
.. automodule:: famex.embed.api
   :members: smiles_to_atoms, embed_smiles
```

### PubChem

```{eval-rst}
.. automodule:: famex.io.pubchem
   :members: fetch_pubchem, parse_pubchem_query
```

### Reaction Paths

```{eval-rst}
.. automodule:: famex.io.path_manager
   :members:
```

### Run Outputs

```{eval-rst}
.. automodule:: famex.io.run_output
   :members:
```

### Vibrational Mode Animations

```{eval-rst}
.. automodule:: famex.io.modes
   :members:
```

## Backends

### Calculator Registry

```{eval-rst}
.. automodule:: famex.backends.registry
   :members: CalculatorRegistry, create_calculator, calculator_registry
```

### Backend Availability

```{eval-rst}
.. automodule:: famex.backends.availability
   :members: is_backend_available, get_available_backends, get_availability_reason, get_backend_error_message
```

## Analysis

### Frequency Analysis

```{eval-rst}
.. automodule:: famex.analysis.frequency
   :members:
   :exclude-members: ThermodynamicProperties, HessianCalculator
```

### Hessian

```{eval-rst}
.. automodule:: famex.analysis.hessian
   :members: HessianCalculator
```

### Thermochemistry

```{eval-rst}
.. automodule:: famex.analysis.thermodynamics
   :members:
```

## Constraints and Interpolation

### Constraints

```{eval-rst}
.. automodule:: famex.constraints.constraints
   :members:
   :exclude-members: HarmonicAngleASEConstraint
```

### Interpolation

```{eval-rst}
.. automodule:: famex.interpolation.strategies
   :members:
   :exclude-members: idpp_energy_and_forces
```

## Errors

```{eval-rst}
.. automodule:: famex.utils.validation
   :members: FAMEXError, BackendError, DependencyError
   :show-inheritance:
```
