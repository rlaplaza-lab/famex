# ASE Units Integration in QME

This document summarizes the comprehensive integration of ASE units throughout the QME codebase, replacing manual unit conversions with standardized ASE unit handling.

## Overview

QME now uses ASE's unit system consistently throughout the codebase, ensuring proper unit handling and conversions. This integration follows the ASE units documentation: https://ase-lib.org/ase/units.html

## Changes Made

### 1. Test Files (`tests/unit/test_geometric_interface.py`)

**Before:**
```python
mock_progress.xyzs = [
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * 1.8897259886
]
```

**After:**
```python
from ase.units import Bohr
mock_progress.xyzs = [
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]) * Bohr
]
```

- Replaced hardcoded Bohr conversion factor `1.8897259886` with `ASE.units.Bohr`
- Applied to all test mocks for coordinate conversion

### 2. Core Validation (`qme/core/validation.py`)

**Before:**
```python
MIN_ATOM_DISTANCE = 0.1  # Minimum allowed distance between atoms (Å)
MAX_REASONABLE_FORCE = 100.0  # Maximum reasonable force magnitude (eV/Å)
```

**After:**
```python
from ase.units import Ang, eV

MIN_ATOM_DISTANCE = 0.1 * Ang  # Minimum allowed distance between atoms
MAX_REASONABLE_FORCE = 100.0 * eV / Ang  # Maximum reasonable force magnitude
```

### 3. Mock Potential (`qme/potentials/mock_potential.py`)

**Before:**
```python
MIN_BOND_CUTOFF = 2.0  # Minimum bond cutoff distance (Å)
DEFAULT_FORCE_CONSTANT = 1.0  # Default harmonic force constant
```

**After:**
```python
from ase.units import Ang, eV

MIN_BOND_CUTOFF = 2.0 * Ang  # Minimum bond cutoff distance
DEFAULT_FORCE_CONSTANT = 1.0 * eV / Ang**2  # Default harmonic force constant
```

### 4. Frequency Analysis (`qme/analysis/frequency.py`)

**Before:**
```python
# Manual frequency conversions
freq_eV = real_frequencies * units._hplanck * units._c * 100 / units._e
```

**After:**
```python
# Using ASE units for frequency conversion
freq_eV = real_frequencies * units.invcm  # Convert cm^-1 to eV
```

- Replaced manual frequency-to-energy conversions with `units.invcm`
- Applied to all thermodynamic property calculations
- Maintained consistency with ASE's frequency handling

### 5. Constraints (`qme/core/constraints.py`)

**Before:**
```python
force_constant = 10.0  # default
```

**After:**
```python
from ase.units import eV, Ang

force_constant = 10.0 * eV / Ang**2  # default
```

### 6. Documentation Updates

Updated documentation files to reference ASE units:

- `docs/tutorials/basic_optimization.md`: Added ASE units section
- `docs/getting_started.md`: Updated unit references
- Added comprehensive example: `examples/ase_units_example.py`

## Available ASE Units in QME

### Basic Units
- `eV`: Electron volts (energy)
- `Ang`: Angstrom (length)
- `Bohr`: Bohr radius (length)
- `Hartree`: Hartree (energy)

### Derived Units
- `invcm`: Inverse centimeters (frequency to energy conversion)
- `kB`: Boltzmann constant
- `fs`: Femtoseconds (time)

### Physical Constants
- `_hplanck`: Planck constant
- `_c`: Speed of light
- `_e`: Elementary charge
- `_amu`: Atomic mass unit
- `_hbar`: Reduced Planck constant

## Usage Examples

### Basic Unit Conversions
```python
from ase.units import eV, Ang, Bohr, Hartree

# Energy conversions
energy_ev = 1.0 * eV
energy_hartree = energy_ev / Hartree

# Length conversions
distance_ang = 1.5 * Ang
distance_bohr = distance_ang / Bohr

# Force conversions
force_ev_ang = 0.05 * eV / Ang
```

### Frequency Calculations
```python
from ase.units import invcm

# Convert cm^-1 to eV
freq_cm1 = 1000.0  # cm^-1
freq_ev = freq_cm1 * invcm  # eV
```

### QME Integration
```python
from qme.core.validation import MIN_ATOM_DISTANCE, MAX_REASONABLE_FORCE
from qme.potentials.mock_potential import MIN_BOND_CUTOFF

# All QME constants now use ASE units
print(f"Min distance: {MIN_ATOM_DISTANCE}")  # 0.1 Å
print(f"Max force: {MAX_REASONABLE_FORCE}")  # 100.0 eV/Å
```

## Benefits

1. **Consistency**: All unit conversions use the same ASE unit system
2. **Accuracy**: Eliminates manual conversion factor errors
3. **Maintainability**: Centralized unit handling through ASE
4. **Documentation**: Clear unit references throughout codebase
5. **Future-proof**: Easy to update units if ASE changes

## Testing

All changes have been tested and verified:

- Unit tests pass with new ASE units integration
- Frequency analysis works correctly with ASE units
- Coordinate conversions use proper ASE units
- Example script demonstrates all unit conversions

## References

- [ASE Units Documentation](https://ase-lib.org/ase/units.html)
- [CODATA Constants](https://physics.nist.gov/cuu/Constants/)
- [QME Examples](examples/ase_units_example.py)

## Migration Guide

If you have existing code using manual unit conversions:

1. **Replace hardcoded factors**: Use ASE units instead of manual conversion factors
2. **Import ASE units**: Add `from ase.units import eV, Ang, Bohr` as needed
3. **Update constants**: Use ASE unit multiplication for derived units
4. **Test thoroughly**: Verify unit conversions produce expected results

Example migration:
```python
# Old way
distance_bohr = distance_ang * 1.8897259886

# New way
from ase.units import Bohr
distance_bohr = distance_ang / Bohr
```

This integration ensures QME maintains consistency with the broader ASE ecosystem while providing accurate and maintainable unit handling.
