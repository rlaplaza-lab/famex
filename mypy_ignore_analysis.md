# Mypy Ignore Patterns Analysis

This document categorizes and analyzes all 86 mypy ignore comments in the codebase, focusing on defensive patterns and try/except blocks.

## Summary Statistics

- **Total ignores**: 86
- **Patterns with `[unreachable]`**: ~40
- **Patterns with `[attr-defined]`**: ~5
- **Patterns with `[assignment]`**: ~8
- **Patterns with `[no-any-return]`**: ~15
- **Other patterns**: ~18

## Category 1: `type: ignore[unreachable]` with hasattr checks

**Count**: 6 instances

**Locations**:
- `qme/potentials/tblite_potential.py:232` - `hasattr(self._calc, "set")`
- `qme/potentials/tblite_potential.py:326` - `hasattr(self._calc, "get_charges")`
- `qme/potentials/tblite_potential.py:339` - `hasattr(self._calc, "get_dipole_moment")`
- `qme/potentials/tblite_potential.py:352` - `hasattr(self._calc, "get_stress")`
- `qme/potentials/tblite_potential.py:434` - `hasattr(self._calc, "get_property")`
- `qme/potentials/so3lr_potential.py:130` - `self._calc.calculate()` call

**Pattern**:
```python
# In __init__: self._calc = None
# Later:
if self._calc is None:
    self._load_calculator()
assert self._calc is not None
if hasattr(self._calc, "get_charges"):  # type: ignore[unreachable]
    return self._calc.get_charges(atoms)
```

**Analysis**:
- `_calc` is initialized as `None` and typed as `Any | None` (implicitly)
- After `assert self._calc is not None`, mypy correctly narrows the type to non-None
- However, `hasattr` checks for **optional methods** are legitimate runtime checks
- The TBLite calculator may or may not implement these methods depending on version/configuration
- The issue: mypy thinks `hasattr` check is unreachable because we already asserted the object exists, but we're checking for method existence, not object existence
- **Verdict**: **KEEP** - These are legitimate defensive patterns for optional interface methods

**Root Cause**:
- Mypy's flow analysis treats `assert self._calc is not None` as making all code after it "safe"
- But `hasattr` is checking for method existence, not object existence
- This is a limitation of mypy's unreachable code detection

**Recommendation**:
1. **Option A**: Use Protocol types to define optional interfaces:
   ```python
   class TBLiteCalculatorProtocol(Protocol):
       def get_charges(self, atoms: Atoms | None = None) -> np.ndarray: ...
       def get_dipole_moment(self, atoms: Atoms | None = None) -> np.ndarray: ...
       # etc.
   ```
   Then use `isinstance` checks or `getattr` with defaults.

2. **Option B**: Use `getattr` with a default instead:
   ```python
   get_charges = getattr(self._calc, "get_charges", None)
   if get_charges is not None:
       return get_charges(atoms)
   ```

3. **Option C**: Keep as-is but add clear comments explaining why the ignore is needed

## Category 2: `type: ignore[unreachable]` with isinstance checks

**Count**: 8 instances

**Locations**:
- `qme/strategies/growing_string.py:54` - `isinstance(atoms_list, Atoms)` - API validation
- `qme/strategies/minima.py:86` - `isinstance(atoms_list, (list, tuple))` - Internal type narrowing
- `qme/strategies/ts.py:97` - `isinstance(atoms_list, list | tuple)` - Internal type narrowing
- `qme/strategies/irc.py:85` - `isinstance(atoms_list, (list, tuple))` - Internal type narrowing
- `qme/strategies/cineb.py:95` - `isinstance(path[0], list)` - Path flattening check
- `qme/strategies/neb.py:137` - `isinstance(atoms_list, Atoms)` - API validation
- `qme/strategies/minima_interpolate.py:119` - `isinstance(freq_result, dict)` - Type narrowing

**Pattern**:
```python
def run(self, atoms_list: list[Atoms], ...):
    # API boundary validation
    if isinstance(atoms_list, Atoms):  # type: ignore[unreachable]
        raise ValueError("Expected list")

    # OR internal type narrowing
    if not isinstance(atoms_list, (list, tuple)):
        single_input = True  # type: ignore[unreachable]
```

**Analysis**:

**Subcategory 2a: API Boundary Validation** (2 instances)
- Function signatures say `list[Atoms]`, but runtime could receive `Atoms` (API misuse/convenience)
- These are defensive checks for external API misuse
- Mypy correctly identifies these as unreachable based on type signature
- **Verdict**: **REDUNDANT** - Type signature should match reality OR these checks should be removed

**Subcategory 2b: Internal Type Narrowing** (4 instances)
- Code checks `isinstance(atoms_list, (list, tuple))` even though signature says `list[Atoms]`
- These are likely for handling both `list` and `tuple` inputs
- Mypy thinks `tuple` check is unreachable because signature says `list`
- **Verdict**: **FIX TYPE SIGNATURES** - Should be `list[Atoms] | tuple[Atoms, ...]` or `Sequence[Atoms]`

**Subcategory 2c: Path Flattening** (1 instance)
- `qme/strategies/cineb.py:95` - Checks if path contains nested lists
- This is a legitimate runtime check for data structure shape
- **Verdict**: **KEEP** - Runtime data validation

**Subcategory 2d: Result Type Narrowing** (1 instance)
- `qme/strategies/minima_interpolate.py:119` - Checks if frequency result is dict
- This narrows `Any` type from result dictionary
- **Verdict**: **KEEP** - Type narrowing for untyped results

**Recommendation**:
1. **For API validation** (growing_string, neb):
   - Option A: Change signature to `atoms_list: Atoms | list[Atoms]` and handle both cases
   - Option B: Remove the checks if they're truly unnecessary (document that API requires list)

2. **For internal narrowing** (minima, ts, irc):
   - Change signatures to `atoms_list: Sequence[Atoms]` or `atoms_list: list[Atoms] | tuple[Atoms, ...]`
   - This makes the isinstance checks meaningful and removes the need for ignores

3. **For path flattening and result narrowing**: Keep as-is, they're legitimate runtime checks

## Category 3: `type: ignore[unreachable]` in try/except blocks

**Count**: 3 instances

**Locations**:
- `qme/potentials/tblite_potential.py:269` - `try:` wrapping `self._calc.calculate()`
- `qme/strategies/neb.py:94` - `try:` for flattening nested paths
- `qme/potentials/so3lr_potential.py:130` - `self._calc.calculate()` call (no try, but marked unreachable)

**Pattern A** (External library calls):
```python
assert self._calc is not None
try:  # type: ignore[unreachable]
    self._calc.calculate(...)
except (AttributeError, RuntimeError) as e:
    raise RuntimeError(...)
```

**Pattern B** (Data structure handling):
```python
if path and hasattr(path[0], "__iter__") and not isinstance(path[0], Atoms):
    try:  # type: ignore[unreachable]
        flat = []
        for seg in path:
            if isinstance(seg, (list, tuple)):
                flat.extend(seg)
            else:
                flat.append(seg)
        path = flat
    except (TypeError, AttributeError):
        # If flattening fails, keep original path
        pass
```

**Analysis**:

**Subcategory 3a: External Library Calls** (2 instances)
- After `assert self._calc is not None`, mypy thinks the object is guaranteed to exist
- However, **external library calls can still raise exceptions** even with valid objects:
  - `AttributeError`: Method might not exist (though hasattr checked)
  - `RuntimeError`: Calculation might fail (convergence, numerical issues, etc.)
- These exceptions are **runtime errors from external code**, not type errors
- **Verdict**: **KEEP** - External library calls can fail even with valid objects

**Subcategory 3b: Data Structure Handling** (1 instance)
- `neb.py:94` - Defensive flattening of potentially nested paths
- The try/except handles edge cases where path structure is unexpected
- Mypy thinks the code is unreachable because the condition should prevent it
- **Verdict**: **DEFENSIVE OVER-CODING** - The condition already checks for this case

**Recommendation**:
1. **For external library calls** (tblite, so3lr):
   - Keep the try/except - external libraries can fail at runtime
   - Add a comment explaining why: "External library call can raise even with valid object"
   - Consider: Is the exception handling appropriate? Should we catch more specific exceptions?

2. **For data structure handling** (neb):
   - The try/except is likely unnecessary - the condition already checks for nested structures
   - Consider removing the try/except or simplifying the logic
   - If keeping, document why it's needed (defensive programming for malformed input)

## Category 4: `type: ignore[unreachable]` with early returns

**Count**: 2 instances

**Locations**:
- `qme/potentials/tblite_potential.py:160` - Early return in `_load_calculator()`
- `qme/potentials/so3lr_potential.py:79` - Early return in `_load_calculator()`

**Pattern**:
```python
def __init__(self):
    self._calc = None  # Typed implicitly as None

def _load_calculator(self) -> None:
    if self._calc is not None:
        return  # type: ignore[unreachable]
    # Load calculator...
```

**Analysis**:
- `_calc` is initialized as `None` in `__init__` without explicit type annotation
- Mypy infers `_calc: None` initially, then after assignment it becomes `Any`
- The check `if self._calc is not None: return` should be valid
- **Root Cause**: Mypy's flow analysis may be confused by the type narrowing
- **Verdict**: **TYPE ANNOTATION ISSUE** - Need explicit type annotation

**Recommendation**:
1. **Fix type annotation**:
   ```python
   def __init__(self):
       self._calc: Any | None = None
   ```
   This makes the early return check meaningful and removes the need for ignore.

2. **Alternative**: Use `# type: ignore[unreachable]` with comment explaining it's a mypy limitation

## Category 5: Try/except blocks accessing optional attributes

**Count**: 4 instances (no mypy ignore, but defensive patterns worth analyzing)

**Locations**:
- `qme/potentials/so3lr_potential.py:134-142` - Accessing `self._calc.results["energy"]`
- `qme/potentials/so3lr_potential.py:145-153` - Accessing `self._calc.results["forces"]`

**Pattern**:
```python
try:
    self.results["energy"] = self._model.results["energy"]
except (AttributeError, KeyError, TypeError):
    # Fallback: calculator doesn't have .results or key doesn't exist
    # AttributeError: .results doesn't exist
    # KeyError: key doesn't exist in results
    # TypeError: .results exists but isn't dict-like
    self.results["energy"] = self.results.get("energy")
```

**Analysis**:
- These are legitimate defensive patterns for optional attributes
- External calculators may not have `.results` or the key may not exist
- The exceptions caught are:
  - `AttributeError`: `.results` attribute doesn't exist
  - `KeyError`: Key doesn't exist in results dict
  - `TypeError`: `.results` exists but isn't dict-like (can't use `[]` indexing)
- **Verdict**: **KEEP** - Legitimate defensive programming for external library interfaces

**Recommendation**:
1. **Option A**: Use `getattr` with defaults (cleaner):
   ```python
   results = getattr(self._model, "results", {})
   if isinstance(results, dict):
       self.results["energy"] = results.get("energy", self.results.get("energy"))
   else:
       self.results["energy"] = self.results.get("energy")
   ```

2. **Option B**: Define Protocol types for calculator interfaces:
   ```python
   class CalculatorProtocol(Protocol):
       results: dict[str, Any]
       def calculate(self, atoms, properties, system_changes): ...
   ```
   Then use `isinstance` checks or type narrowing.

3. **Option C**: Keep as-is - the try/except is clear and handles all edge cases

## Category 6: `type: ignore[attr-defined]` for dynamic attributes

**Count**: 0 instances

**Note**: This category previously included TorchSim references, which have been removed.

## Category 7: `type: ignore[assignment]` patterns

**Count**: 8 instances

**Locations**:
- `qme/potentials/uma_potential.py:20` - `torch = None  # type: ignore[assignment, misc]`
- `qme/potentials/__init__.py:32` - `BasePotential = type(None)`
- `qme/potentials/__init__.py:45` - `MockCalculator = _MissingMock`
- `qme/cli/cli.py:220` - `ctx = nullcontext()`
- `qme/cli/cli.py:400` - `ctx = nullcontext()`
- `qme/cli/cli.py:607` - `ctx = nullcontext()`
- `qme/core/explorer.py:810` - `geom = filename_or_geom`
- `qme/optimizers/scipy_optimizers.py:188` - `self.fmax: float = getattr(self, "fmax", 0.05)`

**Pattern**:
```python
ctx = nullcontext()  # type: ignore[assignment]
```

**Analysis**:
- Some are fallback patterns for optional dependencies
- Some are type narrowing issues
- **Verdict**: **MIXED** - Some can be fixed with better typing, others are necessary

**Recommendation**:
- Use `TYPE_CHECKING` imports for optional dependencies
- Use proper type narrowing for conditional assignments

## Category 8: `type: ignore[no-any-return]` patterns

**Count**: ~15 instances

**Locations**:
- `qme/optimizers/ase_wrappers.py` - Multiple return statements from untyped calculator methods
- `qme/optimizers/scipy_optimizers.py:247, 278` - Return statements
- `qme/optimizers/rfo_optimizer.py:217` - Return statement

**Pattern**:
```python
return self.calculator.calculate(...)  # type: ignore[no-any-return]
```

**Analysis**:
- These are returns from untyped external libraries (ASE, scipy)
- **Verdict**: **KEEP** - External libraries are untyped

**Recommendation**:
- Use type stubs if available (types-ase, etc.)
- Or create Protocol types for calculator interfaces

## Category 9: `type: ignore[override]` patterns

**Count**: 3 instances

**Locations**:
- `qme/optimizers/scipy_optimizers.py:458` - `def run(...) -> bool`
- `qme/optimizers/scipy_optimizers.py:1607` - `def run(...) -> bool`
- `qme/optimizers/rfo_optimizer.py:748` - `def run(...) -> bool`

**Analysis**:
- These override methods from parent classes with different signatures
- **Verdict**: **INVESTIGATE** - May indicate design issue or legitimate override

**Recommendation**:
- Check if parent class signature can be updated
- Or use `@override` decorator if available

## Category 10: Test files and examples

**Count**: ~15 instances

**Locations**:
- Various test files with intentional type errors for testing
- Example files with type ignores for optional dependencies

**Analysis**:
- These are intentional for testing invalid inputs
- **Verdict**: **KEEP** - Testing patterns

## Recommendations Summary

### High Priority (Can be fixed)

1. **Early return unreachable** (Category 4) - Likely type annotation issues
2. **isinstance unreachable** (Category 2) - Can use Union types in signatures
3. **Assignment ignores** (Category 7) - Some can be fixed with better typing

### Medium Priority (Can be improved)

1. **hasattr unreachable** (Category 1) - Use Protocol types or getattr patterns
2. **Try/except unreachable** (Category 3) - Document why exceptions are possible
3. **Optional attributes** (Category 5) - Use getattr with defaults

### Low Priority (Keep as-is)

1. **External library types** (Category 6, 8) - Wait for library stubs or create Protocols
2. **Test files** (Category 10) - Intentional patterns

---

## Prioritized Refactoring Plan

### Phase 1: Quick Wins (Low Risk, High Impact)

**Goal**: Fix type annotation issues that cause false "unreachable" warnings

#### 1.1 Fix Early Return Type Annotations
**Files**: `qme/potentials/tblite_potential.py`, `qme/potentials/so3lr_potential.py`

**Changes**:
```python
# In __init__:
self._calc: Any | None = None  # Explicit type annotation
```

**Impact**: Removes 2 `type: ignore[unreachable]` comments
**Risk**: Low - Just adding type annotations
**Effort**: 5 minutes

#### 1.2 Fix Strategy Type Signatures
**Files**: `qme/strategies/minima.py`, `qme/strategies/ts.py`, `qme/strategies/irc.py`

**Changes**:
```python
# Change from:
def run(self, atoms_list: list[Atoms], ...):

# To:
from collections.abc import Sequence
def run(self, atoms_list: Sequence[Atoms], ...):
```

**Impact**: Removes 3 `type: ignore[unreachable]` comments, makes isinstance checks meaningful
**Risk**: Low - Only changes type signature, not behavior
**Effort**: 10 minutes

#### 1.3 Remove Redundant API Validation Checks
**Files**: `qme/strategies/growing_string.py`, `qme/strategies/neb.py`

**Changes**:
- Option A: Remove the `isinstance(atoms_list, Atoms)` checks (they're unreachable per type signature)
- Option B: Change signature to `atoms_list: Atoms | list[Atoms]` and handle both cases

**Impact**: Removes 2 `type: ignore[unreachable]` comments
**Risk**: Low if removing checks (type signature already enforces), Medium if changing signature
**Effort**: 15 minutes

**Total Phase 1**: ~30 minutes, removes 7 ignores

---

### Phase 2: Type System Improvements (Medium Risk, Medium Impact)

**Goal**: Use better typing patterns to eliminate defensive code

#### 2.1 Replace hasattr with getattr Pattern
**Files**: `qme/potentials/tblite_potential.py` (5 locations)

**Changes**:
```python
# From:
assert self._calc is not None
if hasattr(self._calc, "get_charges"):  # type: ignore[unreachable]
    return self._calc.get_charges(atoms)

# To:
assert self._calc is not None
get_charges = getattr(self._calc, "get_charges", None)
if get_charges is not None:
    return get_charges(atoms)
```

**Impact**: Removes 5 `type: ignore[unreachable]` comments
**Risk**: Low - Equivalent behavior, cleaner code
**Effort**: 20 minutes

#### 2.2 Improve Optional Attribute Access
**Files**: `qme/potentials/so3lr_potential.py`

**Changes**:
```python
# From:
try:
    self.results["energy"] = self._model.results["energy"]
except (AttributeError, KeyError, TypeError):
    self.results["energy"] = self.results.get("energy")

# To:
results = getattr(self._model, "results", {})
if isinstance(results, dict):
    self.results["energy"] = results.get("energy", self.results.get("energy"))
else:
    self.results["energy"] = self.results.get("energy")
```

**Impact**: No ignores removed, but cleaner code
**Risk**: Low - Equivalent behavior
**Effort**: 30 minutes

**Total Phase 2**: ~50 minutes, removes 5 ignores, improves code quality

---

### Phase 3: Documentation and Edge Cases (Low Risk, Low Impact)

**Goal**: Document legitimate defensive patterns and clean up edge cases

#### 3.1 Document External Library Call Defensive Patterns
**Files**: `qme/potentials/tblite_potential.py`, `qme/potentials/so3lr_potential.py`

**Changes**: Add comments explaining why try/except is needed:
```python
# External library call can raise RuntimeError even with valid object
# (e.g., convergence failures, numerical issues)
try:  # type: ignore[unreachable]
    self._calc.calculate(...)
```

**Impact**: Better documentation, no ignores removed
**Risk**: None
**Effort**: 10 minutes

#### 3.2 Simplify or Remove Unnecessary Try/Except
**Files**: `qme/strategies/neb.py:94`

**Changes**: Review if the try/except for path flattening is necessary given the condition check

**Impact**: May remove 1 ignore, cleaner code
**Risk**: Low - Review carefully
**Effort**: 15 minutes

**Total Phase 3**: ~25 minutes, improves documentation

---

### Phase 4: Advanced Type System (High Risk, High Impact)

**Goal**: Use Protocol types and advanced typing features

#### 4.1 Create Calculator Protocol Types
**Files**: New file `qme/potentials/protocols.py` or in `base_potential.py`

**Changes**:
```python
from typing import Protocol
from collections.abc import Sequence

class CalculatorProtocol(Protocol):
    results: dict[str, Any]
    def calculate(self, atoms, properties, system_changes): ...
    def get_potential_energy(self, atoms, force_consistent): ... -> float
    def get_forces(self, atoms): ... -> np.ndarray

class OptionalCalculatorProtocol(Protocol):
    """Calculator with optional methods."""
    results: dict[str, Any]
    def calculate(self, atoms, properties, system_changes): ...
    # Optional methods don't need to be in Protocol
```

Then use these in type hints:
```python
self._calc: CalculatorProtocol | None = None
```

**Impact**: Could eliminate many ignores, better type safety
**Risk**: Medium - Need to ensure all calculators conform
**Effort**: 2-3 hours

#### 4.2 (Removed - TorchSim no longer supported)
**Risk**: Low if using Protocol (structural typing)
**Effort**: 1 hour

**Total Phase 4**: ~4 hours, high impact but requires careful design

---

## Summary

- **Phase 1 (Quick Wins)**: 7 ignores removed, ~30 minutes
- **Phase 2 (Type Improvements)**: 5 ignores removed, ~50 minutes
- **Phase 3 (Documentation)**: 0-1 ignores removed, ~25 minutes
- **Phase 4 (Advanced)**: Potentially 10+ ignores removed, ~4 hours

**Total Estimated Impact**:
- **Low-effort fixes**: 12 ignores removed in ~1.5 hours
- **Full refactoring**: 20+ ignores removed in ~6 hours

**Recommendation**: Start with Phase 1 and Phase 2 for immediate wins, then evaluate if Phase 4 is worth the effort based on codebase needs.
