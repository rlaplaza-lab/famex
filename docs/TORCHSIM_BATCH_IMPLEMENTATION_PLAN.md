# TorchSim Batch Evaluation Implementation Plan

## Overview

This document outlines the implementation plan for adding batch evaluation capabilities to TorchSim backends in QME. The goal is to leverage TorchSim's automatic batching and GPU memory management to significantly speed up calculations that involve multiple structures, such as frequency analysis and NEB calculations.

## Key Benefits

1. **Automatic Batching**: TorchSim can process multiple structures simultaneously
2. **GPU Memory Management**: Efficient memory usage for large batches
3. **Significant Speedup**: Up to 100x speedup over traditional ASE-based approaches
4. **Backward Compatibility**: Non-TorchSim backends continue to work as before

## Target Modules

### 1. Frequency Analysis (`qme/analysis/frequency.py`)

**Current Implementation**: Uses finite differences with individual energy/force calculations
**Batch Opportunity**: Calculate energies and forces for all displaced structures simultaneously

**Key Changes**:
- Add `supports_batch_evaluation` property to calculators
- Modify `FrequencyAnalysis.calculate_hessian()` to use batch evaluation when available
- Create `BatchFrequencyAnalysis` class for TorchSim backends
- Fall back to individual calculations for non-TorchSim backends

**Implementation Strategy**:
```python
class FrequencyAnalysis:
    def calculate_hessian(self, method="auto"):
        if self._supports_batch_evaluation():
            return self._calculate_hessian_batch()
        else:
            return self._calculate_hessian_individual()
    
    def _supports_batch_evaluation(self):
        return (hasattr(self.calculator, 'supports_batch_evaluation') and 
                self.calculator.supports_batch_evaluation)
```

### 2. NEB in Explorer (`qme/core/twoended_strategies.py`)

**Current Implementation**: Individual optimization of each image in the NEB path
**Batch Opportunity**: Calculate energies and forces for all NEB images simultaneously

**Key Changes**:
- Add batch evaluation support to NEB runner
- Create `BatchNEBRunner` for TorchSim backends
- Modify `twoended_neb_runner()` to detect TorchSim backends
- Implement batch force/energy calculations for NEB images

**Implementation Strategy**:
```python
def twoended_neb_runner(atoms_list, ...):
    if _supports_batch_evaluation(explorer.calculator):
        return _batch_neb_runner(atoms_list, ...)
    else:
        return _individual_neb_runner(atoms_list, ...)
```

## Implementation Details

### 1. Calculator Interface Extensions

Add batch evaluation capabilities to the calculator interface:

```python
class BasePotential:
    @property
    def supports_batch_evaluation(self):
        """Whether this calculator supports batch evaluation."""
        return False
    
    def calculate_batch(self, atoms_list, properties=None):
        """Calculate properties for a batch of structures."""
        raise NotImplementedError("Batch evaluation not supported")
```

### 2. TorchSim Batch Implementation

Extend `TorchSimPotential` to support batch calculations:

```python
class TorchSimPotential(BasePotential):
    @property
    def supports_batch_evaluation(self):
        return True
    
    def calculate_batch(self, atoms_list, properties=None):
        """Calculate properties for multiple structures simultaneously."""
        # Convert all atoms to states
        states = [self._atoms_to_state(atoms) for atoms in atoms_list]
        
        # Batch the states (TorchSim handles this automatically)
        batch_state = self._batch_states(states)
        
        # Calculate properties for the entire batch
        results = self._model(batch_state)
        
        # Split results back to individual structures
        return self._split_batch_results(results, len(atoms_list))
```

### 3. Frequency Analysis Batch Implementation

Create a batch-enabled frequency analysis class:

```python
class BatchFrequencyAnalysis(FrequencyAnalysis):
    def calculate_hessian(self, method="auto"):
        if method == "auto":
            method = "batch" if self._supports_batch_evaluation() else "finite_differences"
        
        if method == "batch":
            return self._calculate_hessian_batch()
        else:
            return super().calculate_hessian(method)
    
    def _calculate_hessian_batch(self):
        """Calculate Hessian using batch evaluation."""
        # Generate all displaced structures
        displaced_structures = self._generate_displaced_structures()
        
        # Calculate energies and forces for all structures in one batch
        batch_results = self.calculator.calculate_batch(
            displaced_structures, 
            properties=["energy", "forces"]
        )
        
        # Construct Hessian matrix from batch results
        return self._construct_hessian_from_batch(batch_results)
```

### 4. NEB Batch Implementation

Create a batch-enabled NEB runner:

```python
def _batch_neb_runner(atoms_list, explorer, **kwargs):
    """NEB runner with batch evaluation support."""
    # Generate initial path
    path = path_generator(atoms_list, **kwargs)
    
    # Create batch NEB optimizer
    batch_neb = BatchNEBOptimizer(
        path, 
        calculator=explorer.calculator,
        **kwargs
    )
    
    # Optimize using batch evaluation
    optimized_path = batch_neb.optimize()
    
    return optimized_path

class BatchNEBOptimizer:
    def __init__(self, path, calculator, **kwargs):
        self.path = path
        self.calculator = calculator
        self.kwargs = kwargs
    
    def optimize(self):
        """Optimize NEB path using batch evaluation."""
        for step in range(self.kwargs.get('steps', 1000)):
            # Calculate forces for all images in one batch
            forces = self.calculator.calculate_batch(
                self.path, 
                properties=["forces"]
            )
            
            # Apply NEB forces (spring + nudging)
            neb_forces = self._apply_neb_forces(forces)
            
            # Update positions
            self._update_positions(neb_forces)
            
            # Check convergence
            if self._is_converged(neb_forces):
                break
        
        return self.path
```

## Backward Compatibility

### 1. Automatic Detection

The system will automatically detect whether a calculator supports batch evaluation:

```python
def _supports_batch_evaluation(calculator):
    """Check if calculator supports batch evaluation."""
    return (hasattr(calculator, 'supports_batch_evaluation') and 
            calculator.supports_batch_evaluation)
```

### 2. Graceful Fallback

If batch evaluation is not supported, the system falls back to individual calculations:

```python
def calculate_frequencies(atoms, calculator, **kwargs):
    if _supports_batch_evaluation(calculator):
        return BatchFrequencyAnalysis(atoms, calculator, **kwargs).run()
    else:
        return FrequencyAnalysis(atoms, calculator, **kwargs).run()
```

## Performance Considerations

### 1. Memory Management

- TorchSim automatically manages GPU memory for batches
- Implement batch size limits to prevent OOM errors
- Add memory monitoring and automatic batch size reduction

### 2. Batch Size Optimization

- Start with small batches and increase based on available memory
- Implement adaptive batch sizing based on structure size
- Add configuration options for maximum batch size

### 3. Error Handling

- Handle batch calculation failures gracefully
- Fall back to individual calculations if batch fails
- Provide detailed error messages for debugging

## Testing Strategy

### 1. Unit Tests

- Test batch evaluation with small structures
- Verify results match individual calculations
- Test error handling and fallback behavior

### 2. Integration Tests

- Test frequency analysis with batch evaluation
- Test NEB with batch evaluation
- Compare performance with individual calculations

### 3. Performance Tests

- Benchmark batch vs individual calculations
- Test with various structure sizes
- Measure memory usage and GPU utilization

## Implementation Phases

### Phase 1: Core Infrastructure
1. Add batch evaluation interface to `BasePotential`
2. Implement batch evaluation in `TorchSimPotential`
3. Add batch detection utilities

### Phase 2: Frequency Analysis
1. Create `BatchFrequencyAnalysis` class
2. Modify `FrequencyAnalysis` to use batch when available
3. Add tests and benchmarks

### Phase 3: NEB Implementation
1. Create `BatchNEBOptimizer` class
2. Modify `twoended_neb_runner` to use batch evaluation
3. Add tests and benchmarks

### Phase 4: Optimization and Polish
1. Optimize batch size and memory usage
2. Add comprehensive error handling
3. Update documentation and examples

## Expected Performance Gains

- **Frequency Analysis**: 10-50x speedup for large molecules
- **NEB Calculations**: 5-20x speedup depending on path length
- **Memory Usage**: More efficient GPU memory utilization
- **Overall**: Significant speedup for multi-structure calculations

## Conclusion

This implementation plan provides a comprehensive approach to adding batch evaluation capabilities to TorchSim backends while maintaining backward compatibility with existing backends. The phased approach ensures that each component is thoroughly tested before moving to the next phase, minimizing the risk of introducing bugs or breaking existing functionality.
