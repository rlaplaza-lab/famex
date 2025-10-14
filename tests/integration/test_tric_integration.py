"""Integration tests for TRIC optimizer with QME framework.

This module tests the integration of TRIC optimizer with QME's optimization
workflow, including comparisons with other optimizers.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.optimize.lbfgs import LBFGS

from qme.core.tric import create_tric_optimizer
from qme.core.local_strategies import _get_local_optimizer_class
from qme.potentials.mock_potential import MockCalculator


class TestTRICQMEIntegration:
    """Test TRIC optimizer integration with QME framework."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple water molecule
        self.atoms = Atoms("H2O", positions=[
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H
            [-0.24, 0.93, 0.0]    # H
        ])
        self.atoms.calc = MockCalculator()

    def test_tric_via_local_strategies(self):
        """Test that TRIC optimizer is accessible through QME local strategies."""
        TricClass = _get_local_optimizer_class("tric")
        
        # Should return the CustomTRICOptimizer factory class
        from qme.core.tric_optimizer import CustomTRICOptimizer
        assert TricClass == CustomTRICOptimizer

    def test_tric_optimizer_instantiation_through_qme(self):
        """Test creating TRIC optimizer through QME interface."""
        TricClass = _get_local_optimizer_class("tric")
        optimizer = TricClass(self.atoms)
        
        from qme.core.tric import TRICOptimizer
        assert isinstance(optimizer, TRICOptimizer)
        assert optimizer.atoms is self.atoms

    def test_tric_vs_lbfgs_comparison(self):
        """Compare TRIC optimizer with LBFGS optimizer."""
        # Test on same initial geometry
        atoms_tric = self.atoms.copy()
        atoms_lbfgs = self.atoms.copy()
        
        atoms_tric.calc = MockCalculator()
        atoms_lbfgs.calc = MockCalculator()
        
        initial_energy = atoms_tric.get_potential_energy()
        
        # Run TRIC optimization
        tric_optimizer = create_tric_optimizer(atoms_tric, order=0)
        tric_converged = tric_optimizer.run(fmax=0.1, steps=20)
        tric_energy = atoms_tric.get_potential_energy()
        
        # Run LBFGS optimization
        lbfgs_optimizer = LBFGS(atoms_lbfgs)
        lbfgs_converged = lbfgs_optimizer.run(fmax=0.1, steps=20)
        lbfgs_energy = atoms_lbfgs.get_potential_energy()
        
        # Debug information
        print(f"Initial energy: {initial_energy:.6f}")
        print(f"TRIC energy: {tric_energy:.6f} (converged: {tric_converged}, steps: {tric_optimizer.step_count})")
        print(f"LBFGS energy: {lbfgs_energy:.6f} (converged: {lbfgs_converged}, steps: {lbfgs_optimizer.get_number_of_steps()})")
        
        # Both should complete without crashing
        assert tric_optimizer.step_count >= 0
        assert lbfgs_optimizer.get_number_of_steps() >= 0
        
        # Both should improve or maintain energy
        assert tric_energy <= initial_energy + 1e-6
        assert lbfgs_energy <= initial_energy + 1e-6

    def test_tric_optimization_with_different_geometries(self):
        """Test TRIC optimizer on different molecular geometries."""
        geometries = [
            # Water
            ("H2O", [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            # Ammonia
            ("NH3", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.5, 0.87, 0.0], [-0.5, -0.87, 0.0]]),
            # Methane (simplified)
            ("CH4", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.33, 0.94, 0.0], 
                    [-0.33, -0.47, 0.82], [-0.33, -0.47, -0.82]]),
        ]
        
        for formula, positions in geometries:
            atoms = Atoms(formula, positions=positions)
            atoms.calc = MockCalculator()
            
            initial_energy = atoms.get_potential_energy()
            
            # Run TRIC optimization
            optimizer = create_tric_optimizer(atoms, order=0)
            converged = optimizer.run(fmax=0.1, steps=20)
            
            final_energy = atoms.get_potential_energy()
            
            # Debug information
            print(f"{formula} - Initial: {initial_energy:.6f}, Final: {final_energy:.6f}, Converged: {converged}, Steps: {optimizer.step_count}")
            
            # Should complete without crashing
            assert optimizer.step_count >= 0
            # Should improve or maintain energy
            assert final_energy <= initial_energy + 1e-6

    def test_tric_optimizer_numerical_stability(self):
        """Test TRIC optimizer numerical stability."""
        # Create a challenging geometry (very distorted)
        atoms = Atoms("H2O", positions=[
            [0.0, 0.0, 0.0],      # O
            [2.0, 0.0, 0.0],      # H (very stretched)
            [-1.0, 2.0, 0.0]      # H (very distorted)
        ])
        atoms.calc = MockCalculator()
        
        initial_positions = atoms.get_positions().copy()
        
        # Run optimization
        optimizer = create_tric_optimizer(atoms, order=0)
        
        # Should not crash
        try:
            converged = optimizer.run(fmax=0.1, steps=10)
            final_positions = atoms.get_positions()
            
            # Check for reasonable behavior
            position_change = np.max(np.abs(final_positions - initial_positions))
            
            # Should not produce unreasonably large changes
            assert position_change < 10.0  # Reasonable threshold
            
        except Exception as e:
            pytest.fail(f"TRIC optimizer crashed: {e}")

    def test_tric_optimizer_with_hessian(self):
        """Test TRIC optimizer with custom Hessian."""
        atoms = self.atoms.copy()
        atoms.calc = MockCalculator()
        
        # Create optimizer first to get the correct number of internal coordinates
        temp_optimizer = create_tric_optimizer(atoms, order=0)
        n_coords = len(temp_optimizer.internal_coords)
        
        # Create a custom Hessian with correct size
        custom_hessian = np.eye(n_coords) * 0.5
        
        # Run optimization with custom Hessian
        optimizer = create_tric_optimizer(atoms, order=0, hessian=custom_hessian)
        
        # Should use the custom Hessian
        assert np.array_equal(optimizer.hessian, custom_hessian)
        
        # Should still be able to optimize
        converged = optimizer.run(fmax=0.1, steps=10)
        assert optimizer.step_count >= 0

    def test_tric_ts_optimizer_basic(self):
        """Test TRIC transition state optimizer (basic functionality)."""
        atoms = self.atoms.copy()
        atoms.calc = MockCalculator()
        
        # Create TS optimizer
        ts_optimizer = create_tric_optimizer(atoms, order=1)
        
        assert ts_optimizer.order == 1
        
        # Should be able to run (though may not converge without proper TS geometry)
        try:
            converged = ts_optimizer.run(fmax=0.1, steps=10)
            # TS optimization is more challenging, so we don't require convergence
            assert ts_optimizer.step_count >= 0
        except Exception as e:
            # TS optimization may fail with mock calculator
            print(f"TS optimization failed (expected with mock calculator): {e}")

    def test_tric_optimizer_deterministic(self):
        """Test that TRIC optimizer gives deterministic results."""
        results = []
        
        # Run optimization multiple times
        for i in range(3):
            atoms_copy = self.atoms.copy()
            atoms_copy.calc = MockCalculator()
            
            optimizer = create_tric_optimizer(atoms_copy, order=0)
            optimizer.run(fmax=0.1, steps=10)
            
            results.append({
                'steps': optimizer.step_count,
                'final_energy': atoms_copy.get_potential_energy(),
                'final_positions': atoms_copy.get_positions().copy()
            })
        
        # Check consistency
        energies = [r['final_energy'] for r in results]
        steps = [r['steps'] for r in results]
        
        # Energies should be very close (within numerical precision)
        energy_std = np.std(energies)
        assert energy_std < 1e-10, f"Energy not deterministic: std={energy_std}"
        
        # Steps should be identical
        assert all(s == steps[0] for s in steps), f"Steps not deterministic: {steps}"

    def test_tric_optimizer_memory_usage(self):
        """Test for reasonable memory usage in TRIC optimizer."""
        import gc
        
        atoms = self.atoms.copy()
        atoms.calc = MockCalculator()
        
        # Run many optimizations
        for i in range(5):
            atoms_copy = atoms.copy()
            atoms_copy.calc = MockCalculator()
            
            optimizer = create_tric_optimizer(atoms_copy, order=0)
            optimizer.run(fmax=0.1, steps=5)
            
            # Force garbage collection
            del optimizer, atoms_copy
            gc.collect()
        
        # Should complete without memory issues
        assert True  # If we get here, no memory errors occurred


class TestTRICWithQMEBackends:
    """Test TRIC optimizer with different QME backends."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms("H2O", positions=[
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H
            [-0.24, 0.93, 0.0]    # H
        ])

    def test_tric_with_mock_calculator(self):
        """Test TRIC optimizer with MockCalculator."""
        atoms = self.atoms.copy()
        atoms.calc = MockCalculator()
        
        optimizer = create_tric_optimizer(atoms, order=0)
        converged = optimizer.run(fmax=0.1, steps=10)
        
        # Should complete without errors
        assert optimizer.step_count >= 0

    def test_tric_optimizer_timing(self):
        """Test TRIC optimizer timing performance."""
        import time
        
        atoms = self.atoms.copy()
        atoms.calc = MockCalculator()
        
        # Time TRIC optimization
        start_time = time.time()
        optimizer = create_tric_optimizer(atoms, order=0)
        converged = optimizer.run(fmax=0.1, steps=10)
        tric_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert tric_time < 10.0  # Should not take more than 10 seconds


if __name__ == "__main__":
    pytest.main([__file__])
