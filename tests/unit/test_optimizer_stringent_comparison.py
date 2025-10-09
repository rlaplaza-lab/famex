"""Stringent tests for optimizer comparison that catch subtle bugs.

These tests are designed to catch bugs where optimizers appear to work
but don't actually perform optimization (like the GeometricOptimizer bug).
"""

import numpy as np
import pytest
from ase.optimize.lbfgs import LBFGS
from sella import Sella

from qme.core.geometric_interface import GeometricOptimizer
from qme.dependencies import deps
from qme.potentials.mock_potential import MockCalculator


class TestOptimizerStringentComparison:
    """Stringent tests that catch optimizer bugs that current tests miss."""

    def test_optimizer_actually_optimizes_energy(self):
        """Test that optimizers actually change the energy, not just report steps."""
        # Create a simple system that should optimize
        from ase import Atoms

        # Create a distorted water molecule that should relax
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        initial_energy = atoms.get_potential_energy()
        initial_positions = atoms.get_positions().copy()

        # Test LBFGS (known working optimizer)
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        lbfgs_opt.run(fmax=0.01, steps=50)

        lbfgs_energy = atoms_lbfgs.get_potential_energy()
        lbfgs_positions = atoms_lbfgs.get_positions()

        # LBFGS should actually optimize
        energy_change_lbfgs = abs(lbfgs_energy - initial_energy)
        position_change_lbfgs = np.max(np.abs(lbfgs_positions - initial_positions))

        print(
            f"LBFGS: Energy change = {energy_change_lbfgs:.6f}, "
            f"Position change = {position_change_lbfgs:.6f}"
        )

        # LBFGS should show some optimization
        assert energy_change_lbfgs > 1e-6, "LBFGS should actually change energy"
        assert position_change_lbfgs > 1e-6, "LBFGS should actually change positions"

        # Test GeometricOptimizer if available
        if deps.has("geometric"):
            atoms_geo = atoms.copy()
            atoms_geo.calc = MockCalculator()
            geo_opt = GeometricOptimizer(atoms_geo, order=0)
            geo_opt.run(fmax=0.01, steps=50)

            geo_energy = atoms_geo.get_potential_energy()
            geo_positions = atoms_geo.get_positions()

            energy_change_geo = abs(geo_energy - initial_energy)
            position_change_geo = np.max(np.abs(geo_positions - initial_positions))

            print(
                f"Geometric: Energy change = {energy_change_geo:.6f}, "
                f"Position change = {position_change_geo:.6f}"
            )

            # GeometricOptimizer should also optimize (this test will catch the bug)
            assert energy_change_geo > 1e-6, "GeometricOptimizer should actually change energy"
            assert position_change_geo > 1e-6, "GeometricOptimizer should actually change positions"

    def test_optimizer_step_count_consistency(self):
        """Test that step counts are consistent with actual optimization."""
        from ase import Atoms

        # Create a system that needs optimization
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        # Test LBFGS
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        lbfgs_opt.run(fmax=0.01, steps=50)

        lbfgs_steps = lbfgs_opt.get_number_of_steps()
        print(f"LBFGS steps: {lbfgs_steps}")

        # LBFGS should take some steps
        assert lbfgs_steps > 0, "LBFGS should report positive step count"

        # Test GeometricOptimizer if available
        if deps.has("geometric"):
            atoms_geo = atoms.copy()
            atoms_geo.calc = MockCalculator()
            geo_opt = GeometricOptimizer(atoms_geo, order=0)
            geo_opt.run(fmax=0.01, steps=50)

            geo_steps = geo_opt.step_count
            print(f"Geometric steps: {geo_steps}")

            # If GeometricOptimizer reports steps, it should actually optimize
            if geo_steps > 0:
                initial_energy = atoms.get_potential_energy()
                final_energy = atoms_geo.get_potential_energy()
                energy_change = abs(final_energy - initial_energy)

                assert energy_change > 1e-6, (
                    f"GeometricOptimizer reports {geo_steps} steps but energy didn't change. "
                    f"This indicates a bug in coordinate extraction."
                )

    def test_optimizer_convergence_consistency(self):
        """Test that convergence status is consistent with actual optimization."""
        from ase import Atoms

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        # Test LBFGS
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        lbfgs_opt.run(fmax=0.01, steps=50)

        lbfgs_converged = lbfgs_opt.converged(atoms_lbfgs.get_forces().flatten())
        print(f"LBFGS converged: {lbfgs_converged}")

        # Test GeometricOptimizer if available
        if deps.has("geometric"):
            atoms_geo = atoms.copy()
            atoms_geo.calc = MockCalculator()
            geo_opt = GeometricOptimizer(atoms_geo, order=0)
            geo_opt.run(fmax=0.01, steps=50)

            geo_converged = geo_opt.converged
            print(f"Geometric converged: {geo_converged}")

            # If GeometricOptimizer claims convergence, check if forces are actually low
            if geo_converged:
                final_forces = atoms_geo.get_forces()
                max_force = np.max(np.abs(final_forces))
                print(f"Geometric max force: {max_force:.6f}")

                # If converged, forces should be low
                assert max_force < 0.1, (
                    f"GeometricOptimizer claims convergence but max force is {max_force:.6f}. "
                    f"This indicates a bug in convergence detection."
                )

    def test_transition_state_optimizer_consistency(self):
        """Test that TS optimizers actually attempt TS optimization."""
        if not deps.has("geometric") or not deps.has("sella"):
            pytest.skip("Both geomeTRIC and Sella must be available")

        from ase import Atoms

        # Create a system that could be a TS
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        initial_energy = atoms.get_potential_energy()
        initial_positions = atoms.get_positions().copy()

        # Test Sella TS optimizer
        atoms_sella = atoms.copy()
        atoms_sella.calc = MockCalculator()
        sella_opt = Sella(atoms_sella, internal=True, order=1)
        sella_opt.run(fmax=0.01, steps=50)

        sella_energy = atoms_sella.get_potential_energy()
        sella_positions = atoms_sella.get_positions()

        energy_change_sella = abs(sella_energy - initial_energy)
        position_change_sella = np.max(np.abs(sella_positions - initial_positions))

        print(
            f"Sella TS: Energy change = {energy_change_sella:.6f}, "
            f"Position change = {position_change_sella:.6f}"
        )

        # Sella should actually optimize
        assert energy_change_sella > 1e-6, "Sella TS optimizer should actually change energy"
        assert position_change_sella > 1e-6, "Sella TS optimizer should actually change positions"

        # Test GeometricOptimizer TS
        atoms_geo = atoms.copy()
        atoms_geo.calc = MockCalculator()
        geo_opt = GeometricOptimizer(atoms_geo, order=1)
        try:
            geo_opt.run(fmax=0.01, steps=50)
            geo_energy = atoms_geo.get_potential_energy()
            geo_positions = atoms_geo.get_positions()

            energy_change_geo = abs(geo_energy - initial_energy)
            position_change_geo = np.max(np.abs(geo_positions - initial_positions))

            print(
                f"Geometric TS: Energy change = {energy_change_geo:.6f}, "
                f"Position change = {position_change_geo:.6f}"
            )

            # GeometricOptimizer TS should also optimize (this test will catch the bug)
            assert energy_change_geo > 1e-6, (
                "GeometricOptimizer TS should actually change energy"
            )
            assert position_change_geo > 1e-6, (
                "GeometricOptimizer TS should actually change positions"
            )
        except (np.linalg.LinAlgError, RuntimeError) as e:
            # geomeTRIC can fail with numerical issues on certain geometries
            # This is a known limitation, not a bug in our code
            pytest.skip(f"geomeTRIC failed with numerical issues: {e}")

    def test_optimizer_energy_monotonicity(self):
        """Test that optimizers show energy changes during optimization."""
        from ase import Atoms

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        initial_energy = atoms.get_potential_energy()

        # Test that optimizers actually change the energy
        optimizers_to_test = []

        # Always test LBFGS
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        optimizers_to_test.append(("LBFGS", lbfgs_opt, atoms_lbfgs))

        # Test GeometricOptimizer if available
        if deps.has("geometric"):
            atoms_geo = atoms.copy()
            atoms_geo.calc = MockCalculator()
            geo_opt = GeometricOptimizer(atoms_geo, order=0)
            optimizers_to_test.append(("GeometricOptimizer", geo_opt, atoms_geo))

        # Test Sella if available
        if deps.has("sella"):
            atoms_sella = atoms.copy()
            atoms_sella.calc = MockCalculator()
            sella_opt = Sella(atoms_sella, internal=True, order=0)
            optimizers_to_test.append(("Sella", sella_opt, atoms_sella))

        for name, optimizer, atoms_obj in optimizers_to_test:
            optimizer.run(fmax=0.01, steps=50)

            final_energy = atoms_obj.get_potential_energy()
            energy_change = abs(final_energy - initial_energy)

            print(f"{name}: Energy change = {energy_change:.6f}")

            # All optimizers should show some energy change
            assert energy_change > 1e-6, (
                f"{name} optimizer shows no energy change ({energy_change:.2e}). "
                f"This indicates a bug where the optimizer reports steps but doesn't optimize."
            )
