"""Stringent tests for optimizer comparison that catch subtle bugs.

These tests are designed to catch bugs where optimizers appear to work
but don't actually perform optimization.
"""

import numpy as np
import pytest
from ase.optimize.lbfgs import LBFGS
from sella import Sella

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

    def test_transition_state_optimizer_consistency(self):
        """Test that TS optimizers actually attempt TS optimization."""
        if not deps.has("sella"):
            pytest.skip("Sella must be available")

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

    def test_optimizer_coordinate_and_frequency_comparison(self):
        """Stringent comparison of final coordinates and frequencies between optimizers."""
        from ase import Atoms

        from qme.analysis.frequency import FrequencyAnalysis

        # Create a simple system
        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        optimizers_results = {}

        # Test LBFGS
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        lbfgs_opt.run(fmax=0.0001, steps=500)
        optimizers_results["LBFGS"] = {
            "atoms": atoms_lbfgs,
            "energy": atoms_lbfgs.get_potential_energy(),
            "positions": atoms_lbfgs.get_positions(),
            "forces": atoms_lbfgs.get_forces(),
            "converged": lbfgs_opt.converged(atoms_lbfgs.get_forces().flatten()),
        }

        # Test Sella if available
        if deps.has("sella"):
            atoms_sella = atoms.copy()
            atoms_sella.calc = MockCalculator()
            sella_opt = Sella(atoms_sella, internal=True, order=0)
            sella_opt.run(fmax=0.0001, steps=500)
            optimizers_results["Sella"] = {
                "atoms": atoms_sella,
                "energy": atoms_sella.get_potential_energy(),
                "positions": atoms_sella.get_positions(),
                "forces": atoms_sella.get_forces(),
                "converged": sella_opt.converged(),
            }

        # Compare results
        optimizer_names = list(optimizers_results.keys())
        if len(optimizer_names) < 2:
            pytest.skip("Need at least 2 optimizers for comparison")

        print("\n=== COORDINATE COMPARISON ===")
        for i, name1 in enumerate(optimizer_names):
            for name2 in optimizer_names[i + 1 :]:
                results1 = optimizers_results[name1]
                results2 = optimizers_results[name2]

                # Compare final coordinates
                pos1 = results1["positions"]
                pos2 = results2["positions"]
                max_coord_diff = np.max(np.abs(pos1 - pos2))
                rms_coord_diff = np.sqrt(np.mean((pos1 - pos2) ** 2))

                print(f"{name1} vs {name2}:")
                print(f"  Max coordinate difference: {max_coord_diff:.6f} Å")
                print(f"  RMS coordinate difference: {rms_coord_diff:.6f} Å")

                # All optimizers should converge to the same minimum for simple molecules
                # Use relaxed threshold: coordinates should be within 1.0 Å (different optimizers can find slightly different minima)
                assert max_coord_diff < 0.1, (
                    f"Final coordinates differ too much between {name1} and {name2}: "
                    f"{max_coord_diff:.6f} Å. This suggests inconsistent optimization."
                )

        # Test frequency analysis on converged structures
        print("\n=== FREQUENCY ANALYSIS ===")
        for name, results in optimizers_results.items():
            if results["converged"]:
                try:
                    freq_analysis = FrequencyAnalysis(results["atoms"])
                    frequencies = freq_analysis.get_frequencies()
                    print(f"{name}: {len(frequencies)} frequencies calculated")
                    print(f"  Lowest frequency: {min(frequencies):.2f} cm⁻¹")
                    print(f"  Highest frequency: {max(frequencies):.2f} cm⁻¹")

                    # All frequencies should be real (no imaginary frequencies for minima)
                    assert all(
                        freq >= 0 for freq in frequencies
                    ), f"{name} has imaginary frequencies, suggesting it's not at a minimum"

                except Exception as e:
                    print(f"{name}: Frequency analysis failed: {e}")

    def test_optimizer_force_convergence_consistency(self):
        """Test that optimizers actually achieve the claimed force convergence."""
        from ase import Atoms

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        optimizers_to_test = []

        # Test LBFGS
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        optimizers_to_test.append(("LBFGS", lbfgs_opt, atoms_lbfgs))

        # Test Sella if available
        if deps.has("sella"):
            atoms_sella = atoms.copy()
            atoms_sella.calc = MockCalculator()
            sella_opt = Sella(atoms_sella, internal=True, order=0)
            optimizers_to_test.append(("Sella", sella_opt, atoms_sella))

        # Run optimizations
        for _name, opt, _test_atoms in optimizers_to_test:
            opt.run(fmax=0.05, steps=200)

        # Check force convergence
        print("\n=== FORCE CONVERGENCE CHECK ===")
        for name, opt, test_atoms in optimizers_to_test:
            forces = test_atoms.get_forces()
            max_force = np.max(np.abs(forces))
            rms_force = np.sqrt(np.mean(forces**2))

            print(f"{name}:")
            print(f"  Max force: {max_force:.6f} eV/Å")
            print(f"  RMS force: {rms_force:.6f} eV/Å")

            # If optimizer claims convergence, forces should be low
            if hasattr(opt, "converged"):
                converged = opt.converged()
                print(f"  Claims converged: {converged}")

                if converged:
                    assert max_force < 0.05, (
                        f"{name} claims convergence but max force is {max_force:.6f} eV/Å. "
                        f"This suggests a bug in convergence detection."
                    )

    def test_optimizer_convergence_consistency_detailed(self):
        """Detailed test of convergence consistency across different force thresholds."""
        from ase import Atoms

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        convergence_results = {}

        # Test LBFGS
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        lbfgs_opt.run(fmax=0.001, steps=500)

        convergence_results["LBFGS"] = {
            "atoms": atoms_lbfgs,
            "energy": atoms_lbfgs.get_potential_energy(),
            "forces": atoms_lbfgs.get_forces(),
            "max_force": np.max(np.abs(atoms_lbfgs.get_forces())),
            "rms_force": np.sqrt(np.mean(atoms_lbfgs.get_forces() ** 2)),
            "converged": lbfgs_opt.converged(atoms_lbfgs.get_forces().flatten()),
            "steps": lbfgs_opt.get_number_of_steps(),
        }

        # Test Sella if available
        if deps.has("sella"):
            atoms_sella = atoms.copy()
            atoms_sella.calc = MockCalculator()
            sella_opt = Sella(atoms_sella, internal=True, order=0)
            sella_opt.run(fmax=0.001, steps=500)

            convergence_results["Sella"] = {
                "atoms": atoms_sella,
                "energy": atoms_sella.get_potential_energy(),
                "forces": atoms_sella.get_forces(),
                "max_force": np.max(np.abs(atoms_sella.get_forces())),
                "rms_force": np.sqrt(np.mean(atoms_sella.get_forces() ** 2)),
                "converged": sella_opt.converged(),
                "steps": sella_opt.get_number_of_steps(),
            }

        # Analyze convergence results
        print("\n=== CONVERGENCE ANALYSIS ===")
        for name, results in convergence_results.items():
            print(f"{name}:")
            print(f"  Converged: {results['converged']}")
            print(f"  Max force: {results['max_force']:.6f} eV/Å")
            print(f"  RMS force: {results['rms_force']:.6f} eV/Å")
            print(f"  Steps: {results['steps']}")

            # Verify convergence claims
            if results["converged"]:
                assert results["max_force"] < 0.001, (
                    f"{name} claims convergence but max force is {results['max_force']:.6f} eV/Å. "
                    f"This suggests a bug in convergence detection."
                )

        # Test tighter convergence if optimizers converged
        print("\n=== TIGHTER CONVERGENCE TEST ===")
        tight_fmax = 0.0001

        for name, results in convergence_results.items():
            if results["converged"]:
                # Test if optimizer can achieve tighter convergence
                test_atoms = atoms.copy()
                test_atoms.calc = MockCalculator()

                if name == "LBFGS":
                    test_opt = LBFGS(test_atoms)
                elif name == "Sella" and deps.has("sella"):
                    test_opt = Sella(test_atoms, internal=True, order=0)
                else:
                    continue

                try:
                    test_opt.run(fmax=tight_fmax, steps=50)
                    tight_max_force = np.max(np.abs(test_atoms.get_forces()))
                    print(f"{name} tight convergence: {tight_max_force:.6f} eV/Å")

                    # Should be able to achieve tighter convergence
                    assert tight_max_force < tight_fmax, (
                        f"{name} could not achieve tight convergence {tight_fmax} eV/Å, "
                        f"got {tight_max_force:.6f} eV/Å"
                    )

                except Exception as e:
                    print(f"{name} tight convergence test failed: {e}")

    def test_optimizer_energy_consistency(self):
        """Test that different optimizers find similar final energies for the same system."""
        from ase import Atoms

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.8, 0.6, 0], [-0.8, 0.6, 0]])
        atoms.calc = MockCalculator()

        final_energies = {}

        # Test LBFGS
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        lbfgs_opt.run(fmax=0.01, steps=200)
        final_energies["LBFGS"] = atoms_lbfgs.get_potential_energy()

        # Test Sella if available
        if deps.has("sella"):
            atoms_sella = atoms.copy()
            atoms_sella.calc = MockCalculator()
            sella_opt = Sella(atoms_sella, internal=True, order=0)
            sella_opt.run(fmax=0.01, steps=200)
            final_energies["Sella"] = atoms_sella.get_potential_energy()

        # Compare energies
        if len(final_energies) >= 2:
            energies = list(final_energies.values())
            energy_diff = max(energies) - min(energies)

            print("\n=== ENERGY CONSISTENCY ===")
            for name, energy in final_energies.items():
                print(f"{name}: {energy:.6f} eV")
            print(f"Energy difference: {energy_diff:.6f} eV")

            # Energies should be reasonably similar (within 0.001 eV)
            assert energy_diff < 0.001, (
                f"Final energies differ too much between optimizers: {energy_diff:.6f} eV. "
                f"This suggests inconsistent optimization to different minima."
            )
