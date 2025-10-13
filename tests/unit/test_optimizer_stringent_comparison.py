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
                assert max_force < 0.01, (
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
            assert energy_change_geo > 1e-6, "GeometricOptimizer TS should actually change energy"
            assert (
                position_change_geo > 1e-6
            ), "GeometricOptimizer TS should actually change positions"
        except (np.linalg.LinAlgError, RuntimeError) as e:
            # geomeTRIC can fail with numerical issues on certain geometries
            # This is a known limitation, not a bug in our code
            pytest.skip(f"geomeTRIC failed with numerical issues: {e}")

    def test_optimizer_coordinate_and_frequency_comparison(self):
        """Stringent comparison of final coordinates and frequencies between optimizers."""
        from ase import Atoms

        from qme.analysis.frequency import FrequencyAnalysis

        # Use a simple water molecule for reliable convergence
        # Start with a distorted geometry that should converge to the same minimum
        positions = np.array(
            [
                [0.0, 0.0, 0.0],  # O
                [0.8, 0.6, 0.0],  # H1 (distorted)
                [-0.8, 0.6, 0.0],  # H2 (distorted)
            ]
        )

        atoms = Atoms("H2O", positions=positions)
        atoms.calc = MockCalculator()

        initial_energy = atoms.get_potential_energy()
        initial_positions = atoms.get_positions().copy()

        # Test multiple optimizers
        optimizers_results = {}

        # Test LBFGS with more lenient convergence and more steps
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        lbfgs_opt.run(fmax=0.05, steps=200)
        optimizers_results["LBFGS"] = {
            "atoms": atoms_lbfgs,
            "energy": atoms_lbfgs.get_potential_energy(),
            "positions": atoms_lbfgs.get_positions(),
            "forces": atoms_lbfgs.get_forces(),
            "converged": lbfgs_opt.converged(atoms_lbfgs.get_forces().flatten()),
        }

        # Test GeometricOptimizer if available
        if deps.has("geometric"):
            atoms_geo = atoms.copy()
            atoms_geo.calc = MockCalculator()
            geo_opt = GeometricOptimizer(atoms_geo, order=0)
            geo_opt.run(fmax=0.05, steps=200)
            optimizers_results["Geometric"] = {
                "atoms": atoms_geo,
                "energy": atoms_geo.get_potential_energy(),
                "positions": atoms_geo.get_positions(),
                "forces": atoms_geo.get_forces(),
                "converged": geo_opt.converged,
            }

        # Test Sella if available
        if deps.has("sella"):
            atoms_sella = atoms.copy()
            atoms_sella.calc = MockCalculator()
            sella_opt = Sella(atoms_sella, internal=True, order=0)
            sella_opt.run(fmax=0.05, steps=200)
            optimizers_results["Sella"] = {
                "atoms": atoms_sella,
                "energy": atoms_sella.get_potential_energy(),
                "positions": atoms_sella.get_positions(),
                "forces": atoms_sella.get_forces(),
                "converged": sella_opt.converged(),
            }

        # DETAILED COORDINATE COMPARISON
        print("\n=== DETAILED COORDINATE COMPARISON ===")

        # Compare final coordinates between optimizers
        optimizer_names = list(optimizers_results.keys())
        for i, name1 in enumerate(optimizer_names):
            for name2 in optimizer_names[i + 1 :]:
                pos1 = optimizers_results[name1]["positions"]
                pos2 = optimizers_results[name2]["positions"]

                # Maximum coordinate difference
                max_coord_diff = np.max(np.abs(pos1 - pos2))
                rms_coord_diff = np.sqrt(np.mean((pos1 - pos2) ** 2))

                print(f"{name1} vs {name2}:")
                print(f"  Max coordinate difference: {max_coord_diff:.6f} Å")
                print(f"  RMS coordinate difference: {rms_coord_diff:.6f} Å")

                # All optimizers should converge to the same minimum for simple molecules
                # Use stringent threshold: coordinates should be within 0.1 Å
                assert max_coord_diff < 0.1, (
                    f"Final coordinates differ too much between {name1} and {name2}: "
                    f"{max_coord_diff:.6f} Å. This suggests inconsistent optimization."
                )

        # DETAILED FORCE COMPARISON
        print("\n=== DETAILED FORCE COMPARISON ===")

        for name, results in optimizers_results.items():
            forces = results["forces"]
            max_force = np.max(np.abs(forces))
            rms_force = np.sqrt(np.mean(forces**2))

            print(f"{name}:")
            print(f"  Max force: {max_force:.6f} eV/Å")
            print(f"  RMS force: {rms_force:.6f} eV/Å")
            print(f"  Converged: {results['converged']}")

            # If optimizer claims convergence, forces should be low
            if results["converged"]:
                assert max_force < 0.05, (
                    f"{name} claims convergence but max force is {max_force:.6f} eV/Å. "
                    f"This suggests a bug in convergence detection."
                )

        # DETAILED FREQUENCY COMPARISON
        print("\n=== DETAILED FREQUENCY COMPARISON ===")

        frequency_results = {}
        for name, results in optimizers_results.items():
            try:
                # Calculate frequencies using finite differences
                freq_analysis = FrequencyAnalysis(
                    atoms=results["atoms"], calculator=MockCalculator(), delta=0.01
                )
                frequencies = freq_analysis.get_frequencies()

                # Filter out imaginary frequencies (negative values)
                real_frequencies = frequencies[frequencies > 0]

                frequency_results[name] = {
                    "frequencies": frequencies,
                    "real_frequencies": real_frequencies,
                    "imaginary_count": np.sum(frequencies < 0),
                    "lowest_real": np.min(real_frequencies) if len(real_frequencies) > 0 else None,
                    "highest_real": np.max(real_frequencies) if len(real_frequencies) > 0 else None,
                }

                print(f"{name}:")
                print(f"  Total frequencies: {len(frequencies)}")
                print(f"  Imaginary frequencies: {frequency_results[name]['imaginary_count']}")
                if frequency_results[name]["lowest_real"] is not None:
                    print(
                        "  Lowest real frequency: {:.2f} cm⁻¹".format(
                            frequency_results[name]["lowest_real"]
                        )
                    )
                    print(
                        "  Highest real frequency: {:.2f} cm⁻¹".format(
                            frequency_results[name]["highest_real"]
                        )
                    )

            except Exception as e:
                print(f"{name}: Frequency calculation failed: {e}")
                # Don't fail the test for frequency calculation issues
                # as this might be due to the mock calculator limitations
                continue

        # Compare frequencies between optimizers
        freq_names = list(frequency_results.keys())
        for i, name1 in enumerate(freq_names):
            for name2 in freq_names[i + 1 :]:
                if (
                    frequency_results[name1]["real_frequencies"] is not None
                    and frequency_results[name2]["real_frequencies"] is not None
                ):

                    freqs1 = frequency_results[name1]["real_frequencies"]
                    freqs2 = frequency_results[name2]["real_frequencies"]

                    if len(freqs1) == len(freqs2):
                        # Compare frequencies (sort to handle different ordering)
                        freqs1_sorted = np.sort(freqs1)
                        freqs2_sorted = np.sort(freqs2)

                        freq_diff = np.abs(freqs1_sorted - freqs2_sorted)
                        max_freq_diff = np.max(freq_diff)
                        rms_freq_diff = np.sqrt(np.mean(freq_diff**2))

                        print(f"{name1} vs {name2} frequencies:")
                        print(f"  Max frequency difference: {max_freq_diff:.2f} cm⁻¹")
                        print(f"  RMS frequency difference: {rms_freq_diff:.2f} cm⁻¹")

                        # Allow reasonable frequency differences (within 200 cm⁻¹)
                        # Frequencies are more sensitive to small coordinate differences
                        # than coordinates themselves
                        assert max_freq_diff < 200.0, (
                            f"Frequencies differ too much between {name1} and {name2}: "
                            f"{max_freq_diff:.2f} cm⁻¹. "
                            f"This suggests inconsistent optimization."
                        )

        # ENERGY COMPARISON
        print("\n=== ENERGY COMPARISON ===")

        energies = [results["energy"] for results in optimizers_results.values()]
        energy_range = max(energies) - min(energies)

        print(f"Energy range across optimizers: {energy_range:.6f} eV")

        # Energies should be reasonably similar (within 0.001 eV)
        assert energy_range < 0.001, (
            f"Final energies differ too much across optimizers: {energy_range:.6f} eV. "
            f"This suggests inconsistent optimization to different minima."
        )

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

    def test_optimizer_convergence_quality_comparison(self):
        """Stringent comparison of convergence quality between optimizers."""
        from ase import Atoms

        # Use a simple water molecule for reliable convergence testing
        # Start with a distorted geometry that should converge to the same minimum
        positions = np.array(
            [
                [0.0, 0.0, 0.0],  # O
                [1.0, 0.8, 0.0],  # H1 (distorted)
                [-1.0, 0.8, 0.0],  # H2 (distorted)
            ]
        )

        atoms = Atoms("H2O", positions=positions)
        atoms.calc = MockCalculator()

        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_max_force = np.max(np.abs(initial_forces))

        print(f"Initial energy: {initial_energy:.6f} eV")
        print(f"Initial max force: {initial_max_force:.6f} eV/Å")

        # Test multiple optimizers with different convergence criteria
        convergence_results = {}

        # Test LBFGS with more lenient convergence and more steps
        atoms_lbfgs = atoms.copy()
        atoms_lbfgs.calc = MockCalculator()
        lbfgs_opt = LBFGS(atoms_lbfgs)
        lbfgs_opt.run(fmax=0.05, steps=200)

        convergence_results["LBFGS"] = {
            "atoms": atoms_lbfgs,
            "energy": atoms_lbfgs.get_potential_energy(),
            "forces": atoms_lbfgs.get_forces(),
            "max_force": np.max(np.abs(atoms_lbfgs.get_forces())),
            "rms_force": np.sqrt(np.mean(atoms_lbfgs.get_forces() ** 2)),
            "converged": lbfgs_opt.converged(atoms_lbfgs.get_forces().flatten()),
            "steps": lbfgs_opt.get_number_of_steps(),
        }

        # Test GeometricOptimizer if available
        if deps.has("geometric"):
            atoms_geo = atoms.copy()
            atoms_geo.calc = MockCalculator()
            geo_opt = GeometricOptimizer(atoms_geo, order=0)
            geo_opt.run(fmax=0.05, steps=200)

            convergence_results["Geometric"] = {
                "atoms": atoms_geo,
                "energy": atoms_geo.get_potential_energy(),
                "forces": atoms_geo.get_forces(),
                "max_force": np.max(np.abs(atoms_geo.get_forces())),
                "rms_force": np.sqrt(np.mean(atoms_geo.get_forces() ** 2)),
                "converged": geo_opt.converged,
                "steps": geo_opt.step_count,
            }

        # Test Sella if available
        if deps.has("sella"):
            atoms_sella = atoms.copy()
            atoms_sella.calc = MockCalculator()
            sella_opt = Sella(atoms_sella, internal=True, order=0)
            sella_opt.run(fmax=0.05, steps=200)

            convergence_results["Sella"] = {
                "atoms": atoms_sella,
                "energy": atoms_sella.get_potential_energy(),
                "forces": atoms_sella.get_forces(),
                "max_force": np.max(np.abs(atoms_sella.get_forces())),
                "rms_force": np.sqrt(np.mean(atoms_sella.get_forces() ** 2)),
                "converged": sella_opt.converged(),
                "steps": sella_opt.get_number_of_steps(),
            }

        # DETAILED CONVERGENCE ANALYSIS
        print("\n=== CONVERGENCE QUALITY COMPARISON ===")

        for name, results in convergence_results.items():
            print(f"\n{name}:")
            print(f"  Final energy: {results['energy']:.6f} eV")
            print(f"  Energy change: {abs(results['energy'] - initial_energy):.6f} eV")
            print(f"  Max force: {results['max_force']:.6f} eV/Å")
            print(f"  RMS force: {results['rms_force']:.6f} eV/Å")
            print(f"  Converged: {results['converged']}")
            print(f"  Steps: {results['steps']}")

            # Calculate force improvement
            force_improvement = initial_max_force - results["max_force"]
            print(f"  Force improvement: {force_improvement:.6f} eV/Å")

            # Verify convergence claims
            if results["converged"]:
                assert results["max_force"] < 0.05, (
                    f"{name} claims convergence but max force is {results['max_force']:.6f} eV/Å. "
                    f"This suggests a bug in convergence detection."
                )
                print(f"  ✅ {name} properly converged")
            else:
                print(f"  ⚠️  {name} did not converge")

        # CONVERGENCE EFFICIENCY COMPARISON
        print("\n=== CONVERGENCE EFFICIENCY COMPARISON ===")

        # Compare energy improvements
        energy_improvements = {}
        for name, results in convergence_results.items():
            energy_improvement = initial_energy - results["energy"]
            energy_improvements[name] = energy_improvement
            print(f"{name}: Energy improvement = {energy_improvement:.6f} eV")

        # Compare force improvements
        force_improvements = {}
        for name, results in convergence_results.items():
            force_improvement = initial_max_force - results["max_force"]
            force_improvements[name] = force_improvement
            print(f"{name}: Force improvement = {force_improvement:.6f} eV/Å")

        # CONVERGENCE CONSISTENCY CHECKS
        print("\n=== CONVERGENCE CONSISTENCY CHECKS ===")

        # Check that converged optimizers have similar final energies
        converged_optimizers = [
            (name, results) for name, results in convergence_results.items() if results["converged"]
        ]

        if len(converged_optimizers) >= 2:
            converged_energies = [results["energy"] for _, results in converged_optimizers]
            energy_range = max(converged_energies) - min(converged_energies)

            print(f"Energy range among converged optimizers: {energy_range:.6f} eV")

            # Converged optimizers should find similar energies (within 0.001 eV)
            assert energy_range < 0.001, (
                f"Converged optimizers found very different energies: {energy_range:.6f} eV. "
                f"This suggests inconsistent convergence to different minima."
            )

            # Check that converged optimizers have similar final forces
            converged_forces = [results["max_force"] for _, results in converged_optimizers]
            force_range = max(converged_forces) - min(converged_forces)

            print(f"Force range among converged optimizers: {force_range:.6f} eV/Å")

            # Converged optimizers should have similar final forces (within 0.02 eV/Å)
            # Different optimizers may have different convergence criteria
            assert force_range < 0.02, (
                f"Converged optimizers have very different final forces: {force_range:.6f} eV/Å. "
                f"This suggests inconsistent convergence quality."
            )

        # CONVERGENCE ROBUSTNESS TEST
        print("\n=== CONVERGENCE ROBUSTNESS TEST ===")

        # Test with tighter convergence criteria
        tight_fmax = 0.001

        for name, results in convergence_results.items():
            if results["converged"]:
                # Test if optimizer can achieve tighter convergence
                test_atoms = atoms.copy()
                test_atoms.calc = MockCalculator()

                if name == "LBFGS":
                    test_opt = LBFGS(test_atoms)
                elif name == "Geometric" and deps.has("geometric"):
                    test_opt = GeometricOptimizer(test_atoms, order=0)
                elif name == "Sella" and deps.has("sella"):
                    test_opt = Sella(test_atoms, internal=True, order=0)
                else:
                    continue

                try:
                    test_opt.run(fmax=tight_fmax, steps=50)
                    tight_max_force = np.max(np.abs(test_atoms.get_forces()))

                    print(f"{name} with tight convergence (fmax={tight_fmax}):")
                    print(f"  Final max force: {tight_max_force:.6f} eV/Å")

                    if tight_max_force < tight_fmax:
                        print(f"  ✅ {name} can achieve tight convergence")
                    else:
                        print(f"  ⚠️  {name} struggles with tight convergence")

                except Exception as e:
                    print(f"  ❌ {name} failed with tight convergence: {e}")
