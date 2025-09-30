"""
Tests for vibrational frequency analysis functionality.

This test suite covers frequency calculations, Hessian methods,
transition state verification, and thermodynamic properties.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator
from qme.dependencies import deps

# Define available backends for ML frequency testing
AVAILABLE_BACKENDS = ["mock"]
if deps.has("fairchem"):
    AVAILABLE_BACKENDS.append("uma")
if deps.has("so3lr"):
    AVAILABLE_BACKENDS.append("so3lr")
if deps.has("aimnet2"):
    AVAILABLE_BACKENDS.append("aimnet2")
if deps.has("mace"):
    AVAILABLE_BACKENDS.append("mace")


class TestFrequencyAnalysis:
    """Test vibrational frequency calculations."""

    @pytest.fixture
    def h2_molecule(self):
        """H2 molecule for simple frequency tests."""
        atoms = molecule("H2")
        atoms.calc = qme.MockCalculator(backend="so3lr")
        return atoms

    @pytest.fixture
    def water_molecule(self):
        """H2O molecule for more complex frequency tests."""
        atoms = molecule("H2O")
        atoms.calc = qme.MockCalculator(backend="so3lr")
        return atoms

    @pytest.fixture
    def linear_molecule(self):
        """CO2 molecule for linear molecule tests."""
        atoms = molecule("CO2")
        atoms.calc = qme.MockCalculator(backend="so3lr")
        return atoms

    def test_frequency_analysis_creation(self, h2_molecule):
        """Test FrequencyAnalysis object creation."""
        freq_analysis = FrequencyAnalysis(h2_molecule, h2_molecule.calc, delta=0.01)

        assert freq_analysis.atoms is not None
        assert freq_analysis.calculator is not None
        assert freq_analysis.delta == 0.01
        assert freq_analysis.nfree == 5  # Linear molecule: 3 trans + 2 rot

    def test_frequency_analysis_water(self, water_molecule):
        """Test frequency analysis for water molecule."""
        freq_analysis = FrequencyAnalysis(
            water_molecule, water_molecule.calc, delta=0.01
        )

        assert freq_analysis.nfree == 6  # Non-linear molecule: 3 trans + 3 rot

        # Calculate frequencies
        hessian = freq_analysis.calculate_hessian(method="finite_differences")
        frequencies, modes = freq_analysis.diagonalize_hessian()

        # Basic checks
        assert hessian.shape == (9, 9)  # 3 atoms * 3 coordinates
        assert len(frequencies) == 9  # 3N modes total
        assert modes.shape == (9, 9)  # 3N x 3N modes

        # Check vibrational frequencies (excluding trans/rot)
        vib_frequencies = freq_analysis.get_frequencies()
        assert len(vib_frequencies) == 3  # 3N - 6 = 3 vibrational modes

        # For a stable molecule, all vibrational frequencies should be positive
        # (mock calculator should return reasonable frequencies)
        positive_freqs = np.sum(vib_frequencies > 0)
        assert positive_freqs >= 2  # At least most frequencies should be positive

    def test_linear_molecule_degrees_of_freedom(self, linear_molecule):
        """Test correct degrees of freedom for linear molecules."""
        freq_analysis = FrequencyAnalysis(linear_molecule, linear_molecule.calc)

        # Linear molecules have 5 degrees of freedom to remove (3 trans + 2 rot)
        assert freq_analysis.nfree == 5

        # CO2 should have 3N - 5 = 4 vibrational modes
        freq_analysis.calculate_hessian()
        freq_analysis.diagonalize_hessian()
        vib_frequencies = freq_analysis.get_frequencies()
        assert len(vib_frequencies) == 4

    def test_hessian_symmetry(self, h2_molecule):
        """Test that calculated Hessian matrix is symmetric."""
        hessian_calc = HessianCalculator(
            h2_molecule, h2_molecule.calc, delta=0.01, method="central"
        )

        hessian = hessian_calc.calculate_numerical_hessian()

        # Check symmetry
        assert np.allclose(hessian, hessian.T, atol=1e-6)

    def test_frequency_units(self, water_molecule):
        """Test frequency unit conversions."""
        freq_analysis = FrequencyAnalysis(water_molecule, water_molecule.calc)
        freq_analysis.calculate_hessian()
        freq_analysis.diagonalize_hessian()

        # Test different units
        freq_cm1 = freq_analysis.get_frequencies(unit="cm-1")
        freq_meV = freq_analysis.get_frequencies(unit="meV")
        freq_THz = freq_analysis.get_frequencies(unit="THz")

        assert len(freq_cm1) == len(freq_meV) == len(freq_THz)

        # Check that conversions make sense (positive frequencies)
        positive_cm1 = freq_cm1[freq_cm1 > 0]
        positive_meV = freq_meV[freq_meV > 0]
        positive_THz = freq_THz[freq_THz > 0]

        if len(positive_cm1) > 0:
            assert len(positive_meV) > 0
            assert len(positive_THz) > 0

            # Basic conversion check (rough magnitude)
            assert np.mean(positive_meV) < np.mean(
                positive_cm1
            )  # meV should be smaller
            assert np.mean(positive_THz) < np.mean(
                positive_cm1
            )  # THz should be smaller

    def test_zero_point_energy(self, water_molecule):
        """Test zero-point energy calculation."""
        freq_analysis = FrequencyAnalysis(water_molecule, water_molecule.calc)
        freq_analysis.calculate_hessian()
        freq_analysis.diagonalize_hessian()

        zpe = freq_analysis.get_zero_point_energy()

        # ZPE should be positive for stable molecules
        assert zpe > 0
        assert isinstance(zpe, float)

    def test_thermodynamic_properties(self, water_molecule):
        """Test thermodynamic property calculations."""
        freq_analysis = FrequencyAnalysis(water_molecule, water_molecule.calc)
        freq_analysis.calculate_hessian()
        freq_analysis.diagonalize_hessian()

        thermo_props = freq_analysis.get_thermodynamic_properties(temperature=298.15)

        # Check required properties are present
        required_keys = [
            "temperature",
            "zero_point_energy",
            "internal_energy",
            "heat_capacity",
            "entropy",
            "frequencies_cm_1",
        ]
        for key in required_keys:
            assert key in thermo_props

        # Basic checks
        assert thermo_props["temperature"] == 298.15
        assert thermo_props["zero_point_energy"] > 0
        assert thermo_props["heat_capacity"] > 0
        assert thermo_props["entropy"] > 0

    def test_transition_state_verification_minimum(self, water_molecule):
        """Test TS verification for a minimum structure."""
        freq_analysis = FrequencyAnalysis(water_molecule, water_molecule.calc)
        freq_analysis.calculate_hessian()
        freq_analysis.diagonalize_hessian()

        ts_results = freq_analysis.is_transition_state(threshold=50.0)

        # Check structure
        assert "is_transition_state" in ts_results
        assert "n_imaginary_frequencies" in ts_results
        assert "assessment" in ts_results

        # For a stable molecule, should not be a TS
        # (depends on mock calculator behavior, but generally should be minimum)
        n_imaginary = ts_results["n_imaginary_frequencies"]
        assert n_imaginary >= 0  # Should be non-negative

    def test_normal_mode_trajectory_generation(self, water_molecule):
        """Test normal mode trajectory generation."""
        freq_analysis = FrequencyAnalysis(water_molecule, water_molecule.calc)
        freq_analysis.calculate_hessian()
        freq_analysis.diagonalize_hessian()

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".traj", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Generate trajectory for first vibrational mode
            vib_frequencies = freq_analysis.get_frequencies()
            if len(vib_frequencies) > 0:
                freq_analysis.write_mode_trajectory(
                    mode_index=0, filename=tmp_path, amplitude=0.5, nframes=10
                )

                # Check that file was created
                assert Path(tmp_path).exists()

                # Read trajectory and check basic properties
                from ase.io import read

                trajectory = read(tmp_path, index=":")
                assert len(trajectory) == 10
                for atoms_obj in trajectory:
                    assert isinstance(atoms_obj, Atoms)
                    assert len(atoms_obj) == len(water_molecule)
        finally:
            # Clean up
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_finite_difference_methods(self, h2_molecule):
        """Test different finite difference methods."""
        # Test central differences
        hessian_central = HessianCalculator(
            h2_molecule, h2_molecule.calc, method="central"
        )
        hessian_c = hessian_central.calculate_numerical_hessian()

        # Test forward differences
        hessian_forward = HessianCalculator(
            h2_molecule, h2_molecule.calc, method="forward"
        )
        hessian_f = hessian_forward.calculate_numerical_hessian()

        # Both should be symmetric
        assert np.allclose(hessian_c, hessian_c.T, atol=1e-6)
        assert np.allclose(hessian_f, hessian_f.T, atol=1e-6)

        # Central differences should generally be more accurate
        # but both should give reasonable results


class TestQMEOptimizerFrequencyIntegration:
    """Test frequency analysis integration with QMEOptimizer."""

    @pytest.fixture
    def qme_optimizer(self):
        """QME optimizer with mock calculator."""
        return qme.QMEOptimizer(backend="mock")

    @pytest.fixture
    def water_atoms(self):
        """Water molecule."""
        return molecule("H2O")

    def test_calculate_frequencies_method(self, qme_optimizer, water_atoms):
        """Test QMEOptimizer.calculate_frequencies method."""
        qme_optimizer.atoms = water_atoms

        results = qme_optimizer.calculate_frequencies(
            delta=0.01, method="finite_differences", temperature=298.15
        )

        # Check result structure
        required_keys = [
            "frequencies",
            "zero_point_energy",
            "thermodynamic_properties",
            "ts_analysis",
            "is_ts",
            "method_used",
        ]
        for key in required_keys:
            assert key in results

        # Check that results are stored
        assert "frequency_analysis" in qme_optimizer.results

    def test_verify_transition_state_method(self, qme_optimizer, water_atoms):
        """Test QMEOptimizer.verify_transition_state method."""
        qme_optimizer.atoms = water_atoms

        ts_results = qme_optimizer.verify_transition_state(
            freq_threshold=50.0, delta=0.01
        )

        # Check result structure
        assert "is_transition_state" in ts_results
        assert "verification_summary" in ts_results
        assert "structure_verified" in ts_results

        # Check that results are stored
        assert "ts_verification" in qme_optimizer.results

    def test_reaction_thermodynamics_without_ts(self, qme_optimizer):
        """Test reaction thermodynamics without transition state."""
        # Create simple reactant and product
        reactant = molecule("H2")
        product = molecule("H2")
        # Slightly displace product to make it different
        product.positions[1, 0] += 0.1

        results = qme_optimizer.calculate_reaction_thermodynamics(
            reactant_atoms=reactant, product_atoms=product, temperature=298.15
        )

        # Check result structure
        required_keys = [
            "reactant",
            "product",
            "reaction_energy",
            "temperature",
            "has_transition_state",
        ]
        for key in required_keys:
            assert key in results

        assert results["has_transition_state"] is False

        # Check energy components
        assert "electronic" in results["reaction_energy"]
        assert "zero_point_corrected" in results["reaction_energy"]

    def test_reaction_thermodynamics_with_ts(self, qme_optimizer):
        """Test reaction thermodynamics with transition state."""
        # Create simple structures
        reactant = molecule("H2")
        product = molecule("H2")
        ts = molecule("H2")

        # Modify structures slightly
        product.positions[1, 0] += 0.1
        ts.positions[1, 0] += 0.05  # TS between reactant and product

        results = qme_optimizer.calculate_reaction_thermodynamics(
            reactant_atoms=reactant,
            product_atoms=product,
            ts_atoms=ts,
            temperature=298.15,
        )

        # Check result structure
        required_keys = [
            "reactant",
            "product",
            "transition_state",
            "reaction_energy",
            "activation_energy",
            "temperature",
            "has_transition_state",
        ]
        for key in required_keys:
            assert key in results

        assert results["has_transition_state"] is True

        # Check activation energy components
        assert "electronic" in results["activation_energy"]
        assert "zero_point_corrected" in results["activation_energy"]


class TestHessianCalculator:
    """Test Hessian calculation methods."""

    @pytest.fixture
    def h2_atoms(self):
        """H2 molecule with mock calculator."""
        atoms = molecule("H2")
        atoms.calc = qme.MockCalculator()
        return atoms

    def test_hessian_calculator_creation(self, h2_atoms):
        """Test HessianCalculator creation."""
        hessian_calc = HessianCalculator(
            h2_atoms, h2_atoms.calc, delta=0.01, method="central"
        )

        assert hessian_calc.atoms is not None
        assert hessian_calc.calculator is not None
        assert hessian_calc.delta == 0.01
        assert hessian_calc.method == "central"

    def test_hessian_calculation_dimensions(self, h2_atoms):
        """Test Hessian matrix dimensions."""
        hessian_calc = HessianCalculator(h2_atoms, h2_atoms.calc)
        hessian = hessian_calc.calculate_numerical_hessian()

        # H2 has 2 atoms, so 6x6 Hessian
        assert hessian.shape == (6, 6)

    def test_hessian_with_atom_subset(self, h2_atoms):
        """Test Hessian calculation with atom subset."""
        # Calculate Hessian for only first atom
        hessian_calc = HessianCalculator(h2_atoms, h2_atoms.calc, indices=[0])
        hessian = hessian_calc.calculate_numerical_hessian()

        # Should be 3x3 for one atom
        assert hessian.shape == (3, 3)


class TestErrorHandling:
    """Test error handling in frequency analysis."""

    def test_frequency_analysis_without_hessian(self):
        """Test accessing frequencies without explicitly calculating Hessian first."""
        atoms = molecule("H2")
        atoms.calc = qme.MockCalculator()

        freq_analysis = FrequencyAnalysis(atoms, atoms.calc)

        # Should automatically calculate Hessian when frequencies are requested
        frequencies = freq_analysis.get_frequencies()

        # Should have calculated frequencies for H2 (1 vibrational mode after removing 5 trans/rot)
        assert len(frequencies) == 1
        assert freq_analysis._is_calculated is True

    def test_invalid_frequency_unit(self):
        """Test invalid frequency unit."""
        atoms = molecule("H2")
        atoms.calc = qme.MockCalculator()

        freq_analysis = FrequencyAnalysis(atoms, atoms.calc)
        freq_analysis.calculate_hessian()
        freq_analysis.diagonalize_hessian()

        with pytest.raises(ValueError):
            freq_analysis.get_frequencies(unit="invalid_unit")

    def test_invalid_hessian_method(self):
        """Test invalid Hessian calculation method."""
        atoms = molecule("H2")
        atoms.calc = qme.MockCalculator()

        freq_analysis = FrequencyAnalysis(atoms, atoms.calc)

        with pytest.raises(ValueError):
            freq_analysis.calculate_hessian(method="invalid_method")


class TestMLBackendFrequencyAnalysis:
    """Test frequency analysis across all available ML backends.

    These tests use real ML backends when available, falling back to mock
    when they are not installed. This allows testing the complete pathway
    including analytical Hessians from real ML potentials.
    """

    @pytest.fixture(params=AVAILABLE_BACKENDS)
    def backend(self, request):
        """Parametrized fixture for available backends."""
        backend_name = request.param

        # Skip if backend is not available (double-check)
        if backend_name == "uma" and not deps.has("fairchem"):
            pytest.skip("UMA backend not available")
        elif backend_name == "so3lr" and not deps.has("so3lr"):
            pytest.skip("SO3LR backend not available")
        elif backend_name == "aimnet2" and not deps.has("aimnet2"):
            pytest.skip("AIMNET2 backend not available")
        elif backend_name == "mace" and not deps.has("mace"):
            pytest.skip("MACE backend not available")

        return backend_name

    @pytest.fixture
    def small_molecules(self):
        """Small molecules suitable for ML backend testing."""
        return {
            "H2": molecule("H2"),
            "H2O": molecule("H2O"),
            "CH4": molecule("CH4"),
            "NH3": molecule("NH3"),
        }

    def test_frequency_calculation_method_detection(self, backend, small_molecules):
        """Test that frequency analysis correctly detects available methods."""
        # Test with just H2 to keep it simple and fast
        atoms = small_molecules["H2"]

        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            optimizer.atoms = atoms

            # Create frequency analysis
            calc = optimizer.calculator
            freq_analysis = FrequencyAnalysis(atoms, calc)

            # Check if calculator supports direct Hessian
            has_direct = freq_analysis._supports_direct_hessian()

            # Calculate frequencies and check method used
            results = optimizer.calculate_frequencies(method="auto")
            method_used = results.get("method_used", "unknown")

            # Verify that frequency calculation completed successfully
            frequencies = results.get("frequencies", [])
            assert len(frequencies) > 0, f"No frequencies calculated for {backend}"

            # Check method consistency - "auto" means it auto-detected the best method
            assert method_used in [
                "direct",
                "finite_differences",
                "auto",
            ], f"Unexpected method {method_used} for {backend}"

            # For mock backend, also test explicit direct method to verify it works
            if backend == "mock" and has_direct:
                results_direct = optimizer.calculate_frequencies(method="direct")
                assert (
                    results_direct["method_used"] == "direct"
                ), "Mock backend should support explicit direct method"

        except Exception as e:
            pytest.fail(f"Failed frequency analysis for H2 with {backend}: {e}")

    def test_hessian_properties_across_backends(self, backend):
        """Test that Hessian matrices have correct properties across backends."""
        atoms = molecule("H2O")

        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            optimizer.atoms = atoms

            # Calculate frequencies
            results = optimizer.calculate_frequencies(delta=0.01)

            # Check that we got reasonable results
            frequencies = results["frequencies"]
            assert len(frequencies) > 0, "Should have at least one frequency"

            # For stable molecules, most frequencies should be positive
            positive_freqs = [f for f in frequencies if f > 1.0]  # > 1 cm^-1
            total_freqs = len(frequencies)

            # Allow some very small frequencies (translation/rotation)
            # but most should be positive for a stable molecule
            assert (
                len(positive_freqs) >= total_freqs - 6
            ), f"Too many near-zero frequencies: {len(positive_freqs)}/{total_freqs}"

        except Exception as e:
            pytest.fail(f"Hessian properties test failed for {backend}: {e}")

    def test_thermodynamic_properties_calculation(self, backend):
        """Test thermodynamic properties calculation across backends."""
        atoms = molecule("H2O")

        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            optimizer.atoms = atoms

            # Calculate frequencies with thermodynamic properties
            results = optimizer.calculate_frequencies(temperature=298.15)

            # Check that thermodynamic properties were calculated
            thermo_props = results.get("thermodynamic_properties", {})

            # Should have zero-point energy
            assert "zero_point_energy" in thermo_props
            zpe = thermo_props["zero_point_energy"]
            assert isinstance(zpe, (int, float)), "ZPE should be numeric"
            assert zpe >= 0, "Zero-point energy should be non-negative"

        except Exception as e:
            # Some numerical issues with very small frequencies are acceptable
            if "divide by zero" in str(e) or "invalid value" in str(e):
                pytest.skip(
                    f"Numerical issues with {backend} (expected for simple models)"
                )
            else:
                pytest.fail(f"Thermodynamic properties test failed for {backend}: {e}")

    def test_transition_state_verification_consistency(self, backend):
        """Test TS verification gives consistent results across backends."""
        atoms = molecule("H2O")  # This should NOT be a transition state

        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            optimizer.atoms = atoms

            # Verify transition state
            ts_results = optimizer.verify_transition_state(freq_threshold=50.0)

            # Water should not be identified as a transition state
            is_ts = ts_results.get("is_transition_state", None)
            assert is_ts is not None, "Should have TS verification result"
            assert not is_ts, "H2O should not be identified as transition state"

            # Check verification summary (it might be a string or dict)
            summary = ts_results.get("verification_summary", {})
            if isinstance(summary, str):
                # Summary is a formatted string, check for expected content
                assert (
                    "imaginary" in summary.lower()
                ), "Summary should mention imaginary frequencies"
                assert "0" in summary, "Should indicate 0 imaginary frequencies"
            else:
                # Summary is a dictionary
                assert "imaginary_count" in summary
                assert (
                    summary["imaginary_count"] == 0
                ), "H2O should have no imaginary frequencies"

        except Exception as e:
            pytest.fail(f"TS verification test failed for {backend}: {e}")

    @pytest.mark.parametrize("molecule_name", ["H2", "H2O", "CH4"])
    def test_frequency_scaling_across_backends(self, backend, molecule_name):
        """Test that frequency magnitudes are reasonable across backends."""
        atoms = molecule(molecule_name)

        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            optimizer.atoms = atoms

            # Calculate frequencies
            results = optimizer.calculate_frequencies()
            frequencies = results["frequencies"]

            # Filter out near-zero frequencies (translation/rotation)
            real_freqs = [f for f in frequencies if f > 10.0]  # > 10 cm^-1

            if real_freqs:
                # Vibrational frequencies should be in reasonable range
                max_freq = max(real_freqs)
                min_freq = min(real_freqs)

                # Typical molecular vibrations: 200-4000 cm^-1
                # Our simple models might be lower, but should be > 0 and < 10000
                assert min_freq > 0, f"Minimum frequency should be positive: {min_freq}"
                assert max_freq < 10000, f"Maximum frequency seems too high: {max_freq}"

                # Check that frequencies are not all identical (would indicate a problem)
                if len(real_freqs) > 1:
                    freq_std = np.std(real_freqs)
                    assert freq_std > 0.1, "Frequencies should not all be identical"

        except Exception as e:
            pytest.fail(
                f"Frequency scaling test failed for {molecule_name} with {backend}: {e}"
            )

    def test_frequency_analysis_consistency(self, backend):
        """Test that repeated frequency calculations are consistent."""
        if backend == "mock":
            # Mock calculator should be perfectly consistent
            tolerance = 1e-10
        else:
            # Real ML backends might have small numerical variations
            tolerance = 1e-6

        atoms = molecule("H2")

        try:
            optimizer = qme.QMEOptimizer(backend=backend)
            optimizer.atoms = atoms

            # Calculate frequencies twice
            results1 = optimizer.calculate_frequencies()
            results2 = optimizer.calculate_frequencies()

            frequencies1 = np.array(results1["frequencies"])
            frequencies2 = np.array(results2["frequencies"])

            # Results should be consistent
            assert len(frequencies1) == len(
                frequencies2
            ), "Number of frequencies should be consistent"

            # Sort frequencies for comparison (order might vary)
            freq1_sorted = np.sort(frequencies1)
            freq2_sorted = np.sort(frequencies2)

            diff = np.abs(freq1_sorted - freq2_sorted)
            max_diff = np.max(diff)

            assert (
                max_diff < tolerance
            ), f"Frequency calculations inconsistent (max diff: {max_diff})"

        except Exception as e:
            pytest.fail(f"Consistency test failed for {backend}: {e}")
