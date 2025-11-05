from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator
from qme.backends.availability import is_backend_available
from tests.test_utils import HarmonicCalculator, TestMoleculeFactory

# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def water_molecule():
    return TestMoleculeFactory.get_water_distorted()


@pytest.fixture
def methane_molecule():
    return Atoms(
        symbols="CHHHH",
        positions=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ],
    )


@pytest.fixture
def harmonic_atoms():
    return Atoms(
        symbols="HH",
        positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
    )


# ============================================================================
# Backend-Specific Consistency Tests
# ============================================================================


class TestMACEHessianConsistency:
    @pytest.mark.skipif(not is_backend_available("mace"), reason="MACE backend not available")
    def test_mace_hessian_consistency(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_mace_calculator(model_name="mace-omol-0")
        atoms.calc.ensure_loaded()

        assert "hessian" in atoms.calc.implemented_properties

        hessian_analytical = atoms.calc.get_hessian(atoms)
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=0.01, atol=0.5)

    @pytest.mark.skipif(not is_backend_available("mace"), reason="MACE backend not available")
    @pytest.mark.parametrize("method", ["central", "5point"], ids=["central", "5point"])
    def test_mace_fd_methods(self, water_molecule, method):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_mace_calculator(model_name="mace-omol-0")
        atoms.calc.ensure_loaded()

        hessian_analytical = atoms.calc.get_hessian(atoms)
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, method=method, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        rtol = 0.02 if method == "central" else 0.01
        atol = 1.0 if method == "central" else 0.5

        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=rtol, atol=atol)

    @pytest.mark.skipif(not is_backend_available("mace"), reason="MACE backend not available")
    def test_mace_hessian_frequency_consistency(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_mace_calculator(model_name="mace-omol-0")
        atoms.calc.ensure_loaded()

        if not hasattr(atoms.calc, "get_hessian"):
            pytest.skip("MACE does not support analytical Hessian")

        freq_analytical = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        freq_analytical.calculate_hessian(method="direct")
        freq_analytical.diagonalize_hessian()
        freqs_analytical = freq_analytical._frequencies
        modes_analytical = freq_analytical._normal_modes

        freq_fd = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        freq_fd.calculate_hessian(method="finite_differences")
        freq_fd.diagonalize_hessian()
        freqs_fd, modes_fd = freq_fd._frequencies, freq_fd._normal_modes

        freqs_analytical_vib = freqs_analytical[6:]
        freqs_fd_vib = freqs_fd[6:]
        modes_analytical_vib = modes_analytical[:, 6:]
        modes_fd_vib = modes_fd[:, 6:]

        np.testing.assert_allclose(
            freqs_analytical_vib,
            freqs_fd_vib,
            rtol=0.05,
            atol=50.0,
            err_msg="MACE frequencies mismatch between analytical and FD",
        )

        for i in range(len(freqs_analytical_vib)):
            freq_diff = np.abs(freqs_fd_vib - freqs_analytical_vib[i])
            closest_idx = np.argmin(freq_diff)
            assert freq_diff[closest_idx] < 50.0, (
                f"Mode {i}: frequency mismatch {freq_diff[closest_idx]:.2f} cm^-1"
            )
            mode_overlap = np.abs(np.dot(modes_analytical_vib[:, i], modes_fd_vib[:, closest_idx]))
            assert mode_overlap > 0.8, f"Mode {i}: poor overlap {mode_overlap:.3f}"

        assert np.all(freqs_analytical_vib > 0), "MACE analytical has negative frequencies"
        assert np.all(freqs_fd_vib > 0), "MACE FD has negative frequencies"

    @pytest.mark.skipif(not is_backend_available("mace"), reason="MACE backend not available")
    def test_mace_methane_hessian_consistency(self, methane_molecule):
        atoms = methane_molecule.copy()
        atoms.calc = qme.get_mace_calculator(model_name="mace-omol-0")
        atoms.calc.ensure_loaded()

        if not hasattr(atoms.calc, "get_hessian"):
            pytest.skip("MACE does not support analytical Hessian")

        hessian_analytical = atoms.calc.get_hessian(atoms)
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, method="central", verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=0.01, atol=0.5)


class TestUMAHessianConsistency:
    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_hessian_consistency(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        assert "hessian" in atoms.calc.implemented_properties

        hessian_analytical = atoms.calc.get_hessian(atoms)
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=0.01, atol=0.5)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_hessian_methane(self, methane_molecule):
        atoms = methane_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        hessian_analytical = atoms.calc.get_hessian(atoms)
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=0.01, atol=0.5)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    @pytest.mark.parametrize(
        "method", ["forward", "central", "5point"], ids=["forward", "central", "5point"]
    )
    def test_uma_all_fd_methods(self, water_molecule, method):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        hessian_analytical = atoms.calc.get_hessian(atoms)
        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, method=method, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        if method == "forward":
            rtol, atol = 0.05, 2.0
        elif method == "central":
            rtol, atol = 0.02, 1.0
        else:  # 5point
            rtol, atol = 0.01, 1.0

        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=rtol, atol=atol)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_frequency_comparison(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        freq_analytical = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        freq_analytical.calculate_hessian(method="direct")
        freq_analytical.diagonalize_hessian()
        freqs_analytical = freq_analytical.get_frequencies()

        freq_fd = FrequencyAnalysis(atoms, atoms.calc, delta=0.001, verbose=0)
        freq_fd.calculate_hessian(method="finite_differences")
        freq_fd.diagonalize_hessian()
        freqs_fd = freq_fd.get_frequencies()

        freqs_analytical_vib = freqs_analytical[6:]
        freqs_fd_vib = freqs_fd[6:]

        np.testing.assert_allclose(
            freqs_analytical_vib,
            freqs_fd_vib,
            rtol=0.05,
            atol=50.0,
        )

        assert np.all(freqs_analytical_vib > 0)
        assert np.all(freqs_fd_vib > 0)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_5point_uma_consistency(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        assert "hessian" in atoms.calc.implemented_properties

        hessian_analytical = atoms.calc.get_hessian(atoms)
        hc_5point = HessianCalculator(atoms, atoms.calc, delta=0.01, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        np.testing.assert_allclose(hessian_5point, hessian_analytical, rtol=0.01, atol=1.0)


class TestUMAHessianMethods:
    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    @pytest.mark.parametrize("method", ["vmap", "double_backward"], ids=["vmap", "double_backward"])
    @pytest.mark.parametrize("symmetrize", [True, False], ids=["sym", "no_sym"])
    def test_uma_hessian_method_consistency(self, water_molecule, method, symmetrize):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        hessian1 = atoms.calc.get_hessian(atoms, method=method, symmetrize=symmetrize)
        hessian2 = atoms.calc.get_hessian(atoms, method=method, symmetrize=symmetrize)

        np.testing.assert_allclose(
            hessian1,
            hessian2,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"UMA {method} symmetrize={symmetrize} not reproducible",
        )

        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, method="central", verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        np.testing.assert_allclose(
            hessian1,
            hessian_fd,
            rtol=0.01,
            atol=0.5,
            err_msg=f"UMA {method} symmetrize={symmetrize} does not match FD",
        )

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_methods_consistency(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        hessian_vmap = atoms.calc.get_hessian(atoms, method="vmap", symmetrize=True)
        hessian_db = atoms.calc.get_hessian(atoms, method="double_backward", symmetrize=True)

        np.testing.assert_allclose(
            hessian_vmap,
            hessian_db,
            rtol=0.02,
            atol=1.0,
            err_msg="UMA vmap and double_backward methods give different results",
        )

        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, method="central", verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        np.testing.assert_allclose(
            hessian_vmap, hessian_fd, rtol=0.01, atol=0.5, err_msg="UMA vmap does not match FD"
        )
        np.testing.assert_allclose(
            hessian_db,
            hessian_fd,
            rtol=0.01,
            atol=0.5,
            err_msg="UMA double_backward does not match FD",
        )

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_uma_symmetrization_effect(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        hessian_sym = atoms.calc.get_hessian(atoms, method="double_backward", symmetrize=True)
        hessian_no_sym = atoms.calc.get_hessian(atoms, method="double_backward", symmetrize=False)

        sym_error_sym = np.max(np.abs(hessian_sym - hessian_sym.T))
        sym_error_no_sym = np.max(np.abs(hessian_no_sym - hessian_no_sym.T))

        assert sym_error_sym < 1e-10, "Symmetrized Hessian should be symmetric"
        assert sym_error_no_sym < 0.01, "Non-symmetrized Hessian should be nearly symmetric"

        hc = HessianCalculator(atoms, atoms.calc, delta=0.01, method="central", verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        error_sym = np.max(np.abs(hessian_sym - hessian_fd))
        error_no_sym = np.max(np.abs(hessian_no_sym - hessian_fd))

        assert error_sym <= error_no_sym * 1.2, (
            f"Symmetrization should improve or maintain accuracy: "
            f"error_sym={error_sym:.6f}, error_no_sym={error_no_sym:.6f}"
        )


# ============================================================================
# Finite Difference Method Tests
# ============================================================================


class TestFiniteDifferenceMethods:
    def test_5point_harmonic_accuracy(self, harmonic_atoms):
        calc = HarmonicCalculator(k=1.0)

        hc_5point = HessianCalculator(harmonic_atoms, calc, delta=0.01, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        hessian_analytical = calc.get_hessian(harmonic_atoms)

        np.testing.assert_allclose(hessian_5point, hessian_analytical, rtol=1e-6, atol=1e-6)

    def test_7point_scheme_works(self, water_molecule):
        calc = HarmonicCalculator()
        hc_7point = HessianCalculator(water_molecule, calc, method="7point", verbose=0)
        hessian_7point = hc_7point.calculate_numerical_hessian()

        assert hessian_7point.shape == (9, 9)
        asym = np.max(np.abs(hessian_7point - hessian_7point.T))
        assert asym < 1e-10

    def test_backward_compatibility(self, water_molecule):
        calc = HarmonicCalculator(k=1.0)

        hc_forward = HessianCalculator(water_molecule, calc, method="forward", verbose=0)
        hessian_forward = hc_forward.calculate_numerical_hessian()

        hc_central = HessianCalculator(water_molecule, calc, method="central", verbose=0)
        hessian_central = hc_central.calculate_numerical_hessian()

        hc_5point = HessianCalculator(water_molecule, calc, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        assert hessian_forward.shape == hessian_central.shape == hessian_5point.shape

    def test_invalid_method_raises_error(self):
        atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
        calc = HarmonicCalculator()

        with pytest.raises(ValueError, match="Unknown finite difference method"):
            HessianCalculator(atoms, calc, method="invalid", verbose=0)


# ============================================================================
# Richardson Extrapolation Tests
# ============================================================================


class TestRichardsonExtrapolation:
    def test_5point_with_richardson(self, harmonic_atoms):
        calc = HarmonicCalculator(k=1.0)

        hc_5point_rich = HessianCalculator(
            harmonic_atoms,
            calc,
            delta=0.01,
            method="5point",
            richardson=True,
            delta2=0.005,
            verbose=0,
        )
        hessian_5point_rich = hc_5point_rich.calculate_numerical_hessian()

        hc_5point = HessianCalculator(harmonic_atoms, calc, delta=0.01, method="5point", verbose=0)
        hessian_5point = hc_5point.calculate_numerical_hessian()

        hessian_analytical = calc.get_hessian(harmonic_atoms)

        error_with_rich = np.max(np.abs(hessian_5point_rich - hessian_analytical))
        error_without_rich = np.max(np.abs(hessian_5point - hessian_analytical))

        assert error_with_rich < 1e-10, "5-point with Richardson should be accurate"
        assert error_without_rich < 1e-10, "5-point without Richardson should be accurate"

    def test_richardson_order_detection(self, harmonic_atoms):
        calc = HarmonicCalculator()

        hc_3 = HessianCalculator(harmonic_atoms, calc, method="central", richardson=True, verbose=0)
        assert hc_3._richardson_order == 2

        hc_5 = HessianCalculator(harmonic_atoms, calc, method="5point", richardson=True, verbose=0)
        assert hc_5._richardson_order == 4

    def test_richardson_with_forward_raises_error(self):
        atoms = Atoms(symbols="H", positions=[[0, 0, 0]])
        calc = HarmonicCalculator()

        with pytest.raises(ValueError, match="Richardson extrapolation currently supported only"):
            HessianCalculator(atoms, calc, method="forward", richardson=True, verbose=0)

    def test_richardson_central_harmonic(self, harmonic_atoms):
        calc = HarmonicCalculator(k=1.0)

        hc_rich = HessianCalculator(
            harmonic_atoms,
            calc,
            delta=0.02,
            method="central",
            richardson=True,
            delta2=0.01,
            verbose=0,
        )
        hessian_rich = hc_rich.calculate_numerical_hessian()

        hc_no_rich = HessianCalculator(
            harmonic_atoms, calc, delta=0.01, method="central", verbose=0
        )
        hessian_no_rich = hc_no_rich.calculate_numerical_hessian()

        hessian_analytical = calc.get_hessian(harmonic_atoms)

        np.testing.assert_allclose(hessian_rich, hessian_analytical, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(hessian_no_rich, hessian_analytical, rtol=1e-6, atol=1e-6)

        error_rich = np.max(np.abs(hessian_rich - hessian_analytical))
        error_no_rich = np.max(np.abs(hessian_no_rich - hessian_analytical))
        assert error_rich < error_no_rich or np.isclose(error_rich, error_no_rich, rtol=1e-4)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_richardson_central_uma(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        hessian_analytical = atoms.calc.get_hessian(atoms)

        hc_rich = HessianCalculator(
            atoms,
            atoms.calc,
            delta=0.02,
            method="central",
            richardson=True,
            delta2=0.01,
            verbose=0,
        )
        hessian_rich = hc_rich.calculate_numerical_hessian()

        hc_no_rich = HessianCalculator(atoms, atoms.calc, delta=0.01, method="central", verbose=0)
        hessian_no_rich = hc_no_rich.calculate_numerical_hessian()

        error_rich = np.max(np.abs(hessian_rich - hessian_analytical))
        error_no_rich = np.max(np.abs(hessian_no_rich - hessian_analytical))

        assert error_rich <= error_no_rich * 1.1, "Richardson should improve or maintain accuracy"
        np.testing.assert_allclose(hessian_rich, hessian_analytical, rtol=0.015, atol=1.0)


# ============================================================================
# Frequency Analysis Method Tests
# ============================================================================


class TestFrequencyAnalysisMethods:
    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_frequency_analysis_methods(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        freq_direct = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_direct = freq_direct.calculate_hessian(method="direct")

        freq_fd = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_fd = freq_fd.calculate_hessian(method="finite_differences")

        freq_auto = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_auto = freq_auto.calculate_hessian(method="auto")

        np.testing.assert_allclose(hessian_direct, hessian_auto, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(hessian_direct, hessian_fd, rtol=0.01, atol=0.5)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_auto_method_selection(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        freq_auto = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_auto = freq_auto.calculate_hessian(method="auto")

        freq_direct = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_direct = freq_direct.calculate_hessian(method="direct")

        np.testing.assert_allclose(hessian_auto, hessian_direct, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_frequency_analysis_batch_method(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        supports_batch = getattr(atoms.calc, "supports_batch_evaluation", False)

        if supports_batch:
            freq_batch = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
            hessian_batch = freq_batch.calculate_hessian(method="batch")

            freq_direct = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
            hessian_direct = freq_direct.calculate_hessian(method="direct")

            np.testing.assert_allclose(hessian_batch, hessian_direct, rtol=0.01, atol=0.5)
        else:
            pytest.skip("Calculator does not support batch evaluation")

    @pytest.mark.parametrize(
        "backend_name,calc_factory",
        [
            ("mace", lambda: qme.get_mace_calculator(model_name="mace-omol-0")),
            ("uma", lambda: qme.get_uma_calculator(model_name="uma-s-1p1")),
        ],
        ids=["mace", "uma"],
    )
    def test_backend_hessian_frequency_consistency(
        self, water_molecule, backend_name, calc_factory
    ):
        if backend_name == "mace" and not is_backend_available("mace"):
            pytest.skip("MACE backend not available")
        if backend_name == "uma" and not is_backend_available("uma"):
            pytest.skip("UMA backend not available")

        atoms = water_molecule.copy()
        atoms.calc = calc_factory()
        atoms.calc.ensure_loaded()

        if not hasattr(atoms.calc, "get_hessian"):
            pytest.skip(f"{backend_name} does not support analytical Hessian")

        freq_analytical = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        freq_analytical.calculate_hessian(method="direct")
        freq_analytical.diagonalize_hessian()
        freqs_analytical = freq_analytical._frequencies
        modes_analytical = freq_analytical._normal_modes

        freq_fd = FrequencyAnalysis(atoms, atoms.calc, delta=0.01, verbose=0)
        freq_fd.calculate_hessian(method="finite_differences")
        freq_fd.diagonalize_hessian()
        freqs_fd, modes_fd = freq_fd._frequencies, freq_fd._normal_modes

        freqs_analytical_vib = freqs_analytical[6:]
        freqs_fd_vib = freqs_fd[6:]
        modes_analytical_vib = modes_analytical[:, 6:]
        modes_fd_vib = modes_fd[:, 6:]

        np.testing.assert_allclose(
            freqs_analytical_vib,
            freqs_fd_vib,
            rtol=0.05,
            atol=50.0,
            err_msg=f"{backend_name} frequencies mismatch between analytical and FD",
        )

        for i in range(min(len(freqs_analytical_vib), len(freqs_fd_vib))):
            freq_diff = np.abs(freqs_fd_vib - freqs_analytical_vib[i])
            closest_idx = np.argmin(freq_diff)

            assert freq_diff[closest_idx] < 50.0, (
                f"Mode {i}: frequency mismatch {freq_diff[closest_idx]:.2f} cm^-1"
            )

            mode_overlap = np.abs(np.dot(modes_analytical_vib[:, i], modes_fd_vib[:, closest_idx]))
            assert mode_overlap > 0.8, (
                f"Mode {i}: poor overlap {mode_overlap:.3f} between analytical and FD modes"
            )

        assert np.all(freqs_analytical_vib > 0), (
            f"{backend_name} analytical has negative frequencies"
        )
        assert np.all(freqs_fd_vib > 0), f"{backend_name} FD has negative frequencies"


# ============================================================================
# Additional Feature Tests
# ============================================================================


class TestHessianAdvancedFeatures:
    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_hessian_with_indices(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        hc_full = HessianCalculator(atoms, atoms.calc, delta=0.01, verbose=0)
        hessian_full = hc_full.calculate_numerical_hessian()

        indices = [0, 1]
        hc_partial = HessianCalculator(atoms, atoms.calc, delta=0.01, indices=indices, verbose=0)
        hessian_partial = hc_partial.calculate_numerical_hessian()

        assert hessian_partial.shape == (6, 6)
        assert hessian_full.shape == (9, 9)

        hessian_full_block = hessian_full[:6, :6]
        np.testing.assert_allclose(hessian_partial, hessian_full_block, rtol=0.01, atol=0.5)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_delta_values(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        hessian_analytical = atoms.calc.get_hessian(atoms)

        deltas = [0.005, 0.01, 0.02]
        hessians = []

        for delta in deltas:
            hc = HessianCalculator(atoms, atoms.calc, delta=delta, method="central", verbose=0)
            hessian = hc.calculate_numerical_hessian()
            hessians.append(hessian)

        for _i, (hessian, delta) in enumerate(zip(hessians, deltas, strict=False)):
            rtol = 0.02 if delta >= 0.02 else 0.015
            atol = 2.0 if delta >= 0.02 else 1.0
            np.testing.assert_allclose(hessian, hessian_analytical, rtol=rtol, atol=atol)

        np.testing.assert_allclose(hessians[0], hessians[1], rtol=0.02, atol=0.5)
        np.testing.assert_allclose(hessians[1], hessians[2], rtol=0.03, atol=0.8)
