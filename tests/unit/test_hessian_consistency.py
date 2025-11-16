from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator
from qme.analysis.utils import has_calculator_property
from qme.backends.availability import is_backend_available
from tests.test_constants import (
    DEFAULT_DELTA,
    EXTRA_TIGHT_TOL,
    FAIRCHEM_LOOP_REPRO_TOL,
    FD_5POINT_TOL,
    FD_CENTRAL_TOL,
    FD_FORWARD_TOL,
    FREQUENCY_MODE_TOL,
    HARMONIC_TOL,
    HESSIAN_NEAR_SYMMETRY_TOL,
    HESSIAN_SYMMETRY_TOL,
    LOOSE_DELTA,
    RICHARDSON_ERROR_TOL,
    TIGHT_DELTA,
    TIGHT_TOL,
    UMA_MACE_FREQUENCY_TOL,
    UMA_MACE_HESSIAN_TOL,
)
from tests.test_utils import HarmonicCalculator, parametrize_backends

# ============================================================================
# Helper Functions
# ============================================================================


def get_backend_calculator_with_hessian(backend_name: str, model_name: str | None = None):
    """Get a calculator for a backend if it's available and supports analytical hessian.

    Args:
        backend_name: Name of the backend ('mace' or 'uma')
        model_name: Optional model name. If None, uses default for backend.

    Returns
    -------
    Calculator instance if backend is available and supports hessian.

    Raises
    ------
    pytest.skip: If backend is not available or doesn't support hessian.
    """
    if not is_backend_available(backend_name):
        pytest.skip(f"{backend_name} backend not available")

    # Create calculator
    if backend_name == "mace":
        calc = qme.get_mace_calculator(model_name=model_name or "mace-omol-0")
    elif backend_name == "uma":
        calc = qme.get_uma_calculator(model_name=model_name or "uma-s-1p1")
    else:
        pytest.skip(f"Unknown backend: {backend_name}")

    calc.ensure_loaded()

    # Check if calculator supports analytical hessian
    if not has_calculator_property(calc, "hessian"):
        pytest.skip(f"{backend_name} does not support analytical Hessian")

    return calc


# ============================================================================
# Backend Consistency Tests
# ============================================================================


class TestBackendHessianConsistency:
    """Unified tests for backend hessian consistency across all available backends."""

    @parametrize_backends(backends=["mace", "uma"])
    @pytest.mark.slow
    def test_backend_hessian_consistency(self, water_molecule, backend):
        """Test that analytical hessian matches finite difference hessian."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian(backend)

        hessian_analytical = atoms.calc.get_hessian(atoms)
        # Use smaller delta for higher accuracy reference
        hc_tight = HessianCalculator(atoms, atoms.calc, delta=TIGHT_DELTA, verbose=0)
        hessian_fd_tight = hc_tight.calculate_numerical_hessian()

        assert hessian_analytical.shape == hessian_fd_tight.shape
        rtol, atol = UMA_MACE_HESSIAN_TOL
        np.testing.assert_allclose(hessian_analytical, hessian_fd_tight, rtol=rtol, atol=atol)

    @parametrize_backends(backends=["mace", "uma"])
    @pytest.mark.slow
    def test_backend_hessian_methane(self, methane_molecule, backend):
        """Test hessian consistency for methane molecule."""
        atoms = methane_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian(backend)

        hessian_analytical = atoms.calc.get_hessian(atoms)
        # Use smaller delta for higher accuracy reference
        hc = HessianCalculator(atoms, atoms.calc, delta=TIGHT_DELTA, method="central", verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        assert hessian_analytical.shape == hessian_fd.shape
        rtol, atol = UMA_MACE_HESSIAN_TOL
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=rtol, atol=atol)

    @parametrize_backends(backends=["mace", "uma"])
    @pytest.mark.parametrize("method", ["central", "5point"], ids=["central", "5point"])
    @pytest.mark.slow
    def test_backend_fd_methods(self, water_molecule, backend, method):
        """Test that analytical hessian matches different FD methods."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian(backend)

        hessian_analytical = atoms.calc.get_hessian(atoms)
        # Use smaller delta for higher accuracy reference
        delta = TIGHT_DELTA if method != "forward" else DEFAULT_DELTA
        hc = HessianCalculator(atoms, atoms.calc, delta=delta, method=method, verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        if method == "forward":
            rtol, atol = FD_FORWARD_TOL
        elif method == "central":
            rtol, atol = FD_CENTRAL_TOL
        else:  # 5point
            rtol, atol = FD_5POINT_TOL

        assert hessian_analytical.shape == hessian_fd.shape
        np.testing.assert_allclose(hessian_analytical, hessian_fd, rtol=rtol, atol=atol)

    @parametrize_backends(backends=["mace", "uma"])
    @pytest.mark.slow
    def test_backend_hessian_frequency_consistency(self, water_molecule, backend):
        """Test that analytical and FD frequencies match."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian(backend)

        freq_analytical = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA, verbose=0)
        freq_analytical.calculate_hessian(method="direct")
        freq_analytical.diagonalize_hessian()
        freqs_analytical = freq_analytical._frequencies
        modes_analytical = freq_analytical._normal_modes

        # Use smaller delta for higher accuracy reference
        freq_fd = FrequencyAnalysis(atoms, atoms.calc, delta=TIGHT_DELTA, verbose=0)
        freq_fd.calculate_hessian(method="finite_differences")
        freq_fd.diagonalize_hessian()
        freqs_fd, modes_fd = freq_fd._frequencies, freq_fd._normal_modes

        freqs_analytical_vib = freqs_analytical[6:]
        freqs_fd_vib = freqs_fd[6:]
        modes_analytical_vib = modes_analytical[:, 6:]
        modes_fd_vib = modes_fd[:, 6:]

        # Analytical frequencies should match FD closely
        rtol, atol = UMA_MACE_FREQUENCY_TOL
        np.testing.assert_allclose(
            freqs_analytical_vib,
            freqs_fd_vib,
            rtol=rtol,
            atol=atol,
            err_msg=f"{backend} frequencies mismatch between analytical and FD",
        )

        # Use tighter frequency mismatch tolerance
        for i in range(len(freqs_analytical_vib)):
            freq_diff = np.abs(freqs_fd_vib - freqs_analytical_vib[i])
            closest_idx = np.argmin(freq_diff)
            assert freq_diff[closest_idx] < FREQUENCY_MODE_TOL, (
                f"Mode {i}: frequency mismatch {freq_diff[closest_idx]:.2f} cm^-1"
            )
            mode_overlap = np.abs(np.dot(modes_analytical_vib[:, i], modes_fd_vib[:, closest_idx]))
            assert mode_overlap > 0.8, f"Mode {i}: poor overlap {mode_overlap:.3f}"

        assert np.all(freqs_analytical_vib > 0), f"{backend} analytical has negative frequencies"
        assert np.all(freqs_fd_vib > 0), f"{backend} FD has negative frequencies"


class TestUMAHessianMethods:
    """Tests specific to UMA backend's different hessian computation methods."""

    @pytest.mark.parametrize(
        "method",
        ["vmap", "double_backward", "fairchem", "fairchem_loop"],
        ids=["vmap", "double_backward", "fairchem", "fairchem_loop"],
    )
    @pytest.mark.parametrize("symmetrize", [True, False], ids=["sym", "no_sym"])
    def test_uma_hessian_method_consistency(self, water_molecule, method, symmetrize):
        """Test UMA's different hessian computation methods."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        hessian1 = atoms.calc.get_hessian(atoms, method=method, symmetrize=symmetrize)
        hessian2 = atoms.calc.get_hessian(atoms, method=method, symmetrize=symmetrize)

        # Use slightly relaxed tolerance for fairchem_loop due to loop-based implementation
        if method == "fairchem_loop":
            rtol_repro, atol_repro = FAIRCHEM_LOOP_REPRO_TOL
        else:
            rtol_repro, atol_repro = HARMONIC_TOL
        np.testing.assert_allclose(
            hessian1,
            hessian2,
            rtol=rtol_repro,
            atol=atol_repro,
            err_msg=f"UMA {method} symmetrize={symmetrize} not reproducible",
        )

        # Use smaller delta for higher accuracy reference
        hc = HessianCalculator(atoms, atoms.calc, delta=TIGHT_DELTA, method="central", verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        # Tightened tolerances: improved implementation should match FD very closely
        rtol, atol = UMA_MACE_HESSIAN_TOL
        np.testing.assert_allclose(
            hessian1,
            hessian_fd,
            rtol=rtol,
            atol=atol,
            err_msg=f"UMA {method} symmetrize={symmetrize} does not match FD",
        )

    def test_uma_methods_consistency(self, water_molecule):
        """Test that all UMA hessian methods give consistent results."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        hessian_vmap = atoms.calc.get_hessian(atoms, method="vmap", symmetrize=True)
        hessian_db = atoms.calc.get_hessian(atoms, method="double_backward", symmetrize=True)
        hessian_fairchem = atoms.calc.get_hessian(atoms, method="fairchem", symmetrize=True)
        hessian_fairchem_loop = atoms.calc.get_hessian(
            atoms, method="fairchem_loop", symmetrize=True
        )

        # All methods should give very similar results with improved implementation
        rtol, atol = UMA_MACE_HESSIAN_TOL
        np.testing.assert_allclose(
            hessian_vmap,
            hessian_db,
            rtol=rtol,
            atol=atol,
            err_msg="UMA vmap and double_backward methods give different results",
        )
        np.testing.assert_allclose(
            hessian_fairchem,
            hessian_db,
            rtol=rtol,
            atol=atol,
            err_msg="UMA fairchem and double_backward methods give different results",
        )
        np.testing.assert_allclose(
            hessian_fairchem_loop,
            hessian_db,
            rtol=rtol,
            atol=atol,
            err_msg="UMA fairchem_loop and double_backward methods give different results",
        )

        # Use smaller delta for higher accuracy reference
        hc = HessianCalculator(atoms, atoms.calc, delta=TIGHT_DELTA, method="central", verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        # Tightened tolerances: improved implementation should match FD very closely
        np.testing.assert_allclose(
            hessian_vmap, hessian_fd, rtol=rtol, atol=atol, err_msg="UMA vmap does not match FD"
        )
        np.testing.assert_allclose(
            hessian_db,
            hessian_fd,
            rtol=rtol,
            atol=atol,
            err_msg="UMA double_backward does not match FD",
        )
        np.testing.assert_allclose(
            hessian_fairchem,
            hessian_fd,
            rtol=rtol,
            atol=atol,
            err_msg="UMA fairchem does not match FD",
        )
        np.testing.assert_allclose(
            hessian_fairchem_loop,
            hessian_fd,
            rtol=rtol,
            atol=atol,
            err_msg="UMA fairchem_loop does not match FD",
        )

    def test_uma_symmetrization_effect(self, water_molecule):
        """Test the effect of symmetrization on UMA hessian."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        hessian_sym = atoms.calc.get_hessian(atoms, method="double_backward", symmetrize=True)
        hessian_no_sym = atoms.calc.get_hessian(atoms, method="double_backward", symmetrize=False)

        sym_error_sym = np.max(np.abs(hessian_sym - hessian_sym.T))
        sym_error_no_sym = np.max(np.abs(hessian_no_sym - hessian_no_sym.T))

        assert sym_error_sym < HESSIAN_SYMMETRY_TOL, "Symmetrized Hessian should be symmetric"
        assert sym_error_no_sym < HESSIAN_NEAR_SYMMETRY_TOL, (
            "Non-symmetrized Hessian should be nearly symmetric"
        )

        # Use smaller delta for higher accuracy reference
        hc = HessianCalculator(atoms, atoms.calc, delta=TIGHT_DELTA, method="central", verbose=0)
        hessian_fd = hc.calculate_numerical_hessian()

        error_sym = np.max(np.abs(hessian_sym - hessian_fd))
        error_no_sym = np.max(np.abs(hessian_no_sym - hessian_fd))

        # With improved implementation, both should be very accurate
        assert error_sym < 0.01, f"Symmetrized Hessian error too large: {error_sym:.6f}"
        assert error_no_sym < 0.02, f"Non-symmetrized Hessian error too large: {error_no_sym:.6f}"
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

        hc_5point = HessianCalculator(
            harmonic_atoms, calc, delta=DEFAULT_DELTA, method="5point", verbose=0
        )
        hessian_5point = hc_5point.calculate_numerical_hessian()

        hessian_analytical = calc.get_hessian(harmonic_atoms)

        np.testing.assert_allclose(
            hessian_5point, hessian_analytical, rtol=HARMONIC_TOL[0], atol=HARMONIC_TOL[1]
        )

    def test_7point_scheme_works(self, water_molecule):
        calc = HarmonicCalculator()
        hc_7point = HessianCalculator(water_molecule, calc, method="7point", verbose=0)
        hessian_7point = hc_7point.calculate_numerical_hessian()

        assert hessian_7point.shape == (9, 9)
        asym = np.max(np.abs(hessian_7point - hessian_7point.T))
        assert asym < HESSIAN_SYMMETRY_TOL

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
            delta=DEFAULT_DELTA,
            method="5point",
            richardson=True,
            delta2=0.005,
            verbose=0,
        )
        hessian_5point_rich = hc_5point_rich.calculate_numerical_hessian()

        hc_5point = HessianCalculator(
            harmonic_atoms, calc, delta=DEFAULT_DELTA, method="5point", verbose=0
        )
        hessian_5point = hc_5point.calculate_numerical_hessian()

        hessian_analytical = calc.get_hessian(harmonic_atoms)

        error_with_rich = np.max(np.abs(hessian_5point_rich - hessian_analytical))
        error_without_rich = np.max(np.abs(hessian_5point - hessian_analytical))

        assert error_with_rich < RICHARDSON_ERROR_TOL, "5-point with Richardson should be accurate"
        assert error_without_rich < RICHARDSON_ERROR_TOL, (
            "5-point without Richardson should be accurate"
        )

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
            delta=LOOSE_DELTA,
            method="central",
            richardson=True,
            delta2=DEFAULT_DELTA,
            verbose=0,
        )
        hessian_rich = hc_rich.calculate_numerical_hessian()

        hc_no_rich = HessianCalculator(
            harmonic_atoms, calc, delta=DEFAULT_DELTA, method="central", verbose=0
        )
        hessian_no_rich = hc_no_rich.calculate_numerical_hessian()

        hessian_analytical = calc.get_hessian(harmonic_atoms)

        np.testing.assert_allclose(
            hessian_rich, hessian_analytical, rtol=EXTRA_TIGHT_TOL[0], atol=EXTRA_TIGHT_TOL[1]
        )
        np.testing.assert_allclose(
            hessian_no_rich, hessian_analytical, rtol=HARMONIC_TOL[0], atol=HARMONIC_TOL[1]
        )

        error_rich = np.max(np.abs(hessian_rich - hessian_analytical))
        error_no_rich = np.max(np.abs(hessian_no_rich - hessian_analytical))
        assert error_rich < error_no_rich or np.isclose(
            error_rich, error_no_rich, rtol=TIGHT_TOL[0]
        )

    def test_richardson_central_uma(self, water_molecule):
        """Test Richardson extrapolation with UMA backend."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        hessian_analytical = atoms.calc.get_hessian(atoms)

        hc_rich = HessianCalculator(
            atoms,
            atoms.calc,
            delta=LOOSE_DELTA,
            method="central",
            richardson=True,
            delta2=DEFAULT_DELTA,
            verbose=0,
        )
        hessian_rich = hc_rich.calculate_numerical_hessian()

        hc_no_rich = HessianCalculator(
            atoms, atoms.calc, delta=TIGHT_DELTA, method="central", verbose=0
        )
        hessian_no_rich = hc_no_rich.calculate_numerical_hessian()

        error_rich = np.max(np.abs(hessian_rich - hessian_analytical))
        error_no_rich = np.max(np.abs(hessian_no_rich - hessian_analytical))

        assert error_rich <= error_no_rich * 1.1, "Richardson should improve or maintain accuracy"
        # Tightened tolerances: improved implementation should match very closely
        np.testing.assert_allclose(
            hessian_rich, hessian_analytical, rtol=FD_CENTRAL_TOL[0], atol=FD_CENTRAL_TOL[1]
        )


# ============================================================================
# Frequency Analysis Method Tests
# ============================================================================


class TestFrequencyAnalysisMethods:
    def test_frequency_analysis_methods(self, water_molecule):
        """Test different frequency analysis methods."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        freq_direct = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA, verbose=0)
        hessian_direct = freq_direct.calculate_hessian(method="direct")

        freq_fd = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA, verbose=0)
        freq_fd.calculate_hessian(method="finite_differences")

        freq_auto = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA, verbose=0)
        hessian_auto = freq_auto.calculate_hessian(method="auto")

        np.testing.assert_allclose(
            hessian_direct, hessian_auto, rtol=TIGHT_TOL[0], atol=TIGHT_TOL[1]
        )
        # Use smaller delta and tightened tolerances
        freq_fd_tight = FrequencyAnalysis(atoms, atoms.calc, delta=TIGHT_DELTA, verbose=0)
        hessian_fd_tight = freq_fd_tight.calculate_hessian(method="finite_differences")
        np.testing.assert_allclose(
            hessian_direct,
            hessian_fd_tight,
            rtol=UMA_MACE_HESSIAN_TOL[0],
            atol=UMA_MACE_HESSIAN_TOL[1],
        )

    def test_auto_method_selection(self, water_molecule):
        """Test automatic method selection for frequency analysis."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        freq_auto = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA, verbose=0)
        hessian_auto = freq_auto.calculate_hessian(method="auto")

        freq_direct = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA, verbose=0)
        hessian_direct = freq_direct.calculate_hessian(method="direct")

        np.testing.assert_allclose(
            hessian_auto, hessian_direct, rtol=TIGHT_TOL[0], atol=TIGHT_TOL[1]
        )

    def test_frequency_analysis_batch_method(self, water_molecule):
        """Test batch evaluation method for frequency analysis."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        supports_batch = getattr(atoms.calc, "supports_batch_evaluation", False)

        if supports_batch:
            freq_batch = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA, verbose=0)
            hessian_batch = freq_batch.calculate_hessian(method="batch")

            freq_direct = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA, verbose=0)
            hessian_direct = freq_direct.calculate_hessian(method="direct")

            # Tightened tolerances: batch method should match direct very closely for UMA
            np.testing.assert_allclose(
                hessian_batch,
                hessian_direct,
                rtol=UMA_MACE_HESSIAN_TOL[0],
                atol=UMA_MACE_HESSIAN_TOL[1],
            )
        else:
            pytest.skip("Calculator does not support batch evaluation")


# ============================================================================
# Additional Feature Tests
# ============================================================================


class TestHessianAdvancedFeatures:
    def test_hessian_with_indices(self, water_molecule):
        """Test hessian calculation with partial atom indices."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        hc_full = HessianCalculator(atoms, atoms.calc, delta=TIGHT_DELTA, verbose=0)
        hessian_full = hc_full.calculate_numerical_hessian()

        indices = [0, 1]
        hc_partial = HessianCalculator(
            atoms, atoms.calc, delta=TIGHT_DELTA, indices=indices, verbose=0
        )
        hessian_partial = hc_partial.calculate_numerical_hessian()

        assert hessian_partial.shape == (6, 6)
        assert hessian_full.shape == (9, 9)

        hessian_full_block = hessian_full[:6, :6]
        # Tightened tolerances: UMA should match FD very closely even with indices
        np.testing.assert_allclose(
            hessian_partial,
            hessian_full_block,
            rtol=UMA_MACE_HESSIAN_TOL[0],
            atol=UMA_MACE_HESSIAN_TOL[1],
        )

    def test_delta_values(self, water_molecule):
        """Test hessian calculation with different delta values."""
        atoms = water_molecule.copy()
        atoms.calc = get_backend_calculator_with_hessian("uma")

        hessian_analytical = atoms.calc.get_hessian(atoms)

        deltas = [0.001, 0.005, 0.01]
        hessians = []

        for delta in deltas:
            hc = HessianCalculator(atoms, atoms.calc, delta=delta, method="central", verbose=0)
            hessian = hc.calculate_numerical_hessian()
            hessians.append(hessian)

        # Tightened tolerances: UMA should match analytical closely even with different deltas
        for _i, (hessian, delta) in enumerate(zip(hessians, deltas, strict=False)):
            rtol = FD_CENTRAL_TOL[0] if delta >= 0.01 else UMA_MACE_HESSIAN_TOL[0]
            atol = FD_CENTRAL_TOL[1] if delta >= 0.01 else UMA_MACE_HESSIAN_TOL[1]
            np.testing.assert_allclose(hessian, hessian_analytical, rtol=rtol, atol=atol)

        # Different deltas should give consistent results
        np.testing.assert_allclose(
            hessians[0], hessians[1], rtol=UMA_MACE_HESSIAN_TOL[0], atol=UMA_MACE_HESSIAN_TOL[1]
        )
        np.testing.assert_allclose(
            hessians[1], hessians[2], rtol=FD_CENTRAL_TOL[0], atol=FD_CENTRAL_TOL[1]
        )
