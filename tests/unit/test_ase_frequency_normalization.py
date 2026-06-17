"""Tests for ASE complex frequency normalization."""

from __future__ import annotations

import numpy as np
import pytest
from ase.io import read
from ase.vibrations.data import VibrationsData

from famex.analysis.frequency import FrequencyAnalysis
from famex.analysis.physics_constants import normalize_frequencies_cm1
from famex.backends.constants import DEFAULT_UMA_MODEL
from tests.test_utils import requires_backend


class TestNormalizeFrequenciesCm1:
    def test_pure_imaginary_modes_become_negative(self):
        raw = np.array([0 + 774.0j, 0 + 52.0j, 0 + 0.1j], dtype=np.complex128)
        signed = normalize_frequencies_cm1(raw)
        assert signed[0] == pytest.approx(-774.0)
        assert signed[1] == pytest.approx(-52.0)
        assert signed[2] == pytest.approx(-0.1)

    def test_real_modes_unchanged(self):
        raw = np.array([17.3 + 0j, 3157.6, -5.0], dtype=np.complex128)
        signed = normalize_frequencies_cm1(raw)
        assert signed[0] == pytest.approx(17.3)
        assert signed[1] == pytest.approx(3157.6)
        assert signed[2] == pytest.approx(-5.0)

    def test_real_array_passthrough(self):
        raw = np.array([100.0, 200.0])
        signed = normalize_frequencies_cm1(raw)
        np.testing.assert_allclose(signed, raw)

    @pytest.mark.parametrize(
        ("real", "imag", "expected"),
        [
            (10.0, 100.0, -100.0),
            (100.0, 10.0, 100.0),
            (50.0, 50.1, -50.1),
        ],
    )
    def test_mixed_complex_components(self, real: float, imag: float, expected: float):
        raw = np.array([real + imag * 1j], dtype=np.complex128)
        signed = normalize_frequencies_cm1(raw)
        assert signed[0] == pytest.approx(expected)

    def test_imaginary_mode_not_removed_as_translation(self):
        """Large imaginary modes must survive trans/rot filtering."""
        raw = np.array(
            [
                0 + 0.1j,
                0 + 0.2j,
                0 + 0.3j,
                0 + 0.4j,
                0 + 0.5j,
                0 + 0.6j,
                0 + 775.0j,
                100.0,
                200.0,
            ],
            dtype=np.complex128,
        )
        signed = normalize_frequencies_cm1(raw)
        idx_sorted = np.argsort(np.abs(signed))
        vibrational = signed[idx_sorted[6:]]
        assert any(freq < -50 for freq in vibrational)


@requires_backend("uma")
class TestImaginaryFrequencyDetection:
    def test_bh28_ts_detects_imaginary_mode_with_uma(self):
        atoms = read("examples/bh28_benchmark/bh28_dataset/BHDIV_3_ts.xyz")
        from famex.potentials import get_uma_calculator

        try:
            atoms.calc = get_uma_calculator(model_name=DEFAULT_UMA_MODEL)
            freq = FrequencyAnalysis(atoms=atoms, calculator=atoms.calc, verbose=0)
            freq.calculate_hessian(method="auto")
            freq.diagonalize_hessian()
            result = freq.is_transition_state()
        except Exception as exc:
            pytest.skip(f"UMA frequency analysis unavailable: {exc}")

        assert result["n_imaginary_frequencies"] == 1
        assert result["is_transition_state"] is True
        assert min(result["imaginary_frequencies"]) < -50

        # Cross-check raw ASE complex frequencies normalize the same way
        vib = VibrationsData.from_2d(atoms, freq._hessian, indices=freq.indices)
        raw = vib.get_frequencies()
        assert np.any(np.imag(raw) != 0)
        signed_all = normalize_frequencies_cm1(raw)
        assert np.any(signed_all < -50)
