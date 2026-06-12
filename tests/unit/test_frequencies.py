from __future__ import annotations

import numpy as np
import pytest

import famex
from famex.analysis.frequency import FrequencyAnalysis, HessianCalculator
from tests.test_constants import DEFAULT_DELTA
from tests.test_utils import StandardTestAssertions


class TestFrequencyAnalysis:
    @pytest.mark.parametrize(
        ("expected_dof", "molecule_fixture"),
        [
            (5, "h2_equilibrium_molecule"),  # Linear molecule: 3N-5
            (6, "h2o_molecule"),  # Non-linear molecule: 3N-6
        ],
    )
    def test_molecule_degrees_of_freedom(
        self, request, expected_dof, molecule_fixture, mock_backend
    ):
        atoms = request.getfixturevalue(molecule_fixture)
        atoms.calc = mock_backend
        fa = FrequencyAnalysis(atoms, atoms.calc, delta=DEFAULT_DELTA)
        assert fa.nfree == expected_dof

    def test_hessian_and_modes(self, h2o_molecule_with_mock):
        fa = FrequencyAnalysis(
            h2o_molecule_with_mock, h2o_molecule_with_mock.calc, delta=DEFAULT_DELTA
        )
        hessian = fa.calculate_hessian(method="finite_differences")
        freqs, modes = fa.diagonalize_hessian()

        # Check dimensions
        expected_size = len(h2o_molecule_with_mock) * 3
        assert hessian.shape == (expected_size, expected_size)
        assert len(freqs) == expected_size
        assert modes.shape == (expected_size, expected_size)

        # Check that frequencies are reasonable
        StandardTestAssertions.assert_frequencies_valid(freqs)

    @pytest.mark.parametrize("unit", ["cm-1", "meV", "THz"])
    def test_frequency_units(self, h2o_molecule_with_mock, unit):
        fa = FrequencyAnalysis(h2o_molecule_with_mock, h2o_molecule_with_mock.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        frequencies = fa.get_frequencies(unit)
        assert isinstance(frequencies, list | np.ndarray)
        assert len(frequencies) > 0
        StandardTestAssertions.assert_frequencies_valid(frequencies)

    def test_zero_point_energy(self, h2o_molecule_with_mock):
        fa = FrequencyAnalysis(h2o_molecule_with_mock, h2o_molecule_with_mock.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        zpe = fa.get_zero_point_energy()
        assert isinstance(zpe, float)
        assert zpe >= 0, "Zero-point energy should be non-negative"

    def test_atoms_not_modified_in_place(self, h2o_molecule, mock_backend):
        h2o_molecule.calc = mock_backend
        atoms = h2o_molecule.copy()
        atoms.calc = mock_backend
        original_positions = atoms.positions.copy()
        original_calc_id = id(atoms.calc)

        # Run frequency analysis
        fa = FrequencyAnalysis(atoms, atoms.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        # Verify positions unchanged
        assert np.allclose(atoms.positions, original_positions), "Atoms positions were modified!"

        # Verify calculator unchanged (same object)
        assert id(atoms.calc) == original_calc_id, "Atoms calculator was replaced!"

        # Test with different calculator (create a new mock calculator instance)
        # Note: Intentionally creating a new instance (not using mock_backend fixture)
        # to test that FrequencyAnalysis can replace atoms.calc with a different calculator instance
        different_calc = famex.MockCalculator(backend="mock")
        FrequencyAnalysis(atoms, different_calc)

        # Calculator should be updated to different one
        assert id(atoms.calc) == id(different_calc), "Different calculator not applied!"

        # But positions should still be unchanged
        assert np.allclose(atoms.positions, original_positions), (
            "Positions modified with different calc!"
        )

    @pytest.mark.parametrize(
        ("indices", "expected_shape"),
        [
            (None, (9, 9)),  # All atoms
            ([0], (3, 3)),  # Single atom
            ([0, 1], (6, 6)),  # Two atoms
        ],
    )
    def test_hessian_dimensions(self, h2o_molecule_with_mock, indices, expected_shape):
        hc = HessianCalculator(h2o_molecule_with_mock, h2o_molecule_with_mock.calc, indices=indices)
        hessian = hc.calculate_numerical_hessian()
        StandardTestAssertions.assert_hessian_valid(hessian, expected_shape)
