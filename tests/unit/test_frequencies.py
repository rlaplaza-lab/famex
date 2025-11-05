from __future__ import annotations

import numpy as np
import pytest

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator
from tests.test_utils import StandardTestAssertions, TestMoleculeFactory


@pytest.fixture
def h2_molecule():
    atoms = TestMoleculeFactory.get_h2_equilibrium()
    atoms.calc = qme.MockCalculator(backend="mock")
    return atoms


@pytest.fixture
def h2o_molecule():
    atoms = TestMoleculeFactory.get_h2o_equilibrium()
    atoms.calc = qme.MockCalculator(backend="mock")
    return atoms


class TestFrequencyAnalysis:
    @pytest.mark.parametrize(
        ("expected_dof", "molecule_fixture"),
        [
            (5, "h2_molecule"),  # Linear molecule: 3N-5
            (6, "h2o_molecule"),  # Non-linear molecule: 3N-6
        ],
    )
    def test_molecule_degrees_of_freedom(self, request, expected_dof, molecule_fixture):
        atoms = request.getfixturevalue(molecule_fixture)
        fa = FrequencyAnalysis(atoms, atoms.calc, delta=0.01)
        assert fa.nfree == expected_dof

    def test_hessian_and_modes(self, h2o_molecule):
        fa = FrequencyAnalysis(h2o_molecule, h2o_molecule.calc, delta=0.01)
        hessian = fa.calculate_hessian(method="finite_differences")
        freqs, modes = fa.diagonalize_hessian()

        # Check dimensions
        expected_size = len(h2o_molecule) * 3
        assert hessian.shape == (expected_size, expected_size)
        assert len(freqs) == expected_size
        assert modes.shape == (expected_size, expected_size)

        # Check that frequencies are reasonable
        StandardTestAssertions.assert_frequencies_valid(freqs)

    @pytest.mark.parametrize("unit", ["cm-1", "meV", "THz"])
    def test_frequency_units(self, h2o_molecule, unit):
        fa = FrequencyAnalysis(h2o_molecule, h2o_molecule.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        frequencies = fa.get_frequencies(unit)
        assert isinstance(frequencies, list | np.ndarray)
        assert len(frequencies) > 0
        StandardTestAssertions.assert_frequencies_valid(frequencies)

    def test_zero_point_energy(self, h2o_molecule):
        fa = FrequencyAnalysis(h2o_molecule, h2o_molecule.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()

        zpe = fa.get_zero_point_energy()
        assert isinstance(zpe, float)
        assert zpe >= 0, "Zero-point energy should be non-negative"

    def test_atoms_not_modified_in_place(self, h2o_molecule):
        atoms = h2o_molecule.copy()
        atoms.calc = qme.MockCalculator(backend="mock")
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

        # Test with different calculator
        different_calc = qme.MockCalculator(backend="mock")
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
    def test_hessian_dimensions(self, h2o_molecule, indices, expected_shape):
        hc = HessianCalculator(h2o_molecule, h2o_molecule.calc, indices=indices)
        hessian = hc.calculate_numerical_hessian()
        StandardTestAssertions.assert_hessian_valid(hessian, expected_shape)
