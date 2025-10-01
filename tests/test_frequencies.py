import numpy as np
import pytest
from ase.build import molecule

import qme
from qme.analysis.frequency import FrequencyAnalysis, HessianCalculator


def _mocked_atoms(name: str = "H2"):
    atoms = molecule(name)
    atoms.calc = qme.MockCalculator(backend="mock")
    return atoms


class TestFrequencyBasics:
    def test_linear_molecule_degrees(self):
        atoms = _mocked_atoms("H2")
        fa = FrequencyAnalysis(atoms, atoms.calc, delta=0.01)
        assert fa.nfree == 5

    def test_water_hessian_and_modes(self):
        atoms = _mocked_atoms("H2O")
        fa = FrequencyAnalysis(atoms, atoms.calc, delta=0.01)
        hessian = fa.calculate_hessian(method="finite_differences")
        freqs, modes = fa.diagonalize_hessian()
        assert hessian.shape == (9, 9)
        assert len(freqs) == 9
        assert modes.shape == (9, 9)

    def test_units_and_zpe(self):
        atoms = _mocked_atoms("H2O")
        fa = FrequencyAnalysis(atoms, atoms.calc)
        fa.calculate_hessian()
        fa.diagonalize_hessian()
        cm1 = fa.get_frequencies("cm-1")
        meV = fa.get_frequencies("meV")
        THz = fa.get_frequencies("THz")
        assert len(cm1) == len(meV) == len(THz)
        zpe = fa.get_zero_point_energy()
        assert isinstance(zpe, float)


class TestHessianCalculator:
    def test_hessian_dimensions(self):
        atoms = _mocked_atoms("H2")
        hc = HessianCalculator(atoms, atoms.calc)
        h = hc.calculate_numerical_hessian()
        assert h.shape == (6, 6)

    def test_subset_indices(self):
        atoms = _mocked_atoms("H2")
        hc = HessianCalculator(atoms, atoms.calc, indices=[0])
        h = hc.calculate_numerical_hessian()
        assert h.shape == (3, 3)
