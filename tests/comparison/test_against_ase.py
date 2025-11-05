from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.vibrations import Vibrations

from qme.analysis.frequency import FrequencyAnalysis
from qme.analysis.hessian import HessianCalculator
from qme.analysis.molecular_properties import determine_degrees_of_freedom
from qme.potentials.mock_potential import MockCalculator


class TestQMEvsASEHessian:
    @pytest.mark.parametrize("delta", [0.01])
    def test_hessian_and_frequencies(self, tmp_path, delta):
        # Build a small molecule (water-like geometry)
        atoms = Atoms(
            symbols="H2O",
            positions=[
                [0.0000, 0.0000, 0.0000],
                [0.9572, 0.0000, 0.0000],
                [-0.2390, 0.9270, 0.0000],
            ],
        )
        atoms.calc = MockCalculator(force_constant=1.0)

        indices = list(range(len(atoms)))

        # QME: numerical Hessian (central differences)
        qme_hess = HessianCalculator(
            atoms=atoms,
            calculator=atoms.calc,
            delta=delta,
            method="central",
            indices=indices,
            verbose=0,
        ).calculate_numerical_hessian()

        # ASE: Vibrations Hessian with same delta and indices
        vib = Vibrations(atoms, indices=indices, delta=delta, name=str(tmp_path / "vib"))
        vib.run()
        vib.read()
        ase_hessian = vib.H.copy()  # 2D Hessian array in eV/Å^2

        # Compare Hessians (allowing small numerical tolerance)
        assert qme_hess.shape == ase_hessian.shape == (3 * len(indices), 3 * len(indices))
        # Symmetry checks
        np.testing.assert_allclose(qme_hess, qme_hess.T, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ase_hessian, ase_hessian.T, rtol=1e-6, atol=1e-6)
        # Element-wise closeness
        np.testing.assert_allclose(qme_hess, ase_hessian, rtol=5e-3, atol=5e-3)

        # QME: frequencies via FrequencyAnalysis
        qme_freq = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=delta,
            indices=indices,
            verbose=0,
        )
        qme_freq.calculate_hessian(method="finite_differences")
        qme_freq.diagonalize_hessian()
        fq_qme = qme_freq.get_frequencies(unit="cm-1")

        # ASE: frequencies (cm^-1)
        fq_ase_all = vib.get_frequencies()

        # Remove translational/rotational modes consistently
        nfree = determine_degrees_of_freedom(atoms, indices)
        idx_sorted = np.argsort(np.abs(fq_ase_all))
        fq_ase = fq_ase_all[idx_sorted[nfree:]]

        # Compare vibrational frequencies (signs: imaginary modes negative)
        assert fq_qme.shape == fq_ase.shape
        np.testing.assert_allclose(fq_qme, fq_ase, rtol=5e-2, atol=20.0)

        # Clean up ASE vibrations files
        vib.clean()
