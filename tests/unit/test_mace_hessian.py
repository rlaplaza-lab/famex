"""Unit tests for MACEPotential Hessian symmetrization."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from ase import Atoms

from famex.potentials.mace_potential import MACEPotential


class TestMACEHessianSymmetrization:
    """MACE get_hessian should return (H + H.T) / 2."""

    def _stub(self, hessian: np.ndarray) -> MACEPotential:
        pot = MACEPotential.__new__(MACEPotential)
        pot._calc = MagicMock()
        pot._calc.get_hessian = MagicMock(return_value=hessian)
        pot.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        return pot

    def test_symmetrize_square_hessian(self):
        raw = np.arange(81, dtype=np.float64).reshape(9, 9)
        result = self._stub(raw).get_hessian()
        np.testing.assert_allclose(result, 0.5 * (raw + raw.T))
        assert np.max(np.abs(result - result.T)) < 1e-15

    def test_symmetrize_reshaped_3d_hessian(self):
        raw = np.arange(81, dtype=np.float64).reshape(9, 9)
        result = self._stub(raw.reshape(9, 3, 3)).get_hessian()
        assert result.shape == (9, 9)
        np.testing.assert_allclose(result, 0.5 * (raw + raw.T))
