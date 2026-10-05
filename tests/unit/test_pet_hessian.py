"""Tests for PET analytical Hessian (math SDPA + FD fallback)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from famex.potentials.pet_potential import PETPotential, _math_attention_context
from tests.test_utils import requires_backend


class TestMathAttentionContext:
    def test_context_is_usable(self):
        with _math_attention_context():
            pass


class TestPETHessianFallback:
    def test_get_hessian_falls_back_to_fd(self):
        pot = PETPotential(model_name="pet-mad-s", device="cpu")
        pot.atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        fd = np.eye(6, dtype=np.float64)

        with (
            patch.object(
                PETPotential,
                "_compute_analytical_hessian",
                side_effect=RuntimeError(
                    "derivative for aten::_scaled_dot_product_efficient_attention_backward "
                    "is not implemented"
                ),
            ),
            patch.object(
                PETPotential,
                "_hessian_via_finite_differences",
                return_value=fd,
            ) as mock_fd,
            pytest.warns(UserWarning, match="falling back to finite differences"),
        ):
            result = pot.get_hessian()

        assert np.allclose(result, fd)
        mock_fd.assert_called_once()
        assert "hessian" in pot.results


@requires_backend("pet")
class TestPETHessianIntegration:
    @pytest.mark.slow
    def test_water_analytical_hessian(self, water_molecule):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        atoms = water_molecule.copy()
        calc = PETPotential(model_name="pet-mad-s", device=device)
        atoms.calc = calc

        hessian = calc.get_hessian(atoms)
        assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
        assert np.all(np.isfinite(hessian))
        asymmetry = float(np.max(np.abs(hessian - hessian.T)))
        assert asymmetry < 1e-8, f"Hessian not symmetric: {asymmetry:.2e}"
