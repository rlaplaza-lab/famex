"""Tests for UMA needs_hessian compile policy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from famex.backends.constants import DEFAULT_UMA_MODEL
from famex.core.explorer import Explorer, compute_needs_hessian
from tests.test_utils import requires_backend


class TestComputeNeedsHessian:
    @pytest.mark.parametrize(
        ("target", "optimizer", "freqs", "cleanup", "expected"),
        [
            ("minima", "trust-krylov", False, False, True),
            ("minima", "trust-exact", False, False, True),
            ("minima", "trust-ncg", False, False, True),
            ("minima", "newton-cg", False, False, True),
            ("minima", "lbfgs", False, False, False),
            ("minima", "bfgs", False, False, False),
            ("minima", "fire", False, False, False),
            ("minima", "lbfgs", True, False, True),
            ("minima", "lbfgs", False, True, True),
            ("ts", "trust-krylov", False, False, False),
            ("ts", "trust-ncg", False, False, False),
            ("ts", "trust-exact", False, False, True),
            ("ts", "rfo", False, False, True),
            ("ts", "rational-function", False, False, True),
            ("ts", "sella-analytical", False, False, True),
            ("ts", "sella", False, False, False),
            ("ts", "trust-krylov", True, False, True),
        ],
        ids=[
            "min_trust_krylov",
            "min_trust_exact",
            "min_trust_ncg",
            "min_newton_cg",
            "min_lbfgs",
            "min_bfgs",
            "min_fire",
            "min_lbfgs_freqs",
            "min_lbfgs_cleanup",
            "ts_trust_krylov",
            "ts_trust_ncg",
            "ts_trust_exact",
            "ts_rfo",
            "ts_rational_function",
            "ts_sella_analytical",
            "ts_sella",
            "ts_trust_krylov_freqs",
        ],
    )
    def test_compute_needs_hessian_matrix(self, target, optimizer, freqs, cleanup, expected):
        assert (
            compute_needs_hessian(
                target,
                optimizer,
                calculate_frequencies=freqs,
                cleanup_frequencies=cleanup,
            )
            is expected
        )


class TestExplorerNeedsHessian:
    def test_minima_trust_krylov_sets_needs_hessian(self, h2_molecule):
        explorer = Explorer(
            h2_molecule.copy(),
            backend="mock",
            target="minima",
            local_optimizer="trust-krylov",
        )
        assert explorer._needs_hessian() is True

    def test_frequency_check_sets_needs_hessian(self, h2_molecule):
        explorer = Explorer(
            h2_molecule.copy(),
            backend="mock",
            target="minima",
            local_optimizer="lbfgs",
        )
        assert explorer._needs_hessian() is False
        assert explorer._needs_hessian(calculate_frequencies=True) is True

    def test_lbfgs_does_not_set_needs_hessian(self, h2_molecule):
        explorer = Explorer(
            h2_molecule.copy(),
            backend="mock",
            target="minima",
            local_optimizer="lbfgs",
        )
        assert explorer._needs_hessian() is False

    def test_saddle_trust_krylov_without_frequencies(self, h2_molecule):
        explorer = Explorer(
            h2_molecule.copy(),
            backend="mock",
            target="ts",
            local_optimizer="trust-krylov",
        )
        assert explorer._needs_hessian() is False
        assert explorer._needs_hessian(calculate_frequencies=False) is False

    def test_default_ts_optimizer_needs_hessian(self, h2_molecule):
        explorer = Explorer(h2_molecule.copy(), backend="mock", target="ts")
        assert explorer._needs_hessian() is True

    def test_create_and_attach_forwards_needs_hessian(self, h2_molecule):
        explorer = Explorer(h2_molecule.copy(), backend="mock", target="minima")
        atoms = h2_molecule.copy()
        with patch.object(
            explorer.calculator_manager,
            "create_and_attach_calculator",
            return_value=MagicMock(),
        ) as mock_create:
            explorer._create_and_attach_calculator(atoms, needs_hessian=True)
            mock_create.assert_called_once_with(atoms, needs_hessian=True)


class TestUMAPotentialReload:
    def test_compiled_reloads_once_on_get_hessian(self):
        # uma_potential imports torch at module scope; keep collection torch-free.
        pytest.importorskip("torch")
        from famex.potentials.uma_potential import UMAPotential

        pot = UMAPotential(model_name=DEFAULT_UMA_MODEL, needs_hessian=False, device="cpu")
        pot.atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        load_count = {"n": 0}

        def fake_load() -> None:
            load_count["n"] += 1
            pot._calc = MagicMock()
            pot.predictor = MagicMock()
            pot._hessian_capable = pot.needs_hessian

        hess = np.eye(6, dtype=np.float64)

        with (
            patch.object(UMAPotential, "_load_calculator", side_effect=fake_load),
            patch.object(UMAPotential, "_hessian_via_fairchem_task", return_value=hess) as mock_h,
            patch.object(UMAPotential, "_set_atoms_charge_spin"),
        ):
            pot._calc = MagicMock()
            pot.predictor = MagicMock()
            pot._hessian_capable = False
            pot.needs_hessian = False

            h1 = pot.get_hessian()
            assert load_count["n"] == 1
            assert pot._hessian_capable is True
            assert np.allclose(h1, hess)
            assert mock_h.call_count == 1

            h2 = pot.get_hessian()
            assert load_count["n"] == 1
            assert mock_h.call_count == 2
            assert np.allclose(h2, hess)

    @requires_backend("uma")
    def test_bare_get_uma_calculator_defaults_compiled(self):
        from famex.potentials import get_uma_calculator

        calc = get_uma_calculator(model_name=DEFAULT_UMA_MODEL, device="cpu")
        assert calc.needs_hessian is False
        assert calc._hessian_capable is False


@requires_backend("uma")
class TestUMAHessianReloadIntegration:
    @pytest.mark.slow
    def test_water_hessian_after_reload(self, water_molecule):
        from famex.potentials import get_uma_calculator

        atoms = water_molecule.copy()
        calc = get_uma_calculator(
            model_name=DEFAULT_UMA_MODEL,
            needs_hessian=False,
            device="cpu",
        )
        atoms.calc = calc

        assert calc.needs_hessian is False
        assert calc._hessian_capable is False

        hessian = calc.get_hessian(atoms)

        assert calc._hessian_capable is True
        assert hessian.shape == (3 * len(atoms), 3 * len(atoms))
        assert np.all(np.isfinite(hessian))
        asymmetry = float(np.max(np.abs(hessian - hessian.T)))
        assert asymmetry < 1e-4, f"Hessian not symmetric enough: {asymmetry:.2e}"
