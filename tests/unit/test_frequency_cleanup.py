"""Tests for post-optimization frequency cleanup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, Hookean

from famex.strategies.frequency_cleanup import (
    _apply_perturbation,
    _perturb_along_modes,
    _perturb_random,
    _select_modes_to_follow,
    default_cleanup_optimizer_name,
    prepare_minima_hessian_optimizer_kwargs,
    relevant_imaginary_count,
    run_frequency_cleanup,
    stationary_point_is_clean,
    target_imaginary_count,
)
from tests.test_utils import HarmonicCalculator


class TestFrequencyCleanupHelpers:
    def test_default_cleanup_optimizer_name(self):
        assert default_cleanup_optimizer_name("minima") == "trust-krylov"
        assert default_cleanup_optimizer_name("ts") == "rfo"

    def test_prepare_minima_hessian_optimizer_kwargs_for_trust_krylov(self):
        explorer = MagicMock()
        explorer.optimizer_kwargs = {}
        explorer.force_finite_diff_hessian = False

        kwargs = prepare_minima_hessian_optimizer_kwargs("trust-krylov", explorer)

        assert kwargs["hessian_method"] == "auto"
        assert kwargs["hessian_update_freq"] == 1
        assert kwargs["use_bfgs_update"] is False
        assert kwargs["trust_radius"] == 0.1
        assert kwargs["max_trust_radius"] == 0.3

    def test_target_imaginary_count(self):
        assert target_imaginary_count("minima") == 0
        assert target_imaginary_count("ts") == 1

    def test_stationary_point_is_clean(self):
        assert stationary_point_is_clean({"is_minimum": True}, "minima")
        assert not stationary_point_is_clean({"is_minimum": False}, "minima")
        assert stationary_point_is_clean({"is_ts": True}, "ts")
        assert not stationary_point_is_clean({"is_ts": False}, "ts")

    def test_relevant_imaginary_count(self):
        minima_freq = {
            "minima_analysis": {"n_significant_imaginary_frequencies": 2},
        }
        ts_freq = {
            "ts_analysis": {"n_imaginary_frequencies": 1},
        }
        assert relevant_imaginary_count(minima_freq, "minima") == 2
        assert relevant_imaginary_count(ts_freq, "ts") == 1


class TestSelectModesToFollow:
    def test_minima_follows_all_imaginary(self):
        frequencies = np.array([-200.0, -120.0, 300.0, 450.0])
        assert _select_modes_to_follow(frequencies, "minima") == [0, 1]

    def test_minima_clean_returns_empty(self):
        frequencies = np.array([100.0, 200.0, 300.0])
        assert _select_modes_to_follow(frequencies, "minima") == []

    def test_ts_keeps_reaction_mode_removes_extras(self):
        # -500 is the most negative (reaction coordinate); -150 is the extra to remove.
        frequencies = np.array([-500.0, -150.0, 300.0])
        assert _select_modes_to_follow(frequencies, "ts") == [1]

    def test_ts_collapsed_follows_softest_real_mode(self):
        frequencies = np.array([80.0, 400.0, 250.0])
        assert _select_modes_to_follow(frequencies, "ts") == [0]

    def test_ts_single_imaginary_is_clean(self):
        frequencies = np.array([-300.0, 200.0, 400.0])
        assert _select_modes_to_follow(frequencies, "ts") == []


class TestPerturbations:
    def _two_atom_system(self):
        return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    def test_perturb_along_modes_moves_geometry(self):
        atoms = self._two_atom_system()
        mode = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        frequency_result = {
            "frequencies": [-250.0],
            "normal_modes": mode.reshape(6, 1).tolist(),
        }
        start = atoms.positions.copy()
        applied = _perturb_along_modes(
            atoms, frequency_result, "minima", amplitude=0.1, rng=np.random.default_rng(0)
        )
        assert applied is True
        assert not np.allclose(atoms.positions, start)

    def test_perturb_along_modes_supports_subset_indices(self):
        atoms = Atoms("H3", positions=[[0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
        frequency_result = {
            "frequencies": [-250.0],
            "normal_modes": np.array([[0.0], [0.0], [1.0]]).tolist(),
            "indices": [1],
        }
        start = atoms.positions.copy()
        applied = _perturb_along_modes(
            atoms, frequency_result, "minima", amplitude=0.1, rng=np.random.default_rng(0)
        )
        assert applied is True
        assert np.allclose(atoms.positions[0], start[0])
        assert not np.allclose(atoms.positions[1], start[1])

    def test_perturb_along_modes_bails_on_unmapped_subset(self):
        atoms = self._two_atom_system()
        frequency_result = {
            "frequencies": [-250.0],
            "normal_modes": np.array([[0.0], [0.0], [1.0]]).tolist(),
            "indices": [0, 1],
        }
        applied = _perturb_along_modes(
            atoms, frequency_result, "minima", amplitude=0.1, rng=np.random.default_rng(0)
        )
        assert applied is False

    def test_perturb_along_modes_clean_ts_returns_false(self):
        atoms = self._two_atom_system()
        mode = np.eye(6)[:, :1]
        frequency_result = {
            "frequencies": [-300.0],
            "normal_modes": mode.tolist(),
        }
        # Single imaginary mode => TS already clean => nothing to follow.
        applied = _perturb_along_modes(
            atoms, frequency_result, "ts", amplitude=0.1, rng=np.random.default_rng(0)
        )
        assert applied is False

    def test_perturb_random_moves_geometry(self):
        atoms = self._two_atom_system()
        start = atoms.positions.copy()
        _perturb_random(atoms, amplitude=0.1, rng=np.random.default_rng(0))
        assert not np.allclose(atoms.positions, start)

    def test_perturbation_respects_fixed_atoms(self):
        atoms = self._two_atom_system()
        atoms.set_constraint(FixAtoms(indices=[0]))
        start = atoms.positions.copy()
        _perturb_random(atoms, amplitude=0.2, rng=np.random.default_rng(1))
        # Fixed atom must not move; free atom should.
        assert np.allclose(atoms.positions[0], start[0])
        assert not np.allclose(atoms.positions[1], start[1])

    def test_mode_perturbation_respects_fixed_atoms(self):
        atoms = self._two_atom_system()
        atoms.set_constraint(FixAtoms(indices=[0]))
        mode = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        frequency_result = {
            "frequencies": [-250.0],
            "normal_modes": mode.reshape(6, 1).tolist(),
        }
        start = atoms.positions.copy()
        applied = _perturb_along_modes(
            atoms, frequency_result, "minima", amplitude=0.1, rng=np.random.default_rng(0)
        )
        assert applied is True
        assert np.allclose(atoms.positions[0], start[0])
        assert not np.allclose(atoms.positions[1], start[1])

    def test_apply_perturbation_respects_hookean_constraint(self):
        atoms = self._two_atom_system()
        atoms.set_constraint(Hookean(a1=0, a2=1, k=5.0, rt=0.1))
        initial_bond = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
        _perturb_random(atoms, amplitude=0.05, rng=np.random.default_rng(2))
        final_bond = float(np.linalg.norm(atoms.positions[1] - atoms.positions[0]))
        assert abs(final_bond - initial_bond) <= 0.1 + 1e-6

    def test_apply_perturbation_prefers_mode_then_random(self):
        atoms = self._two_atom_system()
        mode = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
        mode_result = {
            "frequencies": [-250.0],
            "normal_modes": mode.reshape(6, 1).tolist(),
        }
        assert (
            _apply_perturbation(
                atoms, mode_result, "minima", amplitude=0.1, rng=np.random.default_rng(0)
            )
            == "mode"
        )

        atoms2 = self._two_atom_system()
        empty_result: dict = {"frequencies": [], "normal_modes": []}
        assert (
            _apply_perturbation(
                atoms2, empty_result, "minima", amplitude=0.1, rng=np.random.default_rng(0)
            )
            == "random"
        )


class TestRunFrequencyCleanup:
    def test_skips_when_already_clean(self, h2_equilibrium_molecule):
        atoms = h2_equilibrium_molecule.copy()
        atoms.calc = HarmonicCalculator()
        explorer = MagicMock()
        explorer.calculate_frequencies.return_value = {
            "is_minimum": True,
            "minima_analysis": {"n_significant_imaginary_frequencies": 0},
        }
        explorer.optimizer_kwargs = {}
        explorer.ts_kwargs = {}

        _, freq_result, cleanup_steps, summary = run_frequency_cleanup(
            atoms,
            explorer,
            target="minima",
            fmax=0.05,
            temperature=298.15,
            verbose=0,
            prepare_optimizer_kwargs=lambda _name, _explorer: {},
            prepare_frequency_kwargs=None,
        )

        assert freq_result["is_minimum"] is True
        assert cleanup_steps == 0
        assert summary["skipped"] is True
        explorer.calculate_frequencies.assert_called_once()

    @pytest.mark.skipif(
        not __import__("famex").deps.has("sella"),
        reason="Hessian optimizers required for cleanup loop",
    )
    def test_runs_cleanup_from_current_geometry(self, h2_equilibrium_molecule):
        atoms = h2_equilibrium_molecule.copy()
        atoms.calc = HarmonicCalculator()
        explorer = MagicMock()
        explorer.optimizer_kwargs = {}
        explorer.ts_kwargs = {
            "cleanup_frequency_max_attempts": 1,
            "cleanup_frequency_steps": 2,
        }
        explorer.calculate_frequencies.side_effect = [
            {
                "is_minimum": False,
                "minima_analysis": {"n_significant_imaginary_frequencies": 1},
            },
            {
                "is_minimum": True,
                "minima_analysis": {"n_significant_imaginary_frequencies": 0},
            },
        ]

        initial_positions = atoms.positions.copy()
        with patch(
            "famex.strategies.frequency_cleanup._get_local_optimizer_class"
        ) as mock_get_optimizer:
            mock_optimizer = MagicMock()
            mock_optimizer.return_value.run.return_value = True
            mock_optimizer.return_value.get_number_of_steps.return_value = 2
            mock_get_optimizer.return_value = mock_optimizer

            result_atoms, freq_result, cleanup_steps, summary = run_frequency_cleanup(
                atoms,
                explorer,
                target="minima",
                fmax=0.05,
                temperature=298.15,
                verbose=0,
                prepare_optimizer_kwargs=lambda _name, _explorer: {},
                prepare_frequency_kwargs=None,
            )

        assert result_atoms is atoms
        assert atoms.positions.shape == initial_positions.shape
        mock_optimizer.assert_called_once_with(atoms, verbose=0)
        assert cleanup_steps >= 0
        assert summary["is_clean"] is True
        assert freq_result["is_minimum"] is True
        assert explorer.calculate_frequencies.call_count == 2

    def test_perturbs_geometry_between_attempts(self, h2_equilibrium_molecule):
        atoms = h2_equilibrium_molecule.copy()
        atoms.calc = HarmonicCalculator()
        explorer = MagicMock()
        explorer.optimizer_kwargs = {}
        explorer.ts_kwargs = {
            "cleanup_frequency_max_attempts": 2,
            "cleanup_frequency_steps": 2,
        }
        mode = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0]).reshape(6, 1).tolist()
        # Stays dirty after attempt 1 (forces a perturbation before attempt 2),
        # then becomes clean.
        explorer.calculate_frequencies.side_effect = [
            {
                "is_minimum": False,
                "minima_analysis": {"n_significant_imaginary_frequencies": 1},
                "frequencies": [-250.0],
                "normal_modes": mode,
            },
            {
                "is_minimum": False,
                "minima_analysis": {"n_significant_imaginary_frequencies": 1},
                "frequencies": [-250.0],
                "normal_modes": mode,
            },
            {
                "is_minimum": True,
                "minima_analysis": {"n_significant_imaginary_frequencies": 0},
                "frequencies": [250.0],
                "normal_modes": mode,
            },
        ]

        positions_before_attempt2 = {}
        call_count = {"n": 0}

        def fake_optimizer_factory(_atoms, **_kwargs):
            opt = MagicMock()
            opt.run.return_value = True
            opt.get_number_of_steps.return_value = 1

            def record_run(*_args, **_kwargs):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    positions_before_attempt2["pos"] = _atoms.positions.copy()
                return True

            opt.run.side_effect = record_run
            return opt

        with patch(
            "famex.strategies.frequency_cleanup._get_local_optimizer_class"
        ) as mock_get_optimizer:
            mock_get_optimizer.return_value = fake_optimizer_factory

            start = atoms.positions.copy()
            _, _, _, summary = run_frequency_cleanup(
                atoms,
                explorer,
                target="minima",
                fmax=0.05,
                temperature=298.15,
                verbose=0,
                prepare_optimizer_kwargs=lambda _name, _explorer: {},
                prepare_frequency_kwargs=None,
            )

        # Two optimization attempts ran, and a mode-following perturbation was
        # applied before the second attempt (geometry changed between them).
        assert len(summary["attempts"]) == 2
        assert summary["attempts"][0]["perturbation"] is None
        assert summary["attempts"][1]["perturbation"] == "mode"
        assert "pos" in positions_before_attempt2
        assert not np.allclose(positions_before_attempt2["pos"], start)
