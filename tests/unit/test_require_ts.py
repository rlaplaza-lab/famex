"""Tests for require_ts resolution and enforcement across TS strategies."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from famex.core.explorer import Explorer
from famex.strategies.growing_string import MultiStructureGrowingStringStrategy
from famex.strategies.helpers import (
    enforce_require_ts,
    resolve_require_ts,
    strip_require_ts_from_optimizer_kwargs,
    ts_has_one_imaginary_mode,
)
from famex.strategies.ts import LocalTSStrategy
from famex.strategies.ts_interpolate import MultiStructureTSGuessStrategy


def _water() -> Atoms:
    return Atoms("H2O", positions=[(0, 0, 0), (0.96, 0, 0), (-0.24, 0.93, 0)])


class TestRequireTsHelpers:
    def test_resolve_explicit_true_overrides_explorer(self) -> None:
        explorer = MagicMock()
        explorer.ts_kwargs = {"require_ts": False}
        assert resolve_require_ts(True, explorer) is True

    def test_resolve_none_reads_explorer(self) -> None:
        explorer = MagicMock()
        explorer.ts_kwargs = {"require_ts": True}
        assert resolve_require_ts(None, explorer) is True
        explorer.ts_kwargs = {}
        assert resolve_require_ts(None, explorer) is False

    def test_strip_require_ts_from_optimizer_kwargs(self) -> None:
        kwargs = {"require_ts": True, "trust_radius": 0.02}
        strip_require_ts_from_optimizer_kwargs(kwargs)
        assert "require_ts" not in kwargs
        assert kwargs["trust_radius"] == 0.02

    def test_ts_has_one_imaginary_mode(self) -> None:
        assert ts_has_one_imaginary_mode(None) is False
        assert ts_has_one_imaginary_mode({"is_ts": True}) is True
        assert ts_has_one_imaginary_mode({"n_imaginary_frequencies": 1}) is True
        assert ts_has_one_imaginary_mode({"n_imaginary_frequencies": 0}) is False
        assert ts_has_one_imaginary_mode({"n_imaginary_frequencies": 2}) is False
        assert ts_has_one_imaginary_mode({"is_ts": True, "n_imaginary_frequencies": 0}) is False
        assert ts_has_one_imaginary_mode({"is_ts": True, "n_imaginary_frequencies": 2}) is False
        assert ts_has_one_imaginary_mode({"ts_analysis": {"is_transition_state": True}}) is True
        assert (
            ts_has_one_imaginary_mode(
                {"is_ts": True, "ts_analysis": {"n_imaginary_frequencies": 0}}
            )
            is False
        )

    def test_enforce_raises_when_not_saddle(self) -> None:
        explorer = MagicMock()
        with pytest.raises(RuntimeError, match="require_ts=True"):
            enforce_require_ts(
                require_ts=True,
                atoms=_water(),
                explorer=explorer,
                freq_analysis={"is_ts": False, "n_imaginary_frequencies": 0},
                check_converged=True,
            )

    def test_enforce_raises_when_not_converged(self) -> None:
        with pytest.raises(RuntimeError, match="failed to converge"):
            enforce_require_ts(
                require_ts=True,
                atoms=_water(),
                explorer=MagicMock(),
                freq_analysis={"is_ts": True},
                check_converged=False,
            )

    def test_enforce_noop_when_disabled(self) -> None:
        freq = {"is_ts": False}
        assert (
            enforce_require_ts(
                require_ts=False,
                atoms=_water(),
                explorer=MagicMock(),
                freq_analysis=freq,
            )
            is freq
        )

    def test_enforce_computes_frequencies_when_missing(self) -> None:
        explorer = MagicMock()
        explorer.calculate_frequencies.return_value = {
            "is_ts": True,
            "n_imaginary_frequencies": 1,
        }
        result = enforce_require_ts(
            require_ts=True,
            atoms=_water(),
            explorer=explorer,
            freq_analysis=None,
            check_converged=True,
        )
        explorer.calculate_frequencies.assert_called_once()
        assert result["is_ts"] is True


class TestLocalTSRequireTs:
    def test_raises_when_require_ts_and_not_saddle(self) -> None:
        atoms = _water()
        explorer = Explorer(atoms, backend="mock", target="ts", strategy="local")
        strategy = LocalTSStrategy(explorer)
        fake_result: dict[str, Any] = {
            "optimized_atoms": atoms,
            "converged": True,
            "strategy": "ts:local",
            "steps_taken": 1,
            "frequency_analysis": {"is_ts": False, "n_imaginary_frequencies": 0},
        }
        with (
            patch("famex.strategies.ts._validate_ts_optimization_setup"),
            patch(
                "famex.strategies.ts._run_local_optimization_common",
                return_value=fake_result,
            ),
            pytest.raises(RuntimeError, match="require_ts=True"),
        ):
            strategy.run([atoms], require_ts=True)

    def test_does_not_raise_when_allow_ts(self) -> None:
        atoms = _water()
        explorer = Explorer(atoms, backend="mock", target="ts", strategy="local")
        strategy = LocalTSStrategy(explorer)
        fake_result: dict[str, Any] = {
            "optimized_atoms": atoms,
            "converged": True,
            "strategy": "ts:local",
            "steps_taken": 1,
            "frequency_analysis": {"is_ts": False, "n_imaginary_frequencies": 0},
        }
        with (
            patch("famex.strategies.ts._validate_ts_optimization_setup"),
            patch(
                "famex.strategies.ts._run_local_optimization_common",
                return_value=fake_result,
            ),
        ):
            result = strategy.run([atoms], require_ts=False)
        assert result["optimized_atoms"] is atoms

    def test_reads_require_ts_from_explorer_ts_kwargs(self) -> None:
        atoms = _water()
        explorer = Explorer(
            atoms,
            backend="mock",
            target="ts",
            strategy="local",
            ts_kwargs={"require_ts": True},
        )
        strategy = LocalTSStrategy(explorer)
        fake_result: dict[str, Any] = {
            "optimized_atoms": atoms,
            "converged": True,
            "strategy": "ts:local",
            "steps_taken": 1,
            "frequency_analysis": {"is_ts": False, "n_imaginary_frequencies": 2},
        }
        with (
            patch("famex.strategies.ts._validate_ts_optimization_setup"),
            patch(
                "famex.strategies.ts._run_local_optimization_common",
                return_value=fake_result,
            ),
            pytest.raises(RuntimeError, match="require_ts=True"),
        ):
            strategy.run([atoms])

    def test_require_ts_stripped_from_optimizer_kwargs(self) -> None:
        atoms = _water()
        explorer = Explorer(
            atoms,
            backend="mock",
            target="ts",
            strategy="local",
            ts_kwargs={"require_ts": True, "trust_radius": 0.02},
        )
        strategy = LocalTSStrategy(explorer)
        captured: dict[str, Any] = {}

        def fake_run(**kwargs: Any) -> dict[str, Any]:
            prepare = kwargs["prepare_optimizer_kwargs"]
            captured["opt_kwargs"] = prepare("rfo", explorer)
            return {
                "optimized_atoms": atoms,
                "converged": True,
                "strategy": "ts:local",
                "steps_taken": 1,
                "frequency_analysis": {"is_ts": True, "n_imaginary_frequencies": 1},
            }

        with (
            patch("famex.strategies.ts._validate_ts_optimization_setup"),
            patch(
                "famex.strategies.ts._run_local_optimization_common",
                side_effect=lambda **kw: fake_run(**kw),
            ),
        ):
            strategy.run([atoms], require_ts=True)

        assert "require_ts" not in captured["opt_kwargs"]
        assert captured["opt_kwargs"]["trust_radius"] == 0.02


class TestInterpolateInheritsRequireTs:
    def test_interpolate_raises_via_local_ts(self) -> None:
        reactant = _water()
        product = reactant.copy()
        product.positions[0, 0] += 0.1
        explorer = Explorer(
            [reactant, product],
            backend="mock",
            target="ts",
            strategy="interpolate",
            ts_kwargs={"require_ts": True},
        )
        strategy = MultiStructureTSGuessStrategy(explorer)
        fake_result: dict[str, Any] = {
            "optimized_atoms": reactant,
            "converged": True,
            "strategy": "ts:local",
            "steps_taken": 1,
            "frequency_analysis": {"is_ts": False, "n_imaginary_frequencies": 0},
        }

        with (
            patch("famex.strategies.ts_interpolate.PathManager") as mock_pm,
            patch(
                "famex.strategies.ts_interpolate.LocalTSStrategy",
                side_effect=lambda explorer: LocalTSStrategy(explorer),
            ),
            patch("famex.strategies.ts._validate_ts_optimization_setup"),
            patch(
                "famex.strategies.ts._run_local_optimization_common",
                return_value=fake_result,
            ),
            pytest.raises(RuntimeError, match="require_ts=True"),
        ):
            mock_pm.return_value.interpolate.return_value = [reactant, product]
            mock_pm.find_ts_guess.return_value = (reactant, 0)
            mock_pm.attach_calculators = MagicMock()
            strategy.run([reactant, product], npoints=3, steps=1)


class TestGrowingStringRequireTs:
    def test_growing_string_passes_require_ts_false_to_local_refine(self) -> None:
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])
        explorer = Explorer(
            [reactant, product],
            backend="mock",
            target="ts",
            strategy="growing_string",
            ts_kwargs={"require_ts": True},
        )
        strategy = MultiStructureGrowingStringStrategy(explorer)
        captured_kwargs: list[dict[str, Any]] = []

        def fake_local_run(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            captured_kwargs.append(dict(kwargs))
            return {
                "optimized_atoms": reactant,
                "converged": True,
                "strategy": "ts:local",
                "steps_taken": 1,
                "frequency_analysis": {"is_ts": True, "n_imaginary_frequencies": 1},
            }

        with patch.object(LocalTSStrategy, "run", side_effect=fake_local_run):
            result = strategy.run(
                [reactant, product],
                npoints=4,
                steps=2,
                optimize_endpoints=False,
                refine_ts=True,
                require_ts=True,
            )

        assert captured_kwargs
        assert all(kw.get("require_ts") is False for kw in captured_kwargs)
        assert result.get("is_ts") is True

    def test_growing_string_raises_when_not_saddle_even_if_strings_not_met(self) -> None:
        reactant = Atoms("H2", positions=[(0, 0, 0), (0.7, 0, 0)])
        product = Atoms("H2", positions=[(0, 0, 0), (1.5, 0, 0)])
        explorer = Explorer(
            [reactant, product],
            backend="mock",
            target="ts",
            strategy="growing_string",
        )
        strategy = MultiStructureGrowingStringStrategy(explorer)

        def fake_local_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "optimized_atoms": reactant,
                "converged": True,
                "strategy": "ts:local",
                "steps_taken": 1,
                "frequency_analysis": {"is_ts": False, "n_imaginary_frequencies": 0},
                "strings_met": False,
            }

        with (
            patch.object(LocalTSStrategy, "run", side_effect=fake_local_run),
            pytest.raises(RuntimeError, match="require_ts=True"),
        ):
            strategy.run(
                [reactant, product],
                npoints=4,
                steps=2,
                optimize_endpoints=False,
                refine_ts=True,
                require_ts=True,
            )
