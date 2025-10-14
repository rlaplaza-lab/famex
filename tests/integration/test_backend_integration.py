"""Lightweight backend integration smoke tests for Explorer."""

import pytest

from qme import Explorer
from qme.dependencies import deps
from tests.test_utils import (
    BackendTestRunner,
    StandardTestAssertions,
    TestMoleculeFactory,
    TestResultHandler,
)


class TestBackendMinimaIntegration:
    def test_h2_minima_across_backends(self):
        """Ensure minima optimization succeeds across available backends."""

        def _run_minima(backend: str):
            atoms = TestMoleculeFactory.get_h2_stretched()
            explorer = Explorer(
                atoms=atoms,
                backend=backend,
                target="minima",
                strategy="local",
            )
            result = explorer.run(
                mode="minima",
                local_optimizer_name="BFGS",
                fmax=0.05,
                steps=20,
            )
            processed = TestResultHandler.process_result(result, backend)
            StandardTestAssertions.assert_optimization_result(processed)
            StandardTestAssertions.assert_reasonable_geometry(processed["optimized_atoms"], backend)

            bond = processed["optimized_atoms"].get_distance(0, 1)
            assert 0.6 < bond < 1.0, f"H-H distance out of range: {bond:.3f} Å"
            return {"bond": bond, "converged": processed.get("converged", False)}

        results = BackendTestRunner.run_with_warnings(_run_minima, include_mock=True)
        BackendTestRunner.assert_backend_results(results, min_successful=1)


class TestBackendTransitionStateIntegration:
    @pytest.mark.skipif(not deps.has("sella"), reason="Sella is required for TS optimization")
    def test_water_ts_smoke(self):
        """Run a single TS optimization smoke test across available backends."""

        def _run_ts(backend: str):
            atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
            explorer = Explorer(atoms=atoms, backend=backend, target="ts", strategy="local")
            result = explorer.run(mode="ts", fmax=0.1, steps=50)
            processed = TestResultHandler.process_result(result, backend)
            StandardTestAssertions.assert_optimization_result(processed)
            StandardTestAssertions.assert_reasonable_geometry(processed["optimized_atoms"], backend)

            oh1 = processed["optimized_atoms"].get_distance(0, 1)
            oh2 = processed["optimized_atoms"].get_distance(0, 2)
            return {"oh1": oh1, "oh2": oh2}

        results = BackendTestRunner.run_with_warnings(_run_ts, include_mock=False)
        successful = [backend for backend, outcome in results.items() if outcome["success"]]

        if not successful:
            reasons = "; ".join(
                f"{backend}: {outcome['error']}" for backend, outcome in results.items()
            )
            pytest.skip(f"No TS backends succeeded: {reasons}")

        for backend in successful:
            processed = results[backend]["result"]
            assert "oh1" in processed and "oh2" in processed


class TestBackendPathIntegration:
    def test_mock_neb_smoke(self):
        """Exercise NEB workflow with the mock backend to ensure plumbing works."""

        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        product.positions[2] += (1.5, 0.0, 0.0)

        explorer = Explorer(
            atoms=reactant,
            backend="mock",
            target="path",
            strategy="neb",
        )
        explanation = explorer.explain_run("neb")
        assert explanation["valid"] is True
        assert explanation["strategy"] == "neb"
        assert explanation["strategy_type"] == "two-ended"
        assert "runner" in explanation
