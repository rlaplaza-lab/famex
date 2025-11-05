from __future__ import annotations

import pytest
from ase.build import molecule

from qme import Explorer, calculator_registry
from qme.backends.dependencies import deps
from tests.test_utils import BackendTestMixin, StandardTestAssertions, TestResultHandler


def _torchsim_ready():
    if not deps.has("torch_sim") or not deps.has("torch"):
        return False

    try:
        import e3nn  # type: ignore
    except ImportError:
        return True

    version = e3nn.__version__.split(".")
    try:
        major, minor = int(version[0]), int(version[1])
    except (ValueError, IndexError):
        return True
    return major == 0 and minor < 5


class TestTorchSimBackendAvailability:
    @pytest.mark.parametrize("backend", ["torchsim_mace", "torchsim_uma"])
    def test_availability_queries_are_safe(self, backend):
        available = calculator_registry.is_backend_available(backend)
        assert isinstance(available, bool)


class TestTorchSimCalculatorCreation:
    def test_calculator_creation_or_clear_error(self):
        if calculator_registry.is_backend_available("torchsim_mace"):
            calc = calculator_registry.create_calculator(
                backend="torchsim_mace",
                model_name="mace-omol-0",
                device="cpu",
            )
            atoms = molecule("H2")
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            assert isinstance(energy, (float, int))
        else:
            with pytest.raises(ImportError, match="Backend 'torchsim_mace' is not available"):
                calculator_registry.create_calculator(
                    backend="torchsim_mace",
                    model_name="mace-omol-0",
                    device="cpu",
                )


class TestTorchSimOptimization:
    @pytest.mark.skipif(
        not _torchsim_ready(),
        reason="TorchSim or compatible dependencies not available",
    )
    def test_minima_smoke(self):
        BackendTestMixin.require_backend("torchsim_mace")

        atoms = molecule("C6H6")
        explorer = Explorer(
            atoms=atoms,
            backend="torchsim_mace",
            model_name="mace-omol-0",
            device="cpu",
            target="minima",
            strategy="local",
        )

        result = explorer.run(local_optimizer_name="BFGS", fmax=0.1, steps=5)
        processed = TestResultHandler.process_result(result, backend="torchsim_mace")

        StandardTestAssertions.assert_optimization_result(processed)
        StandardTestAssertions.assert_reasonable_geometry(
            processed["optimized_atoms"],
            backend="torchsim_mace",
        )
