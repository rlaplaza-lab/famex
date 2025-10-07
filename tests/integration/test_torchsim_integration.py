"""
Tests for TorchSim integration in QME.

This module tests the TorchSim integration, including:
- TorchSim backend availability detection
- Calculator creation and fallback behavior
- Basic energy and force calculations
- Integration with QME optimization workflows
"""

import tempfile
from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule

from qme import Explorer, calculator_registry
from qme.backend_availability import get_available_backends
from qme.dependencies import deps
from tests.test_utils import BackendTestMixin


def _has_e3nn_compatibility_issue():
    """Check if there's an e3nn compatibility issue that affects TorchSim."""
    try:
        import e3nn

        # e3nn 0.5+ has compatibility issues with MACE models
        version_parts = e3nn.__version__.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])
        return major > 0 or (major == 0 and minor >= 5)
    except ImportError:
        return False


class TestTorchSimIntegration:
    """Test TorchSim integration functionality."""

    def test_torchsim_availability(self):
        """Test TorchSim availability detection."""
        has_torchsim = deps.has("torch_sim")
        has_torch = deps.has("torch")

        # TorchSim requires both torch_sim and torch
        if has_torchsim:
            assert has_torch, "TorchSim requires PyTorch"

        print(f"TorchSim available: {has_torchsim}")
        print(f"PyTorch available: {has_torch}")

    def test_torchsim_backend_availability(self):
        """Test TorchSim backend availability."""
        backends = ["torchsim_mace", "torchsim_uma"]

        for backend in backends:
            available = calculator_registry.is_backend_available(backend)
            print(f"{backend}: {available}")

            # Note: We don't assert specific availability expectations here because
            # backends may be unavailable due to compatibility issues (e.g., e3nn version conflicts)
            # The important thing is that the availability check doesn't crash
            assert isinstance(available, bool), f"Backend {backend} availability should be boolean"

    def test_torchsim_calculator_creation(self):
        """Test TorchSim calculator creation."""
        BackendTestMixin.require_backend(
            "torchsim_mace"
        )  # This will skip if TorchSim is not available

        # Test TorchSim MACE calculator
        calc_mace = calculator_registry.create_calculator(
            backend="torchsim_mace", model_name="mace-omol-0", device="cpu"
        )
        assert calc_mace is not None
        assert hasattr(calc_mace, "backend")

        # Test TorchSim UMA calculator (if fairchem available)
        if deps.has("fairchem"):
            calc_uma = calculator_registry.create_calculator(
                backend="torchsim_uma",
                model_name="uma-s-1p1",
                device="cpu",
            )
            assert calc_uma is not None

    def test_torchsim_fallback_behavior(self):
        """Test TorchSim error behavior when not available."""
        if not deps.has("torch_sim"):
            # Should raise ImportError with clear message when TorchSim is not available
            with pytest.raises(
                ImportError,
                match="TorchSim MACE calculator requires torch-sim-atomistic",
            ):
                calculator_registry.create_calculator(
                    backend="torchsim_mace", model_name="mace-omol-0", device="cpu"
                )
        else:
            # If TorchSim is available, should create a real calculator
            calc = calculator_registry.create_calculator(
                backend="torchsim_mace", model_name="mace-omol-0", device="cpu"
            )
            assert calc is not None
            # Should be a real TorchSim calculator, not mock
            assert "mock" not in str(type(calc)).lower()

    @pytest.mark.skipif(
        _has_e3nn_compatibility_issue(), reason="e3nn version incompatibility with MACE models"
    )
    def test_torchsim_energy_calculation(self):
        """Test TorchSim energy calculation."""
        BackendTestMixin.require_backend("torchsim_mace")

        # Create a simple molecule
        benzene = molecule("C6H6")

        # Create TorchSim calculator
        calc = calculator_registry.create_calculator(
            backend="torchsim_mace", model_name="mace-omol-0", device="cpu"
        )

        # Attach calculator to atoms
        benzene.calc = calc

        # Test energy calculation
        energy = benzene.get_potential_energy()
        assert isinstance(energy, (int, float))
        print(f"TorchSim energy: {energy:.6f} eV")

    @pytest.mark.skipif(
        _has_e3nn_compatibility_issue(), reason="e3nn version incompatibility with MACE models"
    )
    def test_torchsim_forces_calculation(self):
        """Test TorchSim forces calculation."""
        BackendTestMixin.require_backend("torchsim_mace")

        # Create a simple molecule
        benzene = molecule("C6H6")

        # Create TorchSim calculator
        calc = calculator_registry.create_calculator(
            backend="torchsim_mace", model_name="mace-omol-0", device="cpu"
        )

        # Attach calculator to atoms
        benzene.calc = calc

        # Test forces calculation
        forces = benzene.get_forces()
        assert forces.shape == (len(benzene), 3)
        print(f"TorchSim forces shape: {forces.shape}")
        print(f"Max force: {forces.max():.6f} eV/Å")

    @pytest.mark.skipif(
        _has_e3nn_compatibility_issue(), reason="e3nn version incompatibility with MACE models"
    )
    def test_torchsim_optimization(self):
        """Test TorchSim optimization workflow."""
        BackendTestMixin.require_backend("torchsim_mace")

        # Create a simple molecule with slight distortion
        benzene = molecule("C6H6")
        # Add some distortion to make optimization non-trivial
        pos = benzene.get_positions()
        pos[0, 0] += 0.1  # Move first atom slightly
        benzene.set_positions(pos)

        # Create QME optimizer with TorchSim
        qme_opt = Explorer(
            atoms=benzene,
            backend="torchsim_mace",
            model_name="mace-omol-0",
            device="cpu",
        )

        # Optimize (structure already loaded in constructor)
        result = qme_opt.run(
            mode="minima",
            local_optimizer_name="BFGS",
            fmax=0.05,
            steps=10,  # Small number for testing
        )

        # Handle list return format from run() method
        if isinstance(result, list) and len(result) > 0:
            strategy_result = result[0]
        else:
            strategy_result = result

        assert strategy_result is not None
        assert "converged" in strategy_result
        print(f"TorchSim optimization converged: {strategy_result['converged']}")

    @pytest.mark.skipif(
        _has_e3nn_compatibility_issue(), reason="e3nn version incompatibility with MACE models"
    )
    def test_torchsim_cli_integration(self):
        """Test TorchSim CLI integration."""
        BackendTestMixin.require_backend("torchsim_mace")

        from click.testing import CliRunner

        from qme.cli import main

        # Create a simple molecule
        benzene = molecule("C6H6")

        with tempfile.TemporaryDirectory() as tmp:
            # Save molecule to file
            xyz_path = Path(tmp) / "benzene.xyz"
            benzene.write(str(xyz_path))

            # Test CLI with TorchSim backend
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "opt",
                    str(xyz_path),
                    "--backend",
                    "torchsim_mace",
                    "--model-name",
                    "mace-omol-0",
                    "--device",
                    "cpu",
                    "--steps",
                    "5",  # Small number for testing
                    "--fmax",
                    "0.1",  # Relaxed convergence for testing
                ],
            )

            assert result.exit_code == 0, f"CLI failed: {result.output}"

            # Check output file was created
            output_path = xyz_path.with_suffix(".opt.xyz")
            assert output_path.exists()

    @pytest.mark.skipif(
        _has_e3nn_compatibility_issue(), reason="e3nn version incompatibility with MACE models"
    )
    def test_torchsim_performance_comparison(self):
        """Test TorchSim performance compared to standard backends."""
        BackendTestMixin.require_backend("torchsim_mace")

        import time

        # Create a simple molecule
        benzene = molecule("C6H6")

        # Test standard MACE (if available)
        if calculator_registry.is_backend_available("mace"):
            calc_standard = calculator_registry.create_calculator(
                backend="mace", model_name="mace-omol-0", device="cpu"
            )
            benzene.calc = calc_standard

            start_time = time.time()
            energy_standard = benzene.get_potential_energy()
            time_standard = time.time() - start_time

            print(f"Standard MACE energy: {energy_standard:.6f} eV")
            print(f"Standard MACE time: {time_standard:.4f} s")
        else:
            print("Standard MACE not available - skipping comparison")

        # Test TorchSim MACE
        calc_torchsim = calculator_registry.create_calculator(
            backend="torchsim_mace", model_name="mace-omol-0", device="cpu"
        )
        benzene.calc = calc_torchsim

        start_time = time.time()
        energy_torchsim = benzene.get_potential_energy()
        time_torchsim = time.time() - start_time

        print(f"TorchSim MACE energy: {energy_torchsim:.6f} eV")
        print(f"TorchSim MACE time: {time_torchsim:.4f} s")

        # Energies should be similar (within reasonable tolerance)
        if deps.has("mace"):
            energy_diff = abs(energy_standard - energy_torchsim)
            print(f"Energy difference: {energy_diff:.6f} eV")
            # Allow some tolerance due to different implementations
            assert energy_diff < 0.1, f"Energy difference too large: {energy_diff} eV"


if __name__ == "__main__":
    # Run tests directly for debugging
    test = TestTorchSimIntegration()

    print("Testing TorchSim Integration...")
    print("=" * 50)

    try:
        test.test_torchsim_availability()
        test.test_torchsim_backend_availability()
        test.test_torchsim_calculator_creation()
        test.test_torchsim_fallback_behavior()

        if deps.has("torch_sim"):
            test.test_torchsim_energy_calculation()
            test.test_torchsim_forces_calculation()
            test.test_torchsim_optimization()
            test.test_torchsim_cli_integration()
            test.test_torchsim_performance_comparison()

        print("\n✅ All TorchSim integration tests passed!")

    except Exception as e:
        print(f"\n❌ TorchSim integration test failed: {e}")
        raise
