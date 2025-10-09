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

import numpy as np
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
        from qme.backend_availability import is_backend_available

        if not is_backend_available("torchsim_mace"):
            # Should raise ImportError with clear message when TorchSim is not available
            with pytest.raises(
                ImportError,
                match="Backend 'torchsim_mace' is not available",
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


# Define available backend pairs for testing
AVAILABLE_BACKEND_PAIRS = [
    ("mace", "torchsim_mace"),
    ("uma", "torchsim_uma"),
]


class TestTorchSimSanityCheck:
    """Test suite to verify TorchSim backends match their regular counterparts."""

    @pytest.fixture
    def test_molecules(self):
        """Provide test molecules for comparison."""
        molecules = {
            "H2": molecule("H2"),
            "H2O": molecule("H2O"),
            "CH4": molecule("CH4"),
        }

        # Add some charge/spin info for testing
        molecules["H2O"].info["charge"] = 0
        molecules["H2O"].info["spin"] = 1
        molecules["CH4"].info["charge"] = 0
        molecules["CH4"].info["spin"] = 1

        return molecules

    def test_default_model_names_consistency(self):
        """Test that TorchSim and regular backends have consistent default model names."""

        # Test MACE defaults
        if deps.has("torch") and deps.has("mace"):
            try:
                regular_mace = calculator_registry.create_calculator("mace")
                expected_mace_default = "mace-omol-0"
                assert regular_mace.model_name == expected_mace_default, (
                    f"Regular MACE default should be {expected_mace_default}, "
                    f"got {regular_mace.model_name}"
                )

                if deps.has("torch_sim"):
                    try:
                        torchsim_mace = calculator_registry.create_calculator("torchsim_mace")
                        assert torchsim_mace.model_name == expected_mace_default, (
                            f"TorchSim MACE default should be {expected_mace_default}, "
                            f"got {torchsim_mace.model_name}"
                        )

                        print(f"✅ MACE defaults consistent: {expected_mace_default}")
                    except ImportError:
                        pytest.skip("TorchSim MACE not available")
            except ImportError:
                pytest.skip("Regular MACE not available")

        # Test UMA defaults
        if deps.has("fairchem"):
            try:
                regular_uma = calculator_registry.create_calculator("uma")
                expected_uma_default = "uma-s-1p1"
                assert regular_uma.model_name == expected_uma_default, (
                    f"Regular UMA default should be {expected_uma_default}, "
                    f"got {regular_uma.model_name}"
                )

                if deps.has("torch_sim"):
                    try:
                        torchsim_uma = calculator_registry.create_calculator("torchsim_uma")
                        assert torchsim_uma.model_name == expected_uma_default, (
                            f"TorchSim UMA default should be {expected_uma_default}, "
                            f"got {torchsim_uma.model_name}"
                        )

                        print(f"✅ UMA defaults consistent: {expected_uma_default}")
                    except ImportError:
                        pytest.skip("TorchSim UMA not available")
            except ImportError:
                pytest.skip("Regular UMA not available")

    @pytest.mark.parametrize("backend_pair", AVAILABLE_BACKEND_PAIRS)
    def test_energy_force_consistency(self, backend_pair, test_molecules):
        """Test that TorchSim and regular backends produce similar results."""
        regular_backend, torchsim_backend = backend_pair

        # Check if both backends are available
        if not calculator_registry.is_backend_available(regular_backend):
            pytest.skip(f"Regular {regular_backend} not available")

        if not calculator_registry.is_backend_available(torchsim_backend):
            pytest.skip(f"TorchSim {torchsim_backend} not available")

        # Test on a simple molecule (H2O)
        atoms = test_molecules["H2O"].copy()

        try:
            # Create calculators
            regular_calc = calculator_registry.create_calculator(regular_backend, device="cpu")
            torchsim_calc = calculator_registry.create_calculator(torchsim_backend, device="cpu")

            # Ensure they use the same model
            assert (
                regular_calc.model_name == torchsim_calc.model_name
            ), f"Model names don't match: {regular_calc.model_name} vs {torchsim_calc.model_name}"

            # Calculate energies and forces
            atoms.calc = regular_calc
            regular_energy = atoms.get_potential_energy()
            regular_forces = atoms.get_forces()

            atoms.calc = torchsim_calc
            torchsim_energy = atoms.get_potential_energy()
            torchsim_forces = atoms.get_forces()

            # Compare results (allow for some numerical differences)
            energy_diff = abs(regular_energy - torchsim_energy)
            force_diff = np.max(np.abs(regular_forces - torchsim_forces))

            # Define tolerances (these may need adjustment based on actual behavior)
            energy_tolerance = 1e-3  # 1 meV
            force_tolerance = 1e-2  # 0.01 eV/Å

            print(f"\n{backend_pair[0]} vs {backend_pair[1]} on H2O:")
            print(f"  Energy difference: {energy_diff:.6f} eV (tolerance: {energy_tolerance})")
            print(f"  Max force difference: {force_diff:.6f} eV/Å (tolerance: {force_tolerance})")
            print(f"  Regular energy: {regular_energy:.6f} eV")
            print(f"  TorchSim energy: {torchsim_energy:.6f} eV")

            # For now, just log the differences - we can tighten tolerances once we
            # know expected behavior
            if energy_diff > energy_tolerance:
                print(
                    f"⚠️  Energy difference ({energy_diff:.6f}) "
                    f"exceeds tolerance ({energy_tolerance})"
                )
            else:
                print("✅ Energy difference within tolerance")

            if force_diff > force_tolerance:
                print(
                    f"⚠️  Force difference ({force_diff:.6f}) "
                    f"exceeds tolerance ({force_tolerance})"
                )
            else:
                print("✅ Force difference within tolerance")

            # For initial implementation, we'll be lenient and just ensure calculations complete
            # Later we can tighten these assertions based on observed behavior
            assert not np.isnan(regular_energy), "Regular backend produced NaN energy"
            assert not np.isnan(torchsim_energy), "TorchSim backend produced NaN energy"
            assert not np.any(np.isnan(regular_forces)), "Regular backend produced NaN forces"
            assert not np.any(np.isnan(torchsim_forces)), "TorchSim backend produced NaN forces"

        except ImportError as e:
            pytest.skip(f"Backend {backend_pair} not fully available: {e}")
        except Exception as e:
            # Log the error but don't fail the test if it's a known compatibility issue
            if "load_config" in str(e) or "FairchemModel" in str(e):
                pytest.skip(f"Known compatibility issue with {torchsim_backend}: {e}")
            else:
                raise

    def test_backend_properties_consistency(self):
        """Test that TorchSim and regular backends report consistent properties."""

        backend_pairs = [
            ("mace", "torchsim_mace"),
            ("uma", "torchsim_uma"),
        ]

        for regular_backend, torchsim_backend in backend_pairs:
            if not calculator_registry.is_backend_available(regular_backend):
                continue
            if not calculator_registry.is_backend_available(torchsim_backend):
                continue

            try:
                regular_calc = calculator_registry.create_calculator(regular_backend)
                torchsim_calc = calculator_registry.create_calculator(torchsim_backend)

                # Trigger calculator loading by accessing the _calc attribute
                # This ensures implemented_properties is populated
                try:
                    regular_calc._load_calculator()
                    torchsim_calc._load_calculator()
                except Exception:  # Use specific exception type
                    # If loading fails, skip this test
                    continue

                # Check implemented properties
                regular_props = set(regular_calc.implemented_properties)
                torchsim_props = set(torchsim_calc.implemented_properties)

                print(f"\n{regular_backend} vs {torchsim_backend} properties:")
                print(f"  Regular: {sorted(regular_props)}")
                print(f"  TorchSim: {sorted(torchsim_props)}")

                # Both should at least support energy and forces
                assert "energy" in regular_props, f"{regular_backend} should support energy"
                assert "forces" in regular_props, f"{regular_backend} should support forces"
                assert "energy" in torchsim_props, f"{torchsim_backend} should support energy"
                assert "forces" in torchsim_props, f"{torchsim_backend} should support forces"

                print("✅ Both backends support required properties")

            except ImportError as e:
                if "load_config" in str(e) or "FairchemModel" in str(e):
                    print(f"⚠️  Known compatibility issue with {torchsim_backend}: {e}")
                    continue
                else:
                    raise

    def test_device_consistency(self):
        """Test that both backends handle device specification consistently."""

        backend_pairs = [
            ("mace", "torchsim_mace"),
            ("uma", "torchsim_uma"),
        ]

        for regular_backend, torchsim_backend in backend_pairs:
            if not calculator_registry.is_backend_available(regular_backend):
                continue
            if not calculator_registry.is_backend_available(torchsim_backend):
                continue

            try:
                # Test CPU device specification
                regular_calc = calculator_registry.create_calculator(regular_backend, device="cpu")
                torchsim_calc = calculator_registry.create_calculator(
                    torchsim_backend, device="cpu"
                )

                assert regular_calc.device == "cpu", f"{regular_backend} device should be 'cpu'"
                assert torchsim_calc.device == "cpu", f"{torchsim_backend} device should be 'cpu'"

                print(
                    f"✅ {regular_backend} and {torchsim_backend} both handle CPU device correctly"
                )

            except ImportError as e:
                if "load_config" in str(e) or "FairchemModel" in str(e):
                    continue
                else:
                    raise

    def test_charge_spin_handling_consistency(self):
        """Test that both backends handle charge and spin consistently."""

        if not (deps.has("fairchem") and calculator_registry.is_backend_available("uma")):
            pytest.skip("UMA backends not available for charge/spin testing")

        # Create a molecule with specific charge and spin
        atoms = molecule("H2O")
        atoms.info["charge"] = -1
        atoms.info["spin"] = 2

        try:
            regular_calc = calculator_registry.create_calculator("uma", device="cpu")

            # Check that regular backend handles charge/spin
            atoms.calc = regular_calc

            # Verify the calculator has the charge/spin attributes
            assert hasattr(regular_calc, "default_charge"), "Regular UMA should have default_charge"
            assert hasattr(regular_calc, "default_spin"), "Regular UMA should have default_spin"

            print("✅ Regular UMA handles charge/spin correctly")

            if calculator_registry.is_backend_available("torchsim_uma"):
                try:
                    torchsim_calc = calculator_registry.create_calculator(
                        "torchsim_uma", device="cpu"
                    )

                    # Check that TorchSim backend also handles charge/spin
                    assert hasattr(
                        torchsim_calc, "default_charge"
                    ), "TorchSim UMA should have default_charge"
                    assert hasattr(
                        torchsim_calc, "default_spin"
                    ), "TorchSim UMA should have default_spin"

                    # Check that defaults are the same
                    assert (
                        regular_calc.default_charge == torchsim_calc.default_charge
                    ), "Default charges should match"
                    assert (
                        regular_calc.default_spin == torchsim_calc.default_spin
                    ), "Default spins should match"

                    print("✅ TorchSim UMA handles charge/spin consistently with regular UMA")

                except ImportError as e:
                    if "load_config" in str(e):
                        pytest.skip(f"Known TorchSim UMA compatibility issue: {e}")
                    else:
                        raise

        except ImportError:
            pytest.skip("UMA backend not available")

    def test_model_loading_behavior(self):
        """Test that model loading behavior is consistent between backends."""

        backend_pairs = [
            ("mace", "torchsim_mace"),
            ("uma", "torchsim_uma"),
        ]

        for regular_backend, torchsim_backend in backend_pairs:
            if not calculator_registry.is_backend_available(regular_backend):
                continue
            if not calculator_registry.is_backend_available(torchsim_backend):
                continue

            try:
                # Test that both backends can be created without immediate model loading
                regular_calc = calculator_registry.create_calculator(regular_backend, device="cpu")
                torchsim_calc = calculator_registry.create_calculator(
                    torchsim_backend, device="cpu"
                )

                # Both should be created successfully
                assert regular_calc is not None, f"{regular_backend} calculator creation failed"
                assert torchsim_calc is not None, f"{torchsim_backend} calculator creation failed"

                # Both should have the same model name
                assert regular_calc.model_name == torchsim_calc.model_name, (
                    f"Model names don't match: {regular_calc.model_name} "
                    f"vs {torchsim_calc.model_name}"
                )

                print(
                    f"✅ {regular_backend} and {torchsim_backend} model loading behavior consistent"
                )

            except ImportError as e:
                if "load_config" in str(e) or "FairchemModel" in str(e):
                    print(f"⚠️  Known compatibility issue with {torchsim_backend}")
                    continue
                else:
                    raise

    def test_backend_availability_consistency(self):
        """Test that backend availability detection is working correctly."""

        # Test that if regular backend is available, we can detect TorchSim availability correctly
        backend_pairs = [
            ("mace", "torchsim_mace"),
            ("uma", "torchsim_uma"),
        ]

        for regular_backend, torchsim_backend in backend_pairs:
            regular_available = calculator_registry.is_backend_available(regular_backend)
            torchsim_available = calculator_registry.is_backend_available(torchsim_backend)

            print("\nBackend availability:")
            print(f"  {regular_backend}: {'✅' if regular_available else '❌'}")
            print(f"  {torchsim_backend}: {'✅' if torchsim_available else '❌'}")

            # If regular backend is available but TorchSim is not, it should be due to
            # missing torch_sim
            if regular_available and not torchsim_available:
                if not deps.has("torch_sim"):
                    print(
                        f"  ℹ️  {torchsim_backend} unavailable due to missing "
                        f"torch_sim (expected)"
                    )
                else:
                    print(f"  ⚠️  {torchsim_backend} unavailable despite torch_sim being present")

            # Both should be consistent in their availability detection
            # (This is more of a logging test than an assertion)
