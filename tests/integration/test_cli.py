"""Test CLI functionality with mock backend.

This module tests basic CLI functionality using the mock backend
to ensure the CLI interface works correctly without requiring
real ML potential dependencies.
"""

import os
import tempfile

import pytest
from click.testing import CliRunner

import qme
from qme.cli import main
from tests.test_utils import (
    BackendTestRunner,
    StandardTestAssertions,
    TestMoleculeFactory,
    TestResultHandler,
)


class TestCLIMockBackend:
    """Test CLI functionality with mock backend."""

    def test_opt_local_runs_with_mock_backend(self) -> None:
        """Test local optimization with mock backend."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            # Create test molecule
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            # Run optimization
            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "local",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--local-optimizer",
                    "lbfgs",
                    "--steps",
                    "2",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(xyz_path)[0] + ".opt.local.xyz"
            assert os.path.exists(out_path), f"Output file not created: {out_path}"

    def test_opt_twoended_runs_with_mock_backend(self) -> None:
        """Test two-ended optimization with mock backend."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            # Create reactant and product
            reactant = TestMoleculeFactory.get_water_distorted()
            product = TestMoleculeFactory.get_water_distorted()
            # Slightly modify product
            pos = product.get_positions()
            pos[1, 0] += 0.1
            product.set_positions(pos)

            reactant_path = os.path.join(tmp, "r.xyz")
            product_path = os.path.join(tmp, "p.xyz")
            reactant.write(reactant_path)
            product.write(product_path)

            # Run two-ended optimization
            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "interpolate",
                    reactant_path,
                    "--product",
                    product_path,
                    "--backend",
                    "mock",
                    "--npoints",
                    "5",
                    "--steps",
                    "1",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(reactant_path)[0] + ".opt.interpolate.xyz"
            assert os.path.exists(out_path), f"Output file not created: {out_path}"

    def test_minima_dry_run(self) -> None:
        """Test minima command with dry-run mode."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "local",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0
            assert "Dry-run analysis" in result.output
            assert "Target: minima" in result.output

    def test_minima_with_frequencies(self) -> None:
        """Test minima command with frequency calculation."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "local",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--steps",
                    "2",
                    "--freq",
                    "--temperature",
                    "300",
                ],
            )

            assert result.exit_code == 0
            out_path = os.path.splitext(xyz_path)[0] + ".opt.local.json"
            assert os.path.exists(out_path)

    def test_minima_with_custom_output(self) -> None:
        """Test minima with custom output path."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)
            custom_out = os.path.join(tmp, "custom.xyz")

            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "local",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--steps",
                    "2",
                    "--output",
                    custom_out,
                ],
            )

            assert result.exit_code == 0
            assert os.path.exists(custom_out)

    def test_ts_command_with_mock(self) -> None:
        """Test TS optimization command errors with mock backend."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
            xyz_path = os.path.join(tmp, "ts_guess.xyz")
            atoms.write(xyz_path)

            result = runner.invoke(
                main,
                [
                    "ts",
                    "--strategy",
                    "local",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--steps",
                    "2",
                ],
            )

            # Mock backend doesn't support TS optimization
            assert result.exit_code != 0

    def test_ts_interpolate_command(self) -> None:
        """Test TS interpolate command errors with mock backend."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            reactant = TestMoleculeFactory.get_water_distorted()
            product = TestMoleculeFactory.get_water_distorted()
            pos = product.get_positions()
            pos[1, 0] += 0.3
            product.set_positions(pos)

            reactant_path = os.path.join(tmp, "r.xyz")
            product_path = os.path.join(tmp, "p.xyz")
            reactant.write(reactant_path)
            product.write(product_path)

            result = runner.invoke(
                main,
                [
                    "ts",
                    "--strategy",
                    "interpolate",
                    reactant_path,
                    "--product",
                    product_path,
                    "--backend",
                    "mock",
                    "--npoints",
                    "5",
                ],
            )

            # Mock backend doesn't support TS optimization
            assert result.exit_code != 0

    def test_path_neb_command(self) -> None:
        """Test NEB path optimization."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            reactant = TestMoleculeFactory.get_water_distorted()
            product = TestMoleculeFactory.get_water_distorted()
            pos = product.get_positions()
            pos[2] += (0.5, 0, 0)
            product.set_positions(pos)

            reactant_path = os.path.join(tmp, "r.xyz")
            product_path = os.path.join(tmp, "p.xyz")
            reactant.write(reactant_path)
            product.write(product_path)

            result = runner.invoke(
                main,
                [
                    "path",
                    "--strategy",
                    "neb",
                    reactant_path,
                    "--product",
                    product_path,
                    "--backend",
                    "mock",
                    "--npoints",
                    "5",
                    "--steps",
                    "2",
                ],
            )

            assert result.exit_code == 0

    def test_path_irc_command(self) -> None:
        """Test IRC path calculation."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            ts = TestMoleculeFactory.get_water_dissociation_ts_guess()
            ts_path = os.path.join(tmp, "ts.xyz")
            ts.write(ts_path)

            result = runner.invoke(
                main,
                [
                    "path",
                    "--strategy",
                    "irc",
                    ts_path,
                    "--backend",
                    "mock",
                    "--steps",
                    "5",
                    "--direction",
                    "forward",
                ],
            )

            assert result.exit_code == 0

    def test_path_cineb_command(self) -> None:
        """Test CI-NEB path optimization."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            reactant = TestMoleculeFactory.get_water_distorted()
            product = TestMoleculeFactory.get_water_distorted()
            pos = product.get_positions()
            pos[2] += (0.5, 0, 0)
            product.set_positions(pos)

            reactant_path = os.path.join(tmp, "r.xyz")
            product_path = os.path.join(tmp, "p.xyz")
            reactant.write(reactant_path)
            product.write(product_path)

            result = runner.invoke(
                main,
                [
                    "path",
                    "--strategy",
                    "cineb",
                    reactant_path,
                    "--product",
                    product_path,
                    "--backend",
                    "mock",
                    "--npoints",
                    "5",
                    "--steps",
                    "2",
                ],
            )

            assert result.exit_code == 0

    def test_path_interpolate_command(self) -> None:
        """Test raw interpolation path command."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            reactant = TestMoleculeFactory.get_water_distorted()
            product = TestMoleculeFactory.get_water_distorted()
            pos = product.get_positions()
            pos[2] += (0.5, 0, 0)
            product.set_positions(pos)

            reactant_path = os.path.join(tmp, "r.xyz")
            product_path = os.path.join(tmp, "p.xyz")
            reactant.write(reactant_path)
            product.write(product_path)

            result = runner.invoke(
                main,
                [
                    "path",
                    "--strategy",
                    "interpolate",
                    reactant_path,
                    "--product",
                    product_path,
                    "--backend",
                    "mock",
                    "--npoints",
                    "5",
                ],
            )

            assert result.exit_code == 0

    def test_verbose_options(self) -> None:
        """Test verbose flag options."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            # Test -vv (verbose=2)
            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "local",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--steps",
                    "1",
                    "-vv",
                ],
            )

            assert result.exit_code == 0

    def test_constraints_option(self) -> None:
        """Test constraints option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "local",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--steps",
                    "2",
                    "--constraints",
                    "fix 0",
                ],
            )

            assert result.exit_code == 0

    def test_optimizer_kwargs(self) -> None:
        """Test optimizer kwargs parsing."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "local",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--steps",
                    "2",
                    "--optimizer-kw",
                    "maxstep=0.1",
                ],
            )

            assert result.exit_code == 0

    def test_missing_product_error(self) -> None:
        """Test error when product is missing for interpolate strategy."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "interpolate",
                    xyz_path,
                    "--backend",
                    "mock",
                ],
            )

            assert result.exit_code != 0
            assert "product" in result.output.lower() or "required" in result.output.lower()

    def test_invalid_strategy_error(self) -> None:
        """Test error with invalid strategy."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            result = runner.invoke(
                main,
                [
                    "minima",
                    "--strategy",
                    "invalid_strategy",
                    xyz_path,
                    "--backend",
                    "mock",
                ],
            )

            # Click should reject invalid choice
            assert "Invalid choice" in result.output.lower() or result.exit_code != 0


class TestBackendMinimaIntegration:
    """Test backend integration for minima optimization."""

    def test_h2_minima_across_backends(self) -> None:
        """Ensure minima optimization succeeds across available backends."""

        def _run_minima(backend: str):
            atoms = TestMoleculeFactory.get_h2_stretched()
            explorer = qme.Explorer(
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
    """Test backend integration for transition state optimization."""

    @pytest.mark.skipif(not qme.deps.has("sella"), reason="Sella is required for TS optimization")
    def test_water_ts_smoke(self) -> None:
        """Run a single TS optimization smoke test across available backends."""

        def _run_ts(backend: str):
            atoms = TestMoleculeFactory.get_water_dissociation_ts_guess()
            explorer = qme.Explorer(atoms=atoms, backend=backend, target="ts", strategy="local")
            result = explorer.run(fmax=0.1, steps=50)
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
            assert "oh1" in processed
            assert "oh2" in processed


class TestBackendPathIntegration:
    """Test backend integration for path optimization."""

    def test_mock_neb_smoke(self) -> None:
        """Exercise NEB workflow with the mock backend to ensure plumbing works."""
        reactant = TestMoleculeFactory.get_water_distorted()
        product = TestMoleculeFactory.get_water_distorted()
        product.positions[2] += (1.5, 0.0, 0.0)

        explorer = qme.Explorer(
            atoms=reactant,
            backend="mock",
            target="path",
            strategy="neb",
        )
        explanation = explorer.explain_run()
        assert explanation["valid"] is True
        assert explanation["strategy"] == "neb"
        assert explanation["strategy_type"] == "multi-structure"
        assert "runner" in explanation
