"""
Test CLI functionality with mock backend.

This module tests basic CLI functionality using the mock backend
to ensure the CLI interface works correctly without requiring
real ML potential dependencies.
"""

import os
import tempfile

import pytest
from click.testing import CliRunner

from qme.cli import main
from tests.test_utils import BackendTestMixin, TestMoleculeFactory


class TestCLIMockBackend:
    """Test CLI functionality with mock backend."""

    def test_opt_local_runs_with_mock_backend(self):
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
                    "opt",
                    xyz_path,
                    "--backend",
                    "mock",
                    "--optimizer",
                    "lbfgs",
                    "--steps",
                    "2",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(xyz_path)[0] + ".opt.xyz"
            assert os.path.exists(out_path), f"Output file not created: {out_path}"

    def test_opt_twoended_runs_with_mock_backend(self):
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
                    "opt",
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
            out_path = os.path.splitext(reactant_path)[0] + ".opt.twoended.xyz"
            assert os.path.exists(out_path), f"Output file not created: {out_path}"

    def test_tsopt_local_runs_with_real_backend(self):
        """Test transition state optimization with real backend if available."""
        # Check for available backends that can handle transition states
        available_backends = []
        for backend in ["aimnet2", "mace", "uma", "so3lr"]:
            if BackendTestMixin.check_backend_availability(backend):
                available_backends.append(backend)

        if not available_backends:
            pytest.skip("No real backends available for transition state optimization")

        # Use the first available backend
        backend = available_backends[0]

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            # Create test molecule
            atoms = TestMoleculeFactory.get_water_distorted()
            xyz_path = os.path.join(tmp, "test.xyz")
            atoms.write(xyz_path)

            # Run TS optimization
            result = runner.invoke(
                main,
                [
                    "tsopt",
                    xyz_path,
                    "--backend",
                    backend,
                    "--optimizer",
                    "sella",
                    "--steps",
                    "5",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(os.path.basename(xyz_path))[0] + ".ts.xyz"
            assert os.path.exists(
                os.path.join(tmp, out_path)
            ), f"Output file not created: {out_path}"
