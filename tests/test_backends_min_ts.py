"""
Test backend functionality across all available backends.

This module tests basic CLI functionality (minima, TS, NEB) across
all available backends to ensure they work correctly.
"""

import os
import tempfile

import pytest
from click.testing import CliRunner

from qme.cli import main
from qme.backend_availability import get_available_backends
from tests.test_utils import TestMoleculeFactory


class TestBackendCLI:
    """Test CLI functionality across all available backends."""

    @pytest.mark.parametrize("backend", get_available_backends())
    def test_minima_optimization_cli(self, backend: str):
        """Test minima optimization via CLI across all backends."""
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
                    backend,
                    "--optimizer",
                    "lbfgs",
                    "--steps",
                    "5",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(xyz_path)[0] + ".opt.xyz"
            assert os.path.exists(out_path), f"Output file not created: {out_path}"

    @pytest.mark.parametrize("backend", get_available_backends())
    def test_transition_state_optimization_cli(self, backend: str):
        """Test transition state optimization via CLI across all backends."""
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
                    "lbfgs",
                    "--steps",
                    "3",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(os.path.basename(xyz_path))[0] + ".ts.xyz"
            assert os.path.exists(
                os.path.join(tmp, out_path)
            ), f"Output file not created: {out_path}"

    @pytest.mark.parametrize("backend", get_available_backends())
    def test_neb_optimization_cli(self, backend: str):
        """Test NEB optimization via CLI across all backends."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            # Create reactant and product structures
            reactant_atoms = TestMoleculeFactory.get_water_distorted()
            product_atoms = TestMoleculeFactory.get_water_distorted()
            # Slightly modify product
            pos = product_atoms.get_positions()
            pos[1, 0] += 0.1  # Move H atom slightly
            product_atoms.set_positions(pos)

            reactant_path = os.path.join(tmp, "reactant.xyz")
            product_path = os.path.join(tmp, "product.xyz")
            reactant_atoms.write(reactant_path)
            product_atoms.write(product_path)

            # Run NEB optimization
            result = runner.invoke(
                main,
                [
                    "tsopt",
                    reactant_path,
                    "--product",
                    product_path,
                    "--mode",
                    "neb",
                    "--backend",
                    backend,
                    "--npoints",
                    "5",
                    "--steps",
                    "5",
                    "--fmax",
                    "0.1",
                    "--spring-constant",
                    "1.0",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"CLI failed: {result.output}"
            out_path = os.path.splitext(os.path.basename(reactant_path))[0] + ".neb.xyz"
            assert os.path.exists(
                os.path.join(tmp, out_path)
            ), f"Output file not created: {out_path}"
