import os
import tempfile

import pytest
from ase import Atoms
from ase.io import write
from click.testing import CliRunner

from qme.cli import main
from qme.dependencies import deps


def _make_xyz(tmpdir: str, fname: str = "mol.xyz") -> str:
    # Water molecule with mild distortion
    atoms = Atoms(
        "H2O",
        positions=[
            [0.000, 0.000, 0.000],
            [0.950, 0.000, 0.000],
            [-0.239, 0.927, 0.000],
        ],
    )
    path = os.path.join(tmpdir, fname)
    write(path, atoms)
    return path


# Reduced backend list to focus on most important ones
BACKENDS = ["mock", "aimnet2", "mace"]


def _backend_available(name: str) -> bool:
    if name == "mock":
        return True
    if name == "uma":
        return deps.has("fairchem") or deps.has("uma")
    if name in ["torchsim", "torchsim_mace", "torchsim_fairchem"]:
        return deps.has("torch_sim") and deps.has("torch")
    return deps.has(name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_minima_runs_across_backends(backend: str):
    if not _backend_available(backend):
        pytest.skip(f"Backend {backend} not available in this environment")

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        xyz = _make_xyz(tmp)
        result = runner.invoke(
            main,
            [
                "opt",
                xyz,
                "--backend",
                backend,
                "--optimizer",
                "lbfgs",
                "--steps",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output
        out = os.path.splitext(xyz)[0] + ".opt.xyz"
        assert os.path.exists(out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ts_runs_across_backends(backend: str):
    if not _backend_available(backend):
        pytest.skip(f"Backend {backend} not available in this environment")

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        xyz = _make_xyz(tmp)
        result = runner.invoke(
            main,
            [
                "tsopt",
                xyz,
                "--backend",
                backend,
                "--optimizer",
                "lbfgs",
                "--steps",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        out = os.path.splitext(os.path.basename(xyz))[0] + ".ts.xyz"
        assert os.path.exists(os.path.join(tmp, out))


@pytest.mark.parametrize("backend", BACKENDS)
def test_neb_runs_across_backends(backend: str):
    if not _backend_available(backend):
        pytest.skip(f"Backend {backend} not available in this environment")

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        # Create reactant and product structures
        reactant_path = _make_xyz(tmp, "reactant.xyz")
        product_path = _make_xyz(tmp, "product.xyz")

        # Run NEB optimization using tsopt with --mode neb
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
                "5",  # Small number for fast testing
                "--steps",
                "5",  # Very few steps for testing
                "--fmax",
                "0.1",  # Relaxed convergence for testing
                "--spring-constant",
                "1.0",  # Lower spring constant for faster testing
            ],
        )
        assert result.exit_code == 0, result.output
        out = os.path.splitext(os.path.basename(reactant_path))[0] + ".neb.xyz"
        assert os.path.exists(os.path.join(tmp, out))
