import os
import tempfile

import pytest
from ase import Atoms
from ase.io import write
from click.testing import CliRunner

from qme.cli import main
from qme.dependencies import deps


def _make_test_xyz(tmpdir: str, fname: str = "mol.xyz") -> str:
    # Use H2O instead of H2 for more meaningful testing
    h2o = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
    path = os.path.join(tmpdir, fname)
    write(path, h2o)
    return path


def _backend_available(name: str) -> bool:
    """Check if a backend is truly available for testing (can create real calculators, not mock fallbacks)."""
    if name == "mock":
        return True
    if name == "aimnet2":
        # Check if AIMNet2 can actually create a real calculator
        if not deps.has("aimnet2"):
            return False
        try:
            from qme.potentials.aimnet2_potential import AIMNet2Potential

            return True
        except ImportError:
            return False
    if name == "mace":
        # Check if MACE can actually create a real calculator
        if not deps.has("mace"):
            return False
        try:
            from qme.potentials.mace_potential import MACEPotential

            return True
        except ImportError:
            return False
    if name == "uma":
        # Check if UMA can actually create a real calculator
        if not deps.has("fairchem"):
            return False
        try:
            from qme.potentials.uma_potential import UMAPotential

            return True
        except ImportError:
            return False
    if name in ["torchsim_mace", "torchsim_uma"]:
        return deps.has("torch_sim") and deps.has("torch")
    return deps.has(name)


def test_opt_local_runs_with_mock_backend():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        xyz = _make_test_xyz(tmp)
        result = runner.invoke(
            main,
            [
                "opt",
                xyz,
                "--backend",
                "mock",
                "--optimizer",
                "lbfgs",
                "--steps",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        out = os.path.splitext(xyz)[0] + ".opt.xyz"
        assert os.path.exists(out)


def test_opt_twoended_runs_with_mock_backend():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_test_xyz(tmp, "r.xyz")
        p = _make_test_xyz(tmp, "p.xyz")
        result = runner.invoke(
            main,
            [
                "opt",
                r,
                "--product",
                p,
                "--backend",
                "mock",
                "--npoints",
                "5",
                "--steps",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        out = os.path.splitext(r)[0] + ".opt.twoended.xyz"
        assert os.path.exists(out)


def test_tsopt_local_runs():
    """Test transition state optimization with real backend or skip if none available."""
    # Check for available backends that can handle transition states
    available_backends = []
    for backend in ["aimnet2", "mace", "uma", "so3lr"]:
        if _backend_available(backend):
            available_backends.append(backend)

    if not available_backends:
        pytest.skip("No real backends available for transition state optimization")

    # Use the first available backend
    backend = available_backends[0]

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        xyz = _make_test_xyz(tmp)
        result = runner.invoke(
            main,
            [
                "tsopt",
                xyz,
                "--backend",
                backend,
                "--optimizer",
                "sella",  # Use sella for TS optimization
                "--steps",
                "5",  # More steps for real optimization
            ],
        )
        assert result.exit_code == 0, result.output
        out = os.path.splitext(os.path.basename(xyz))[0] + ".ts.xyz"
        assert os.path.exists(os.path.join(tmp, out))
