import os
import tempfile

from ase import Atoms
from ase.io import write
from click.testing import CliRunner

from qme.cli import main


def _make_test_xyz(tmpdir: str, fname: str = "mol.xyz") -> str:
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    path = os.path.join(tmpdir, fname)
    write(path, h2)
    return path


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


def test_tsopt_local_runs_with_mock_backend():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        xyz = _make_test_xyz(tmp)
        result = runner.invoke(
            main,
            [
                "tsopt",
                xyz,
                "--backend",
                "mock",
                "--optimizer",
                "lbfgs",
                "--steps",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        out = os.path.splitext(os.path.basename(xyz))[0] + ".ts.xyz"
        assert os.path.exists(os.path.join(tmp, out))
