"""Tests for vibrational-mode XYZ animations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read as ase_read

from famex.io.modes import write_vibrational_modes
from famex.io.xyz_io import parse_xyz_comment


def _h2() -> Atoms:
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _modes(*columns: list[float]) -> np.ndarray:
    return np.array(columns, dtype=float).T


def test_animation_scales_peak_frame_and_leaves_equilibrium_fixed(tmp_path: Path) -> None:
    atoms = _h2()
    original = atoms.positions.copy()
    written = write_vibrational_modes(
        atoms,
        {"frequencies": [1200.0], "normal_modes": _modes([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
        tmp_path,
        amplitude=0.3,
        nframes=4,
    )

    assert Path(written[0]).name == "mode_000_1200.0cm-1.xyz"
    frames = ase_read(written[0], index=":")
    assert np.allclose(frames[0].positions, original)
    assert np.allclose(atoms.positions, original)
    peak = frames[1].positions - original
    assert np.isclose(np.linalg.norm(peak, axis=1).max(), 0.3)
    assert np.allclose(peak[1], 0.0)


def test_partial_and_fixed_atoms_stay_put(tmp_path: Path) -> None:
    atoms = _h2()
    original = atoms.positions.copy()
    partial = write_vibrational_modes(
        atoms,
        {
            "frequencies": [800.0],
            "normal_modes": _modes([0.0, 0.0, 4.0]),
            "indices": [1],
        },
        tmp_path / "partial",
        amplitude=0.25,
        nframes=4,
    )
    partial_frames = ase_read(partial[0], index=":")
    assert np.allclose(partial_frames[1].positions[0], original[0])
    assert np.isclose(np.linalg.norm(partial_frames[1].positions[1] - original[1]), 0.25)

    constrained = _h2()
    constrained.set_constraint(FixAtoms(indices=[0]))
    constrained_original = constrained.positions.copy()
    fixed = write_vibrational_modes(
        constrained,
        {"frequencies": [500.0], "normal_modes": _modes([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])},
        tmp_path / "fixed",
        amplitude=0.4,
        nframes=4,
    )
    fixed_frames = ase_read(fixed[0], index=":")
    assert np.allclose(fixed_frames[1].positions[0], constrained_original[0])
    assert np.isclose(np.linalg.norm(fixed_frames[1].positions[1] - constrained_original[1]), 0.4)


def test_imaginary_selection_index_override_and_comment(tmp_path: Path) -> None:
    atoms = _h2()
    modes = np.zeros((6, 3))
    modes[0, 0] = 1.0
    modes[1, 1] = 1.0
    modes[2, 2] = 1.0
    result = {"frequencies": [-80.0, 100.0, -50.0], "normal_modes": modes}

    imaginary = write_vibrational_modes(
        atoms, result, tmp_path / "imag", which="imaginary", nframes=2
    )
    assert [Path(path).name for path in imaginary] == ["mode_000_-80.0cm-1.xyz"]

    overridden = write_vibrational_modes(
        atoms,
        result,
        tmp_path / "index",
        which="imaginary",
        mode_indices=[2],
        nframes=2,
    )
    assert Path(overridden[0]).name == "mode_002_-50.0cm-1.xyz"

    with pytest.raises(ValueError, match="No imaginary vibrational modes"):
        write_vibrational_modes(
            atoms,
            {"frequencies": [10.0, 20.0], "normal_modes": modes[:, :2]},
            tmp_path / "empty",
            which="imaginary",
        )

    written = write_vibrational_modes(
        atoms,
        {"frequencies": [-450.2], "normal_modes": _modes([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])},
        tmp_path / "comment",
        filename_prefix="image1_",
        nframes=2,
    )
    path = Path(written[0])
    assert path.name == "image1_mode_000_-450.2cm-1.xyz"
    metadata = parse_xyz_comment(path.read_text().splitlines()[1])
    assert metadata["mode"] == 0
    assert metadata["frequency"] == pytest.approx(-450.2)
