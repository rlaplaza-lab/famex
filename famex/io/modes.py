"""Write vibrational normal modes as multi-frame XYZ animations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from famex.io.xyz_io import write_xyz_with_metadata

# Same cutoff frequency cleanup uses when it decides a mode is imaginary.
IMAGINARY_MODE_THRESHOLD_CM = 50.0

__all__ = [
    "modes_output_dir",
    "write_vibrational_modes",
]


def modes_output_dir(output_path: str | Path) -> Path:
    """Return the mode-animation directory beside a run geometry file.

    ``water.minima.xyz`` becomes ``water.minima_modes``.
    """
    path = Path(output_path)
    return path.with_name(f"{path.stem}_modes")


def write_vibrational_modes(
    atoms: Atoms,
    frequency_result: dict[str, Any],
    output_dir: str | Path,
    *,
    which: str = "all",
    mode_indices: list[int] | None = None,
    amplitude: float = 0.3,
    nframes: int = 20,
    filename_prefix: str = "",
) -> list[str]:
    """Write one multi-frame XYZ animation per selected vibrational mode.

    Frame ``k`` is the equilibrium geometry plus
    ``amplitude * sin(2π k / nframes)`` times the mode direction. The mode is
    scaled so its largest atomic displacement equals ``amplitude`` in Å.
    Frame 0 is the equilibrium geometry. Real and imaginary modes use the
    same oscillation; the frequency is only a label in the filename and the
    XYZ comment.

    Parameters
    ----------
    atoms : Atoms
        Equilibrium structure.
    frequency_result : dict
        Frequency analysis containing ``frequencies`` (cm^-1) and Cartesian
        ``normal_modes`` with one column per mode.
    output_dir : str or Path
        Directory that receives ``mode_III_FREQ.xyz`` files.
    which : {"all", "imaginary"}
        Modes to write when ``mode_indices`` is omitted. ``imaginary`` keeps
        frequencies below -50 cm^-1.
    mode_indices : list of int, optional
        Explicit vibrational-mode columns. Overrides ``which``.
    amplitude : float
        Largest atomic displacement along the mode, in Å.
    nframes : int
        Frames in one oscillation. Must be at least 2.
    filename_prefix : str
        Prefix for each filename, such as ``image0_`` for a multi-image run.

    Returns
    -------
    list of str
        Paths written, in mode order.

    """
    if amplitude <= 0:
        msg = "amplitude must be positive"
        raise ValueError(msg)
    if nframes < 2:
        msg = "nframes must be at least 2"
        raise ValueError(msg)
    if which not in ("all", "imaginary"):
        msg = f"which must be 'all' or 'imaginary', got {which!r}"
        raise ValueError(msg)

    frequencies, normal_modes = _frequencies_and_modes(frequency_result)
    selected = _select_mode_indices(frequencies, which, mode_indices)
    atom_indices = _mode_atom_indices(len(atoms), normal_modes, frequency_result)
    if atom_indices is None:
        msg = (
            f"Normal mode length {normal_modes.shape[0]} does not match "
            f"{len(atoms)} atoms or the frequency result indices"
        )
        raise ValueError(msg)

    destination = Path(output_dir)
    written: list[str] = []
    for column in selected:
        frequency = float(frequencies[column])
        direction = _scaled_mode(
            normal_modes[:, column], atom_indices, len(atoms), amplitude, column
        )
        frames = _oscillate(atoms, direction, nframes, column, frequency)
        filename = destination / (f"{filename_prefix}mode_{column:03d}_{frequency:.1f}cm-1.xyz")
        write_xyz_with_metadata(frames, filename)
        written.append(str(filename))
    return written


def _frequencies_and_modes(
    frequency_result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if "frequencies" not in frequency_result or "normal_modes" not in frequency_result:
        msg = "frequency_result must contain 'frequencies' and 'normal_modes'"
        raise ValueError(msg)

    frequencies = np.asarray(frequency_result["frequencies"], dtype=float).reshape(-1)
    normal_modes = np.asarray(frequency_result["normal_modes"], dtype=float)
    if normal_modes.ndim != 2 or normal_modes.shape[1] != frequencies.shape[0]:
        msg = (
            "normal_modes must have one column per frequency "
            f"(got frequencies {frequencies.shape} and modes {normal_modes.shape})"
        )
        raise ValueError(msg)
    return frequencies, normal_modes


def _select_mode_indices(
    frequencies: np.ndarray,
    which: str,
    mode_indices: list[int] | None,
) -> list[int]:
    n_modes = int(frequencies.shape[0])
    if mode_indices is not None:
        selected = [int(index) for index in mode_indices]
        if not selected:
            msg = "No vibrational modes selected"
            raise ValueError(msg)
        for index in selected:
            if index < 0 or index >= n_modes:
                msg = f"Mode index {index} is outside 0..{n_modes - 1}"
                raise ValueError(msg)
        return selected

    if which == "imaginary":
        selected = [
            int(index)
            for index, frequency in enumerate(frequencies)
            if frequency < -IMAGINARY_MODE_THRESHOLD_CM
        ]
    else:
        selected = list(range(n_modes))

    if not selected:
        if which == "imaginary":
            msg = f"No imaginary vibrational modes below -{IMAGINARY_MODE_THRESHOLD_CM:g} cm^-1"
        else:
            msg = "No vibrational modes selected"
        raise ValueError(msg)
    return selected


def _mode_atom_indices(
    n_atoms: int,
    normal_modes: np.ndarray,
    frequency_result: dict[str, Any],
) -> list[int] | None:
    """Map normal-mode rows to atom indices, including partial Hessian analyses."""
    if normal_modes.shape[0] == 3 * n_atoms:
        return list(range(n_atoms))

    indices = frequency_result.get("indices")
    if indices is not None and normal_modes.shape[0] == 3 * len(indices):
        return [int(index) for index in indices]
    return None


def _scaled_mode(
    mode_column: np.ndarray,
    atom_indices: list[int],
    n_atoms: int,
    amplitude: float,
    mode_index: int,
) -> np.ndarray:
    """Return an (n_atoms, 3) displacement whose largest atom moves by ``amplitude``."""
    local = np.asarray(mode_column, dtype=float).reshape(len(atom_indices), 3)
    max_length = float(np.linalg.norm(local, axis=1).max())
    if max_length == 0.0:
        msg = f"Mode {mode_index} has zero displacement"
        raise ValueError(msg)

    scaled = local * (amplitude / max_length)
    displacement = np.zeros((n_atoms, 3))
    for local_index, atom_index in enumerate(atom_indices):
        displacement[atom_index] = scaled[local_index]
    return displacement


def _fixed_atom_indices(atoms: Atoms) -> np.ndarray:
    indices: list[int] = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices.extend(int(index) for index in constraint.get_indices())
    return np.array(sorted(set(indices)), dtype=int)


def _displace(atoms: Atoms, displacement: np.ndarray) -> Atoms:
    """Copy ``atoms`` and apply a Cartesian displacement that honors constraints."""
    frame = cast(Atoms, atoms.copy())
    shift = np.array(displacement, dtype=float, copy=True)
    fixed = _fixed_atom_indices(frame)
    if fixed.size:
        shift[fixed] = 0.0

    new_positions = frame.positions + shift
    for constraint in frame.constraints:
        adjust_positions = getattr(constraint, "adjust_positions", None)
        if adjust_positions is not None:
            adjust_positions(frame, new_positions)
    frame.set_positions(new_positions)
    return frame


def _oscillate(
    atoms: Atoms,
    direction: np.ndarray,
    nframes: int,
    mode_index: int,
    frequency: float,
) -> list[Atoms]:
    frames: list[Atoms] = []
    for step in range(nframes):
        phase = np.sin(2.0 * np.pi * step / nframes)
        frame = _displace(atoms, phase * direction)
        frame.info["mode"] = mode_index
        frame.info["frequency"] = frequency
        frames.append(frame)
    return frames
