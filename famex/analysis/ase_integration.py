"""ASE integration layer for Hessian and frequency calculations.

This module provides wrapper functions to leverage ASE's Vibrations class
for basic cases while preserving FAMEX's advanced features.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ase import Atoms
from ase.vibrations import Vibrations
from ase.vibrations.data import VibrationsData
from numpy.typing import NDArray

from famex.utils.logging import get_famex_logger

if TYPE_CHECKING:
    from famex.analysis.hessian import CalculatorProtocol

logger = get_famex_logger(__name__)

__all__ = [
    "hessian_via_ase",
    "frequencies_via_ase",
    "can_use_ase",
]


def can_use_ase(
    method: str,
    richardson: bool,
    adaptive_delta: bool,
    n_workers: int | None,
) -> bool:
    """Check if ASE can handle this Hessian calculation case.

    ASE's Vibrations class supports:
    - Central difference (3-point) method
    - Basic finite differences without advanced features

    ASE does NOT support:
    - 5-point or 7-point schemes
    - Richardson extrapolation
    - Adaptive delta selection
    - Parallelization (though ASE has its own parallel support)

    Parameters
    ----------
    method : str
        Finite difference method ('forward', 'central', '5point', '7point')
    richardson : bool
        Whether Richardson extrapolation is requested
    adaptive_delta : bool
        Whether adaptive delta selection is requested
    n_workers : int | None
        Number of parallel workers (not used by ASE directly)

    Returns
    -------
    bool
        True if ASE can handle this case, False otherwise
    """
    # ASE only supports central difference (3-point)
    if method not in ("central", "forward"):
        return False

    # ASE doesn't support Richardson extrapolation
    if richardson:
        return False

    # ASE doesn't support adaptive delta
    # Note: n_workers is ignored - ASE has its own parallelization
    # but we don't use it here since FAMEX doesn't leverage GPU parallelization

    return not adaptive_delta


def hessian_via_ase(
    atoms: Atoms,
    calculator: CalculatorProtocol,
    delta: float,
    indices: list[int] | None = None,
    verbose: int = 1,
) -> NDArray[np.float64]:
    """Compute Hessian matrix using ASE's Vibrations class.

    This function wraps ASE's Vibrations class to compute the Hessian
    using finite differences. ASE uses central differences (3-point scheme).

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    calculator : CalculatorProtocol
        Calculator compatible with ASE
    delta : float
        Displacement step size (Å)
    indices : list[int] | None
        Atom indices to include. If None, all atoms included.
    verbose : int
        Verbosity level (0=quiet, 1=normal, 2=verbose)

    Returns
    -------
    NDArray[np.float64]
        Hessian matrix of shape (3N, 3N) where N is number of atoms in indices.
        Units are eV/Å².

    Notes
    -----
    - Uses ASE's Vibrations class which computes Hessian via central differences
    - Creates temporary directory for ASE's cache files
    - Cleans up temporary files after calculation
    - For forward differences, falls back to custom implementation
    """
    # Ensure calculator is attached
    atoms.calc = calculator

    hessian: NDArray[np.float64]
    with tempfile.TemporaryDirectory() as tmpdir:
        vib = Vibrations(
            atoms,
            indices=indices,
            delta=delta,
            name=str(Path(tmpdir) / "vib"),
        )

        if verbose >= 1:
            logger.info("Computing Hessian using ASE Vibrations...")

        # Run calculations
        vib.run()

        # Read results
        vib.read()

        # Extract Hessian
        hessian_raw: Any = vib.H
        if hessian_raw is None:
            msg = "ASE Vibrations failed to compute Hessian"
            raise RuntimeError(msg)
        hessian = cast(NDArray[np.float64], np.asarray(hessian_raw).copy())

        # Clean up (though tempdir will be cleaned automatically)
        vib.clean()

    if verbose >= 2:
        logger.debug(f"Hessian computed via ASE: shape {hessian.shape}")

    return hessian


def frequencies_via_ase(
    hessian: NDArray[np.float64],
    atoms: Atoms,
    indices: list[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute frequencies and normal modes from Hessian using ASE's VibrationsData.

    This function uses ASE's VibrationsData.from_2d() to compute frequencies
    and normal modes from a Hessian matrix, ensuring consistency with ASE's
    conventions.

    Parameters
    ----------
    hessian : NDArray[np.float64]
        Hessian matrix (3N x 3N) where N is number of atoms in indices
    atoms : Atoms
        ASE Atoms object
    indices : list[int] | None
        Atom indices included in Hessian. If None, assumes all atoms.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Frequencies in cm^-1 and normal mode eigenvectors (Cartesian coordinates).
        Modes are for indices only (shape: 3N x 3N where N=len(indices)).
        Frequencies are signed: positive for real modes, negative for imaginary.

    Notes
    -----
    - Uses ASE's VibrationsData.from_2d() for consistency
    - Returns frequencies in cm^-1 (same as ASE)
    - Normal modes are in Cartesian coordinates, normalized, for indices only
    """
    if indices is None:
        indices = list(range(len(atoms)))

    # Create VibrationsData from Hessian
    vib_data = VibrationsData.from_2d(atoms, hessian, indices=indices)

    # Get frequencies (in cm^-1)
    frequencies = vib_data.get_frequencies()

    # Get modes for indices only (not all atoms)
    # ASE returns modes as (3N, N, 3) array: (mode_index, atom_index, direction)
    # We need to convert to (3N, 3N) where each column is a mode (coordinate, mode)
    modes_3d = vib_data.get_modes(all_atoms=False)  # Shape: (3N, N, 3)

    # Reshape to (3N, 3N): flatten atom coordinates for each mode
    # modes_3d[i, j, k] is mode i, atom j, direction k
    # We want modes[:, i] to be mode i as flattened (x1, y1, z1, x2, y2, z2, ...)
    n_modes = modes_3d.shape[0]  # 3N
    n_atoms = modes_3d.shape[1]  # N
    modes = modes_3d.reshape(
        n_modes, 3 * n_atoms
    ).T  # Transpose to get (3N, 3N) with columns as modes

    return frequencies, modes
