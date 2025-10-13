"""ASE-compatible interface for geomeTRIC optimization.

This module provides an efficient ASE optimizer wrapper around geomeTRIC's optimization
capabilities, avoiding file I/O and supporting Hessian input for TS optimization.
"""

import logging
import os
import sys
import tempfile
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.units import Bohr

from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


class _DevNull:
    """Null output device to suppress unwanted output."""

    def write(self, msg: Any) -> None:
        pass

    def flush(self) -> None:
        pass


# Global null device for suppressing output
_DEVNULL = _DevNull()

# Import geomeTRIC modules at module level for efficiency
try:
    # Redirect stdout/stderr during geomeTRIC imports to suppress warnings
    with redirect_stdout(_DEVNULL), redirect_stderr(_DEVNULL):
        import geometric.ase_engine
        import geometric.molecule
        import geometric.optimize as geo_opt
        from geometric.errors import GeomOptNotConvergedError

    # Suppress geomeTRIC warnings at module level
    warnings.filterwarnings("ignore", category=UserWarning, module="geometric")
    warnings.filterwarnings("ignore", message=".*OutOfPlane atoms are the same.*")
    warnings.filterwarnings("ignore", message=".*Warning: OutOfPlane atoms are the same.*")

    # Also suppress geomeTRIC's internal logging
    geometric_logger = logging.getLogger("geometric")
    geometric_logger.setLevel(logging.CRITICAL)

    # Suppress geomeTRIC's nifty module logging
    nifty_logger = logging.getLogger("geometric.nifty")
    nifty_logger.setLevel(logging.CRITICAL)

    # Monkeypatch EngineASE to preserve charge and spin information
    # This fixes the issue where geomeTRIC doesn't transfer atoms.info metadata
    _original_enginease_init = geometric.ase_engine.EngineASE.__init__

    def _patched_enginease_init(self, molecule, calculator):
        """Monkeypatched EngineASE.__init__ to preserve charge and spin info."""
        # Call the original __init__ method first
        _original_enginease_init(self, molecule, calculator)

        # Ensure ase_atoms has charge and spin info to prevent UMA warnings
        self.ase_atoms.info.setdefault("charge", 0)
        self.ase_atoms.info.setdefault("spin", 1)

    # Apply the monkeypatch
    geometric.ase_engine.EngineASE.__init__ = _patched_enginease_init

    GEOMETRIC_AVAILABLE = True
except ImportError:
    # Will be handled when optimization is actually called
    GEOMETRIC_AVAILABLE = False


class _StepTrackingEngine:
    """Custom engine that tracks optimization steps and prints ASE format progress."""

    def __init__(self, molecule, calc, atoms_ref=None):
        """Initialize the step tracking engine."""
        self._engine = geometric.ase_engine.EngineASE(molecule, calc)
        self.step_count = 0
        self._atoms_ref = atoms_ref

    def set_atoms_reference(self, atoms):
        """Set reference to the atoms object being optimized."""
        self._atoms_ref = atoms

    def calc_new(self, coords, dirname):
        """Override to track optimization steps."""
        # Update atoms with new coordinates for step tracking
        positions = coords.reshape(-1, 3) * Bohr  # Bohr to Angstrom
        self._atoms_ref.set_positions(positions)

        # Delegate to the underlying engine for actual calculation
        return self._engine.calc_new(coords, dirname)

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying engine."""
        return getattr(self._engine, name)


class GeometricOptimizer(Optimizer):
    """Efficient ASE-compatible wrapper for geomeTRIC optimization.

    This class provides an ASE optimizer interface that uses geomeTRIC
    for both minima and transition state optimization without file I/O.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to optimize
    order : int, default 0
        Order of saddle point to find (0 for minima, 1 for TS, etc.)
    hessian : np.ndarray, optional
        Initial Hessian matrix for optimization (3N x 3N)
    **kwargs
        Additional keyword arguments passed to geomeTRIC
    """

    def __init__(
        self,
        atoms: Atoms,
        order: int = 0,
        hessian: Optional[np.ndarray] = None,
        **kwargs,
    ):
        # Store geomeTRIC-specific parameters before filtering
        self.order = order
        self.initial_hessian = hessian

        # Separate ASE and geomeTRIC kwargs efficiently
        ase_kwargs = {}
        self.geometric_kwargs = {}

        # geomeTRIC-specific keys to exclude from ASE
        geometric_keys = {"trust", "convergence", "maxiter", "hessian"}
        # ASE-specific keys to exclude from geomeTRIC
        ase_keys = {"logfile", "restart", "append_trajectory"}

        for key, value in kwargs.items():
            if key in geometric_keys:
                self.geometric_kwargs[key] = value
            elif key not in ase_keys:
                ase_kwargs[key] = value

        # Initialize ASE Optimizer base class with filtered kwargs
        super().__init__(atoms, **ase_kwargs)

        # Set up geomeTRIC-specific defaults
        self.geometric_kwargs.setdefault(
            "convergence", {"energy": 1e-6, "gradient": 1e-3, "step": 1e-3}
        )
        self.geometric_kwargs.setdefault("maxiter", 1000)
        self.geometric_kwargs.setdefault("trust", 0.1)

        # Track optimization state
        self.step_count = 0
        self.converged = False

    def _create_molecule_from_atoms(self, atoms: Atoms):
        """Create geomeTRIC molecule object efficiently using minimal file I/O."""
        # Create a minimal temporary XYZ file (more efficient than full file I/O)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            # Write minimal XYZ content
            f.write(f"{len(atoms)}\n")
            f.write("Generated from ASE atoms\n")
            for i, pos in enumerate(atoms.get_positions()):
                symbol = atoms.get_chemical_symbols()[i]
                f.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            temp_file = f.name

        try:
            # Create molecule from the temporary file
            return geometric.molecule.Molecule(temp_file)
        finally:
            # Clean up the temporary file immediately
            os.unlink(temp_file)

    def _create_optimization_params(self, fmax: float, steps: int):
        """Create and configure geomeTRIC optimization parameters."""
        params = geo_opt.OptParams()
        # Set optimization type: False for minima, True for transition state
        params.transition = self.order > 0
        params.maxiter = steps
        # Set convergence criteria using correct geomeTRIC parameter names
        params.Convergence_gmax = fmax  # Maximum force threshold
        params.Convergence_energy = self.geometric_kwargs["convergence"]["energy"]
        params.Convergence_dmax = self.geometric_kwargs["convergence"]["step"]
        params.trust = self.geometric_kwargs["trust"]
        # Explicitly set xyzout to None to prevent file output issues
        params.xyzout = None
        # Disable frequency analysis to prevent the NoneType.replace() error
        params.frequency = False
        # Enable verbose output to capture optimization steps
        params.verbose = True

        # Set initial Hessian if provided
        if self.initial_hessian is not None:
            expected_shape = (3 * len(self.atoms), 3 * len(self.atoms))
            if self.initial_hessian.shape != expected_shape:
                raise ValueError(
                    f"Hessian must be {expected_shape} but got " f"{self.initial_hessian.shape}"
                )
            # Store the Hessian data and set hessian to 'first' to use it
            params.hess_data = self.initial_hessian.copy()
            params.hessian = "first"

        return params

    def _print_ase_format_steps(self, optimizer):
        """Print all optimization steps in ASE format at once."""
        # Print ASE format header
        logger.info("       Step     Time          Energy          fmax")

        # Print all optimization steps in ASE format
        for i, coords in enumerate(optimizer.progress.xyzs):
            # Update atoms with coordinates for this step
            positions = coords  # Already in Angstrom from geomeTRIC
            self.atoms.set_positions(positions)

            # Get energy and forces for this step
            energy = self.atoms.get_potential_energy()
            forces = self.atoms.get_forces()

            fmax = float(np.max(np.abs(forces)))
            energy_val = float(energy)

            # Format time string
            time_str = time.strftime("%H:%M:%S", time.localtime())

            # Print step information in ASE format
            logger.info(f"GEOMETRIC: {i:>4} {time_str:>8} " f"{energy_val:>12.6f} {fmax:>12.6f}")

    def _run_optimization(self, optimizer, step_engine):
        """Run the optimization and handle convergence."""
        # Capture stdout to suppress geomeTRIC's verbose output
        import io
        import sys
        from contextlib import redirect_stdout

        # Create a string buffer to capture output
        captured_output = io.StringIO()

        try:
            # Redirect stdout to capture geomeTRIC's output
            with redirect_stdout(captured_output):
                optimizer.optimizeGeometry()
            self.converged = True
        except GeomOptNotConvergedError:
            # Normal convergence failure - this is expected in many cases
            self.converged = False
        except Exception as e:
            # Unexpected error during optimization
            self.converged = False
            raise RuntimeError(f"Optimization failed: {e}") from e

        # Extract step count from geomeTRIC's optimization progress
        self.step_count = len(optimizer.progress.xyzs) - 1  # Subtract 1 for initial step
        self.step_count = max(0, self.step_count)  # Ensure non-negative

        # Print optimization steps in ASE format using captured progress data
        # Temporarily restore stdout to print ASE format steps
        original_stdout = sys.stdout
        sys.stdout = sys.__stdout__  # Use the original stdout
        self._print_ase_format_steps(optimizer)
        sys.stdout = original_stdout  # Restore redirected stdout

        # Update atoms with optimized geometry
        # geomeTRIC stores optimization history in optimizer.progress.xyzs (Angstrom)
        final_xyz = optimizer.progress.xyzs[-1]
        self.atoms.set_positions(final_xyz)

        # Check convergence using geomeTRIC's state system
        # geomeTRIC uses state=2 for converged, state=1 for not converged
        state = optimizer.state
        self.converged = (state == 2).any() if hasattr(state, "__iter__") else state == 2

    def run(self, fmax: float = 0.05, steps: int = 1000) -> bool:
        """Run geomeTRIC optimization efficiently without file I/O.

        Parameters
        ----------
        fmax : float
            Maximum force threshold for convergence
        steps : int
            Maximum number of optimization steps

        Returns
        -------
        bool
            True if optimization converged, False otherwise
        """
        # Update convergence criteria based on ASE parameters
        self.geometric_kwargs["convergence"]["gradient"] = fmax
        self.geometric_kwargs["maxiter"] = steps

        # Store original stdout/stderr for restoration
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            # Redirect stdout/stderr for the entire optimization process
            sys.stdout = _DEVNULL
            sys.stderr = _DEVNULL

            # Create molecule object directly from ASE atoms (no file I/O)
            molecule = self._create_molecule_from_atoms(self.atoms)

            # Set up optimization parameters
            params = self._create_optimization_params(fmax, steps)

            # Get initial coordinates in Bohr
            coords = self.atoms.get_positions().flatten() / Bohr  # Angstrom to Bohr

            # Create internal coordinates
            IC = geo_opt.DelocalizedInternalCoordinates(
                molecule, build=True, connect=False, addcart=False
            )

            # Use a minimal temporary directory (only for geomeTRIC's internal needs)
            with tempfile.TemporaryDirectory() as tmpdir:
                # Initial step will be printed with all other steps after optimization

                # Create custom engine with step tracking
                step_engine = _StepTrackingEngine(molecule, self.atoms.calc, self.atoms)

                # Create optimizer
                optimizer = geo_opt.Optimizer(
                    coords=coords,
                    molecule=molecule,
                    IC=IC,
                    engine=step_engine,
                    dirname=str(tmpdir),
                    params=params,
                    print_info=False,  # Disable geomeTRIC output to avoid duplication
                )

                # Run optimization and handle convergence
                self._run_optimization(optimizer, step_engine)

                return self.converged
        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr


class GeometricTSOptimizer(GeometricOptimizer):
    """Specialized geomeTRIC optimizer for transition state searches.

    This is a convenience class that sets order=1 for transition state optimization.
    """

    def __init__(self, atoms: Atoms, **kwargs):
        super().__init__(atoms, order=1, **kwargs)


def create_geometric_optimizer(
    atoms: Atoms, order: int = 0, hessian: Optional[np.ndarray] = None, **kwargs
) -> GeometricOptimizer:
    """Factory function to create geomeTRIC optimizer."""
    if order == 1:
        return GeometricTSOptimizer(atoms, hessian=hessian, **kwargs)
    return GeometricOptimizer(atoms, order=order, hessian=hessian, **kwargs)


# Convenience functions for common use cases
def geometric_minima_optimizer(
    atoms: Atoms, hessian: Optional[np.ndarray] = None, **kwargs
) -> GeometricOptimizer:
    """Create geomeTRIC optimizer for minima search."""
    return GeometricOptimizer(atoms, order=0, hessian=hessian, **kwargs)


def geometric_ts_optimizer(
    atoms: Atoms, hessian: Optional[np.ndarray] = None, **kwargs
) -> GeometricTSOptimizer:
    """Create geomeTRIC optimizer for transition state search."""
    return GeometricTSOptimizer(atoms, hessian=hessian, **kwargs)
