"""ASE-compatible interface for geomeTRIC optimization.

This module provides an efficient ASE optimizer wrapper around geomeTRIC's optimization
capabilities, avoiding file I/O and supporting Hessian input for TS optimization.
"""

import tempfile
import warnings
from typing import Optional

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer


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
        self.geometric_kwargs = kwargs.copy()

        # Remove geomeTRIC-specific kwargs that ASE doesn't understand
        ase_kwargs = kwargs.copy()
        ase_kwargs.pop("trust", None)
        ase_kwargs.pop("convergence", None)
        ase_kwargs.pop("maxiter", None)
        ase_kwargs.pop("hessian", None)

        # Initialize ASE Optimizer base class with filtered kwargs
        super().__init__(atoms, **ase_kwargs)

        # Remove ASE-specific kwargs that geomeTRIC doesn't understand
        self.geometric_kwargs.pop("logfile", None)
        self.geometric_kwargs.pop("restart", None)
        self.geometric_kwargs.pop("append_trajectory", None)

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
        import os

        import geometric.molecule

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
            molecule = geometric.molecule.Molecule(temp_file)
            return molecule
        finally:
            # Clean up the temporary file immediately
            os.unlink(temp_file)

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
        # Import geomeTRIC modules
        try:
            import geometric.ase_engine
            import geometric.optimize as geo_opt
            from geometric.errors import GeomOptNotConvergedError
        except ImportError as e:
            raise ImportError(f"geomeTRIC is required for optimization: {e}")

        # Update convergence criteria based on ASE parameters
        self.geometric_kwargs["convergence"]["gradient"] = fmax
        self.geometric_kwargs["maxiter"] = steps

        # Create molecule object directly from ASE atoms (no file I/O)
        molecule = self._create_molecule_from_atoms(self.atoms)

        # Create ASE engine with the calculator from atoms
        if self.atoms.calc is None:
            raise RuntimeError("Atoms object must have a calculator attached")

        engine = geometric.ase_engine.EngineASE(molecule, self.atoms.calc)

        # Set up optimization parameters
        params = geo_opt.OptParams()
        params.order = self.order
        params.maxiter = steps
        params.convergence_gradient = fmax
        params.convergence_energy = self.geometric_kwargs["convergence"]["energy"]
        params.convergence_step = self.geometric_kwargs["convergence"]["step"]
        params.trust = self.geometric_kwargs["trust"]
        # Explicitly set xyzout to None to prevent file output issues
        params.xyzout = None
        # Disable frequency analysis to prevent the NoneType.replace() error
        params.frequency = False

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

        # Get initial coordinates in Bohr
        from ase.units import Bohr

        coords = self.atoms.get_positions().flatten() / Bohr  # Angstrom to Bohr

        # Create internal coordinates
        IC = geo_opt.DelocalizedInternalCoordinates(
            molecule, build=True, connect=False, addcart=False
        )

        # Use a minimal temporary directory (only for geomeTRIC's internal needs)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create optimizer
            optimizer = geo_opt.Optimizer(
                coords=coords,
                molecule=molecule,
                IC=IC,
                engine=engine,
                dirname=str(tmpdir),
                params=params,
                print_info=False,
            )

            # Run optimization - only catch GeomOptNotConvergedError
            try:
                optimizer.optimizeGeometry()
            except GeomOptNotConvergedError:
                # Normal convergence failure - still need to update coordinates
                self.step_count = getattr(
                    optimizer,
                    "Iteration",
                    getattr(optimizer, "iter", getattr(optimizer, "iteration", 0)),
                )
                self.converged = False
                # Don't return yet - continue to coordinate extraction

            # Extract step count from geomeTRIC optimizer
            # geomeTRIC stores the iteration count in different attributes
            if hasattr(optimizer, "Iteration") and optimizer.Iteration is not None:
                self.step_count = optimizer.Iteration
            elif hasattr(optimizer, "iter") and optimizer.iter is not None:
                self.step_count = optimizer.iter
            elif hasattr(optimizer, "iteration") and optimizer.iteration is not None:
                self.step_count = optimizer.iteration
            elif hasattr(optimizer, "step_count") and optimizer.step_count is not None:
                self.step_count = optimizer.step_count
            elif hasattr(optimizer, "nsteps") and optimizer.nsteps is not None:
                self.step_count = optimizer.nsteps
            else:
                # Fallback: try to get from history or progress
                if hasattr(optimizer, "history") and optimizer.history:
                    self.step_count = len(optimizer.history)
                elif hasattr(optimizer, "xyzs") and optimizer.xyzs:
                    self.step_count = len(optimizer.xyzs) - 1  # Subtract initial structure
                elif hasattr(optimizer, "progress") and optimizer.progress:
                    self.step_count = len(optimizer.progress)
                else:
                    # Last resort: check if there's a convergence log or similar
                    # geomeTRIC sometimes stores step info in the molecule object
                    if hasattr(molecule, "xyzs") and molecule.xyzs:
                        self.step_count = len(molecule.xyzs) - 1  # Subtract initial structure
                    else:
                        self.step_count = 0

            # Check if optimization converged using geomeTRIC's state system
            state = getattr(optimizer, "state", None)
            if state is None:
                converged = False
            elif np.isscalar(state):
                converged = state == 2
            else:
                converged = (state == 2).any()

            # Update atoms with optimized geometry
            # geomeTRIC stores the final coordinates in the molecule object
            if hasattr(molecule, "xyzs") and len(molecule.xyzs) > 0:
                # Get the final optimized geometry
                final_xyz = molecule.xyzs[-1]
                # Convert from Bohr to Angstrom
                final_xyz_ang = final_xyz * Bohr  # Bohr to Angstrom
                self.atoms.set_positions(final_xyz_ang.reshape(-1, 3))
                self.converged = converged
                return converged
            elif hasattr(optimizer, "xyzs") and len(optimizer.xyzs) > 0:
                # Fallback: check optimizer.xyzs
                final_xyz = optimizer.xyzs[-1]
                final_xyz_ang = final_xyz * Bohr  # Bohr to Angstrom
                self.atoms.set_positions(final_xyz_ang.reshape(-1, 3))
                self.converged = converged
                return converged
            else:
                # If no coordinates found, check if we can get them from the current
                # state
                if hasattr(optimizer, "coords") and optimizer.coords is not None:
                    # Get current coordinates from optimizer
                    final_xyz_ang = optimizer.coords * Bohr  # Bohr to Angstrom
                    self.atoms.set_positions(final_xyz_ang.reshape(-1, 3))
                    self.converged = converged
                    return converged
                else:
                    raise RuntimeError("geomeTRIC optimization did not return valid results")


class GeometricTSOptimizer(GeometricOptimizer):
    """Specialized geomeTRIC optimizer for transition state searches.

    This is a convenience class that sets order=1 for transition state optimization.
    """

    def __init__(self, atoms: Atoms, **kwargs):
        super().__init__(atoms, order=1, **kwargs)


def create_geometric_optimizer(
    atoms: Atoms, order: int = 0, hessian: Optional[np.ndarray] = None, **kwargs
) -> GeometricOptimizer:
    """Factory function to create geomeTRIC optimizer.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to optimize
    order : int, default 0
        Order of saddle point to find (0 for minima, 1 for TS, etc.)
    hessian : np.ndarray, optional
        Initial Hessian matrix for optimization (3N x 3N)
    **kwargs
        Additional keyword arguments

    Returns
    -------
    GeometricOptimizer
        Configured geomeTRIC optimizer
    """
    if order == 1:
        return GeometricTSOptimizer(atoms, hessian=hessian, **kwargs)
    else:
        return GeometricOptimizer(atoms, order=order, hessian=hessian, **kwargs)


# Convenience functions for common use cases
def geometric_minima_optimizer(
    atoms: Atoms, hessian: Optional[np.ndarray] = None, **kwargs
) -> GeometricOptimizer:
    """Create geomeTRIC optimizer for minima search.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to optimize
    hessian : np.ndarray, optional
        Initial Hessian matrix for optimization (3N x 3N)
    **kwargs
        Additional keyword arguments

    Returns
    -------
    GeometricOptimizer
        Configured geomeTRIC optimizer for minima search
    """
    return GeometricOptimizer(atoms, order=0, hessian=hessian, **kwargs)


def geometric_ts_optimizer(
    atoms: Atoms, hessian: Optional[np.ndarray] = None, **kwargs
) -> GeometricTSOptimizer:
    """Create geomeTRIC optimizer for transition state search.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms object to optimize
    hessian : np.ndarray, optional
        Initial Hessian matrix for optimization (3N x 3N)
    **kwargs
        Additional keyword arguments

    Returns
    -------
    GeometricTSOptimizer
        Configured geomeTRIC optimizer for transition state search
    """
    return GeometricTSOptimizer(atoms, hessian=hessian, **kwargs)
