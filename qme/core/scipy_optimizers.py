"""
SciPy-based optimizers with Hessian support for QME.

This module provides ASE-compatible optimizer wrappers for SciPy's
second-order optimization methods that require Hessian matrices.
These methods are particularly useful for transition state searches
and challenging optimization landscapes.

The implementation leverages QME's efficient Hessian calculation
infrastructure (FrequencyAnalysis) which can compute Hessians cheaply
using machine learning potentials via finite differences, batch evaluation,
or direct calculation.

Supported optimizers:
- Trust-Krylov: Trust region method with Krylov subspace solver
- Trust-NCG: Trust region Newton Conjugate Gradient
- Trust-Exact: Exact trust region method (nearly exact solver)
- Newton-CG: Newton's method with Conjugate Gradient

Based on ASE's SciPyOptimizer pattern but extended for Hessian-based methods.
"""

from typing import IO, Any, Callable, Optional, Union

import numpy as np
from ase import Atoms, units
from ase.optimize.optimize import Optimizer
from scipy.optimize import minimize

from qme.analysis.frequency import FrequencyAnalysis
from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


class Converged(Exception):
    """Exception raised when optimizer has converged."""

    pass


class SciPyHessianOptimizer(Optimizer):
    """
    Base class for SciPy optimizers that use Hessian information.

    This class provides an ASE-compatible interface to SciPy's second-order
    optimization methods. It computes Hessians efficiently using QME's
    FrequencyAnalysis class.

    Parameters
    ----------
    atoms : Atoms
        The Atoms object to optimize.
    method : str
        SciPy optimization method ('trust-krylov', 'trust-ncg', 'trust-exact', 'Newton-CG')
    logfile : Union[IO, str]
        File object or filename for logging. Use '-' for stdout.
    trajectory : Optional[str]
        Trajectory file to store optimization path.
    hessian_update_freq : int
        Frequency of full Hessian recalculation (in steps). Default is 1 (every step).
        Set higher to reduce computational cost at the expense of accuracy.
    hessian_method : str
        Method for Hessian calculation: 'auto', 'batch', 'finite_differences', 'direct'
        Default is 'auto' which selects the best available method.
    hessian_delta : float
        Step size for finite difference Hessian calculation (Å). Default is 0.01.
    initial_hessian : Optional[np.ndarray]
        Initial Hessian matrix. If None, computed on first step.
    alpha : float
        Initial scaling factor for Hessian. Default is 1.0 (no scaling).
        Can be adjusted to improve convergence.
    use_bfgs_update : bool
        Use BFGS approximate updates between full Hessian calculations.
        Default is True. Significantly reduces computational cost.
    adaptive_hessian : bool
        Use adaptive Hessian update frequency based on convergence behavior.
        Default is True. Updates more frequently when forces increase.
    force_threshold_ratio : float
        Ratio of force increase that triggers a full Hessian update.
        Default is 2.0. Only used if adaptive_hessian=True.
    **kwargs
        Additional arguments passed to Optimizer base class.

    Attributes
    ----------
    freq_analysis : FrequencyAnalysis
        FrequencyAnalysis instance for Hessian computation.
    hessian : Optional[np.ndarray]
        Current Hessian matrix.
    force_calls : int
        Number of force/gradient evaluations.
    hessian_calls : int
        Number of Hessian evaluations.
    """

    def __init__(
        self,
        atoms,
        restart=None,
        logfile="-",
        trajectory=None,
        maxstep=0.2,
        master=None,
        force_consistent=None,
        hessian_method="auto",
        hessian_update_freq=None,
        use_bfgs_update=False,
        adaptive_hessian=False,
        force_threshold_ratio=2.0,
    ):
        """Initialize Newton-CG optimizer.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to optimize
        hessian_method : str, optional
            Method for Hessian calculation: 'auto', 'fd', 'direct', 'batch'
        hessian_update_freq : int or None, optional
            How often to recompute the full Hessian (in steps).
            If None, compute only once at the beginning.
            Default is None (single Hessian at start).
        use_bfgs_update : bool, optional
            Use BFGS approximate updates between full Hessians.
            Default is False.
        adaptive_hessian : bool, optional
            Enable adaptive Hessian update frequency based on force changes.
            Default is False.
        force_threshold_ratio : float, optional
            Ratio threshold for adaptive Hessian updates. If forces increase
            by more than this ratio, trigger a full Hessian update.
            Default is 2.0.
        """
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=master,
            force_consistent=force_consistent,
        )
        self.method = "Newton-CG"
        self.hessian_method = hessian_method
        self.hessian_update_freq = hessian_update_freq
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio
        """Initialize Trust-Exact optimizer.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to optimize
        hessian_method : str, optional
            Method for Hessian calculation: 'auto', 'fd', 'direct', 'batch'
        hessian_update_freq : int or None, optional
            How often to recompute the full Hessian (in steps).
            If None, compute only once at the beginning.
            Default is None (single Hessian at start).
        use_bfgs_update : bool, optional
            Use BFGS approximate updates between full Hessians.
            Default is False.
        adaptive_hessian : bool, optional
            Enable adaptive Hessian update frequency based on force changes.
            Default is False.
        force_threshold_ratio : float, optional
            Ratio threshold for adaptive Hessian updates. If forces increase
            by more than this ratio, trigger a full Hessian update.
            Default is 2.0.
        """
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=master,
            force_consistent=force_consistent,
        )
        self.method = "trust-exact"
        self.hessian_method = hessian_method
        self.hessian_update_freq = hessian_update_freq
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio
        """Initialize Trust-NCG optimizer.

        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to optimize
        hessian_method : str, optional
            Method for Hessian calculation: 'auto', 'fd', 'direct', 'batch'
        hessian_update_freq : int or None, optional
            How often to recompute the full Hessian (in steps).
            If None, compute only once at the beginning.
            Default is None (single Hessian at start).
        use_bfgs_update : bool, optional
            Use BFGS approximate updates between full Hessians.
            Default is False.
        adaptive_hessian : bool, optional
            Enable adaptive Hessian update frequency based on force changes.
            Default is False.
        force_threshold_ratio : float, optional
            Ratio threshold for adaptive Hessian updates. If forces increase
            by more than this ratio, trigger a full Hessian update.
            Default is 2.0.
        """
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=master,
            force_consistent=force_consistent,
        )
        self.method = "trust-ncg"
        self.hessian_method = hessian_method
        self.hessian_update_freq = hessian_update_freq
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio
        """Initialize SciPy Hessian-based optimizer."""
        # Don't use restart for SciPy optimizers
        restart = None
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **kwargs)

        self.method = method
        self.hessian_update_freq = hessian_update_freq
        self.hessian_method = hessian_method
        self.alpha = alpha
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio

        # Initialize FrequencyAnalysis for Hessian computation
        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator attached")

        self.freq_analysis = FrequencyAnalysis(
            atoms=atoms, calculator=atoms.calc, delta=hessian_delta
        )

        # Hessian state
        self.hessian = initial_hessian
        self.force_calls = 0
        self.hessian_calls = 0
        self._last_hessian_step = -1
        self._last_full_hessian_step = -1
        self.bfgs_updates = 0

        # State for BFGS updates
        self._last_positions = None
        self._last_gradient = None
        self._previous_fmax = None

        # SciPy optimization state
        self.max_steps = 0
        self._scipy_result = None

        # Validate method
        valid_methods = ["trust-krylov", "trust-ncg", "trust-exact", "Newton-CG"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Valid methods: {valid_methods}"
            )

        logger.info(f"Initialized {method} optimizer with Hessian support")
        if adaptive_hessian:
            logger.info(
                f"Adaptive Hessian updates enabled (base frequency: every {hessian_update_freq} step(s))"
            )
            if use_bfgs_update:
                logger.info("BFGS approximate updates enabled between full Hessians")
        else:
            logger.info(
                f"Fixed Hessian update frequency: every {hessian_update_freq} step(s)"
            )
        logger.info(f"Hessian calculation method: {hessian_method}")

    def _positions_to_x(self, atoms: Optional[Atoms] = None) -> np.ndarray:
        """Convert atoms positions to 1D array for SciPy."""
        if atoms is None:
            atoms = self.atoms
        return atoms.get_positions().ravel()

    def _x_to_positions(self, x: np.ndarray) -> np.ndarray:
        """Convert 1D array to positions array."""
        return x.reshape(-1, 3)

    def objective(self, x: np.ndarray) -> float:
        """
        Objective function for minimization (potential energy).

        Parameters
        ----------
        x : np.ndarray
            Flattened position array.

        Returns
        -------
        float
            Potential energy scaled by alpha.
        """
        self.atoms.set_positions(self._x_to_positions(x))
        energy = self.atoms.get_potential_energy()
        # Scale by alpha (Hessian scaling factor)
        return energy / self.alpha

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of objective function (negative forces).

        Parameters
        ----------
        x : np.ndarray
            Flattened position array.

        Returns
        -------
        np.ndarray
            Gradient (negative forces), flattened and scaled.
        """
        self.atoms.set_positions(self._x_to_positions(x))
        self.force_calls += 1

        # Forces are negative gradient
        forces = self.atoms.get_forces()
        gradient = -forces.ravel()
        
        # Store for BFGS updates
        if self.use_bfgs_update:
            if self._last_gradient is None:
                self._last_gradient = gradient.copy()
            if self._last_positions is None:
                self._last_positions = x.copy()

        # Scale by alpha
        return gradient / self.alpha

    def hessian_func(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix with adaptive updates and BFGS approximation.

        This uses QME's FrequencyAnalysis to compute the Hessian efficiently.
        Supports:
        - Adaptive update frequency based on convergence behavior
        - BFGS approximate updates between full Hessian calculations
        - Caching and periodic updates

        Parameters
        ----------
        x : np.ndarray
            Flattened position array.

        Returns
        -------
        np.ndarray
            Hessian matrix (3N x 3N), scaled by alpha.
        """
        current_positions = x.copy()
        steps_since_last = self.nsteps - self._last_hessian_step
        steps_since_full = self.nsteps - self._last_full_hessian_step
        
        # Determine if we need a full Hessian update
        need_full_update = False
        reason = "unknown"
        
        if self.hessian is None:
            # Always compute on first step
            need_full_update = True
            reason = "initial"
        elif self.adaptive_hessian:
            # Adaptive logic: update when needed
            current_fmax = self._get_current_fmax()
            
            # Force-based criterion: large forces suggest need for update
            if current_fmax is not None and self._previous_fmax is not None:
                if current_fmax > self._previous_fmax * self.force_threshold_ratio:
                    need_full_update = True
                    reason = f"force increase ({current_fmax:.4f} > {self._previous_fmax:.4f} × {self.force_threshold_ratio})"
            
            # Periodic update based on base frequency
            if not need_full_update and steps_since_full >= self.hessian_update_freq:
                need_full_update = True
                reason = f"periodic (every {self.hessian_update_freq} steps)"
        else:
            # Fixed frequency mode
            if steps_since_full >= self.hessian_update_freq:
                need_full_update = True
                reason = f"fixed frequency ({self.hessian_update_freq} steps)"
        
        if need_full_update:
            # Compute full Hessian
            self.atoms.set_positions(self._x_to_positions(x))
            self.hessian_calls += 1
            self._last_hessian_step = self.nsteps
            self._last_full_hessian_step = self.nsteps

            logger.info(
                f"Computing full Hessian at step {self.nsteps} "
                f"(call #{self.hessian_calls}, reason: {reason})"
            )

            # Update FrequencyAnalysis with current atoms state
            self.freq_analysis.atoms = self.atoms.copy()
            self.freq_analysis.atoms.calc = self.atoms.calc

            # Compute Hessian using QME's infrastructure
            hessian = self.freq_analysis.calculate_hessian(method=self.hessian_method)

            # Store for potential reuse
            self.hessian = hessian

            logger.info(f"Full Hessian computed (shape: {hessian.shape})")
            
            # Reset BFGS state after full Hessian
            self._last_positions = current_positions.copy()
            self._last_gradient = None
            self.bfgs_updates = 0
            
        elif self.use_bfgs_update and self._last_positions is not None and self._last_gradient is not None:
            # BFGS approximate update
            self._last_hessian_step = self.nsteps
            
            # Get current gradient
            current_gradient = self.gradient(x)
            
            # Compute differences
            s = current_positions - self._last_positions  # position change
            y = current_gradient - self._last_gradient    # gradient change
            
            # BFGS update formula: H_new = H + (y⊗y)/(y·s) - (H·s⊗H·s)/(s·H·s)
            sy = np.dot(s, y)
            
            if sy > 1e-10:  # Ensure positive definiteness
                # Rank-2 update (BFGS formula)
                Hs = np.dot(self.hessian, s)
                sHs = np.dot(s, Hs)
                
                # Update Hessian
                self.hessian = (
                    self.hessian 
                    + np.outer(y, y) / sy 
                    - np.outer(Hs, Hs) / sHs
                )
                
                self.bfgs_updates += 1
                logger.debug(
                    f"BFGS Hessian update at step {self.nsteps} "
                    f"(#{self.bfgs_updates} since last full, "
                    f"{steps_since_full} steps since full Hessian)"
                )
            else:
                logger.debug(f"Skipping BFGS update (sy = {sy:.2e} too small)")
            
            # Store current state for next update
            self._last_positions = current_positions.copy()
            self._last_gradient = current_gradient.copy()
        else:
            # Reuse cached Hessian
            logger.debug(
                f"Reusing Hessian (computed {steps_since_full} steps ago)"
            )

        # Scale by alpha - hessian should never be None at this point
        if self.hessian is None:
            raise RuntimeError("Hessian is None after update logic")
        return self.hessian / self.alpha
    
    def _get_current_fmax(self) -> Optional[float]:
        """Get current maximum force magnitude."""
        try:
            forces = self.atoms.get_forces()
            return np.max(np.abs(forces))
        except Exception:
            return None

    def callback(self, x: np.ndarray, **kwargs) -> None:
        """
        Callback function called by SciPy after each iteration.

        Parameters
        ----------
        x : np.ndarray
            Current position array.
        """
        # Update positions
        self.atoms.set_positions(self._x_to_positions(x))

        # Get forces and log
        forces = self.atoms.get_forces()
        fmax = np.max(np.abs(forces))
        
        # Store for adaptive Hessian updates
        self._previous_fmax = fmax
        
        self.log(forces)
        self.call_observers()

        # Increment step counter
        if self.nsteps < self.max_steps:
            self.nsteps += 1

        # Check convergence - converged() expects 1D gradient array
        forces_flat = forces.ravel()
        if self.converged(forces_flat):
            raise Converged

    def run(self, fmax: float = 0.05, steps: int = 100) -> bool:
        """
        Run the optimization.

        Parameters
        ----------
        fmax : float
            Maximum force convergence criterion (eV/Å).
        steps : int
            Maximum number of optimization steps.

        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        self.fmax = fmax
        self.max_steps = steps + self.nsteps

        # Get initial position
        x0 = self._positions_to_x()

        # Log initial state if first step
        if self.nsteps == 0:
            forces = self.atoms.get_forces()
            self.log(forces)
            self.call_observers()

        logger.info(f"Starting {self.method} optimization")
        logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")
        logger.info(f"Maximum steps: {steps}")

        try:
            # Set up options for SciPy minimize
            options = {
                "maxiter": steps,
                "disp": False,  # We handle logging ourselves
            }

            # Run SciPy optimization
            self._scipy_result = minimize(
                fun=self.objective,
                x0=x0,
                method=self.method,
                jac=self.gradient,
                hess=self.hessian_func,
                callback=self.callback,
                options=options,
            )

            # Update final positions
            self.atoms.set_positions(self._x_to_positions(self._scipy_result.x))

        except Converged:
            logger.info("Optimization converged!")
            return True

        # Check final convergence
        forces = self.atoms.get_forces()
        forces_flat = forces.ravel()
        converged = self.converged(forces_flat)

        if converged:
            logger.info("Optimization converged!")
        else:
            logger.warning(
                f"Optimization stopped after {steps} steps without converging"
            )
            logger.warning(f"Final max force: {np.max(np.abs(forces)):.6f} eV/Å")

        return converged

    def get_number_of_steps(self) -> int:
        """Get the number of optimization steps taken."""
        return self.nsteps

    def dump(self, data):
        """Dump optimizer state (not implemented for SciPy optimizers)."""
        pass

    def load(self):
        """Load optimizer state (not implemented for SciPy optimizers)."""
        pass


class TrustKrylov(SciPyHessianOptimizer):
    """
    Trust-Krylov optimizer for ASE.

    This optimizer uses SciPy's trust-krylov method, which is a trust-region
    algorithm that uses a Krylov subspace to approximately solve the
    trust-region subproblem. It's particularly good for:
    
    - Transition state searches (handles indefinite Hessians)
    - Large systems (doesn't require storing full factorization)
    - Challenging potential energy surfaces

    The Hessian is computed using QME's FrequencyAnalysis, which leverages
    efficient batch evaluation or finite differences suitable for ML potentials.

    Parameters
    ----------
    atoms : Atoms
        The Atoms object to optimize.
    logfile : Union[IO, str]
        File object or filename for logging. Use '-' for stdout.
    trajectory : Optional[str]
        Trajectory file to store optimization path.
    hessian_update_freq : int
        Base frequency of full Hessian recalculation (in steps). Default is 5.
        With adaptive_hessian=True, this is the periodic update interval.
        With adaptive_hessian=False, this is the fixed update interval.
    hessian_method : str
        Method for Hessian calculation: 'auto', 'batch', 'finite_differences', 'direct'
    hessian_delta : float
        Step size for finite difference Hessian (Å). Default is 0.01.
    initial_hessian : Optional[np.ndarray]
        Initial Hessian matrix. If None, computed on first step.
    use_bfgs_update : bool
        Use BFGS approximate updates between full Hessians. Default is True.
    adaptive_hessian : bool
        Adapt update frequency based on convergence. Default is True.
    **kwargs
        Additional arguments passed to Optimizer base class.

    Example
    -------
    >>> from ase.build import molecule
    >>> from qme.core.scipy_optimizers import TrustKrylov
    >>> atoms = molecule('H2O')
    >>> # atoms.calc = ... (attach your calculator)
    >>> opt = TrustKrylov(atoms, trajectory='opt.traj')
    >>> opt.run(fmax=0.05, steps=100)
    """

    def __init__(
        self,
        atoms: Atoms,
        restart=None,
        logfile: Union[IO, str] = "-",
        trajectory: Optional[str] = None,
        maxstep=0.2,
        master=None,
        force_consistent=None,
        hessian_method: str = "auto",
        hessian_update_freq: Optional[int] = None,
        use_bfgs_update: bool = False,
        adaptive_hessian: bool = False,
        force_threshold_ratio: float = 2.0,
    ):
        """Initialize Trust-Krylov optimizer.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to optimize
        hessian_method : str, optional
            Method for Hessian calculation: 'auto', 'fd', 'direct', 'batch'
        hessian_update_freq : int or None, optional
            How often to recompute the full Hessian (in steps).
            If None, compute only once at the beginning.
            Default is None (single Hessian at start).
        use_bfgs_update : bool, optional
            Use BFGS approximate updates between full Hessians.
            Default is False.
        adaptive_hessian : bool, optional
            Enable adaptive Hessian update frequency based on force changes.
            Default is False.
        force_threshold_ratio : float, optional
            Ratio threshold for adaptive Hessian updates. If forces increase
            by more than this ratio, trigger a full Hessian update.
            Default is 2.0.
        """
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=master,
            force_consistent=force_consistent,
        )
        self.method = "trust-krylov"
        self.hessian_method = hessian_method
        self.hessian_update_freq = hessian_update_freq
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio


class TrustNCG(SciPyHessianOptimizer):
    """
    Trust-NCG (Newton Conjugate Gradient) optimizer for ASE.

    Uses SciPy's trust-ncg method, which is a trust-region algorithm
    that uses Newton's conjugate gradient method to approximately solve
    the trust-region subproblem.

    Parameters
    ----------
    See TrustKrylov for parameter documentation.
    """

    def __init__(
        self,
        atoms: Atoms,
        restart=None,
        logfile: Union[IO, str] = "-",
        trajectory: Optional[str] = None,
        maxstep=0.2,
        master=None,
        force_consistent=None,
        hessian_method: str = "auto",
        hessian_update_freq: Optional[int] = None,
        use_bfgs_update: bool = False,
        adaptive_hessian: bool = False,
        force_threshold_ratio: float = 2.0,
    ):
        """Initialize Trust-NCG optimizer.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to optimize
        hessian_method : str, optional
            Method for Hessian calculation: 'auto', 'fd', 'direct', 'batch'
        hessian_update_freq : int or None, optional
            How often to recompute the full Hessian (in steps).
            If None, compute only once at the beginning.
            Default is None (single Hessian at start).
        use_bfgs_update : bool, optional
            Use BFGS approximate updates between full Hessians.
            Default is False.
        adaptive_hessian : bool, optional
            Enable adaptive Hessian update frequency based on force changes.
            Default is False.
        force_threshold_ratio : float, optional
            Ratio threshold for adaptive Hessian updates. If forces increase
            by more than this ratio, trigger a full Hessian update.
            Default is 2.0.
        """
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=master,
            force_consistent=force_consistent,
        )
        self.method = "trust-ncg"
        self.hessian_method = hessian_method
        self.hessian_update_freq = hessian_update_freq
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio


class TrustExact(SciPyHessianOptimizer):
    """
    Trust-Exact optimizer for ASE.

    Uses SciPy's trust-exact method, which is a trust-region algorithm
    that uses a nearly exact solver for the trust-region subproblem.
    Most accurate but also most computationally expensive.

    Parameters
    ----------
    See TrustKrylov for parameter documentation.
    """

    def __init__(
        self,
        atoms: Atoms,
        restart=None,
        logfile: Union[IO, str] = "-",
        trajectory: Optional[str] = None,
        maxstep=0.2,
        master=None,
        force_consistent=None,
        hessian_method: str = "auto",
        hessian_update_freq: Optional[int] = None,
        use_bfgs_update: bool = False,
        adaptive_hessian: bool = False,
        force_threshold_ratio: float = 2.0,
    ):
        """Initialize Trust-Exact optimizer.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to optimize
        hessian_method : str, optional
            Method for Hessian calculation: 'auto', 'fd', 'direct', 'batch'
        hessian_update_freq : int or None, optional
            How often to recompute the full Hessian (in steps).
            If None, compute only once at the beginning.
            Default is None (single Hessian at start).
        use_bfgs_update : bool, optional
            Use BFGS approximate updates between full Hessians.
            Default is False.
        adaptive_hessian : bool, optional
            Enable adaptive Hessian update frequency based on force changes.
            Default is False.
        force_threshold_ratio : float, optional
            Ratio threshold for adaptive Hessian updates. If forces increase
            by more than this ratio, trigger a full Hessian update.
            Default is 2.0.
        """
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=master,
            force_consistent=force_consistent,
        )
        self.method = "trust-exact"
        self.hessian_method = hessian_method
        self.hessian_update_freq = hessian_update_freq
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio


class NewtonCG(SciPyHessianOptimizer):
    """
    Newton-CG (Newton Conjugate Gradient) optimizer for ASE.

    Uses SciPy's Newton-CG method, which is Newton's method where
    the linear system is solved using conjugate gradients.

    Parameters
    ----------
    See TrustKrylov for parameter documentation.
    """

    def __init__(
        self,
        atoms: Atoms,
        restart=None,
        logfile: Union[IO, str] = "-",
        trajectory: Optional[str] = None,
        maxstep=0.2,
        master=None,
        force_consistent=None,
        hessian_method: str = "auto",
        hessian_update_freq: Optional[int] = None,
        use_bfgs_update: bool = False,
        adaptive_hessian: bool = False,
        force_threshold_ratio: float = 2.0,
    ):
        """Initialize Newton-CG optimizer.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to optimize
        hessian_method : str, optional
            Method for Hessian calculation: 'auto', 'fd', 'direct', 'batch'
        hessian_update_freq : int or None, optional
            How often to recompute the full Hessian (in steps).
            If None, compute only once at the beginning.
            Default is None (single Hessian at start).
        use_bfgs_update : bool, optional
            Use BFGS approximate updates between full Hessians.
            Default is False.
        adaptive_hessian : bool, optional
            Enable adaptive Hessian update frequency based on force changes.
            Default is False.
        force_threshold_ratio : float, optional
            Ratio threshold for adaptive Hessian updates. If forces increase
            by more than this ratio, trigger a full Hessian update.
            Default is 2.0.
        """
        super().__init__(
            atoms,
            restart=restart,
            logfile=logfile,
            trajectory=trajectory,
            maxstep=maxstep,
            master=master,
            force_consistent=force_consistent,
        )
        self.method = "Newton-CG"
        self.hessian_method = hessian_method
        self.hessian_update_freq = hessian_update_freq
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio
