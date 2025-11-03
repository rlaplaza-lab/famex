"""SciPy-based optimizers with Hessian support for QME.

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

from typing import IO, Any

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer
from scipy.optimize import minimize

# FrequencyAnalysis imported locally to avoid circular imports
from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


class ConvergedError(Exception):
    """Exception raised when optimizer has converged."""


class SciPyHessianOptimizer(Optimizer):
    """Base class for SciPy optimizers that use Hessian information.

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
    hessian_update_freq : Optional[int]
        Optional frequency of full Hessian recalculation (in steps).
        Default is None, which only computes the Hessian once at the beginning.
        Set to an integer to recompute every N steps.
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
        Default is False. Enable to trigger additional Hessian evaluations
        when forces increase sharply.
    force_threshold_ratio : float
        Ratio of force increase that triggers a full Hessian update.
        Default is 2.0. Only used if adaptive_hessian=True.
    verbose : int
        Verbosity level for optimization output:
        - 0: Quiet (minimal output)
        - 1: Normal (default, shows progress)
        - 2: Verbose (detailed information)
    **kwargs
        Additional arguments passed to Optimizer base class.

    Attributes:
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
        atoms: Atoms,
        method: str = "trust-krylov",
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        alpha: float = 1.0,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        force_threshold_ratio: float = 2.0,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize SciPy Hessian-based optimizer."""
        # Store verbosity level
        self.verbose = verbose

        # Set up logging based on verbosity
        if verbose == 0:
            # Quiet mode: suppress ASE logging by using None logfile
            logfile = None
        elif verbose == 1:
            # Normal mode: use provided logfile or default
            pass
        else:
            # Verbose mode: use provided logfile or default
            pass

        # Don't use restart for SciPy optimizers
        restart = None
        # Filter out profiler argument as ASE Optimizer doesn't accept it
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "profiler"}
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **filtered_kwargs)

        self.method = method
        freq = hessian_update_freq
        if freq is not None and freq <= 0:
            logger.warning("hessian_update_freq <= 0 provided; disabling periodic Hessian updates")
            freq = None

        self.hessian_update_freq: int | None = freq
        self.hessian_method = hessian_method
        self.alpha = alpha
        self.use_bfgs_update = use_bfgs_update
        self.adaptive_hessian = adaptive_hessian
        self.force_threshold_ratio = force_threshold_ratio

        # Initialize FrequencyAnalysis for Hessian computation
        if atoms.calc is None:
            msg = "Atoms object must have a calculator attached"
            raise ValueError(msg)

        # Import locally to avoid circular imports
        from qme.analysis.frequency import FrequencyAnalysis

        self.freq_analysis = FrequencyAnalysis(
            atoms=atoms,
            calculator=atoms.calc,
            delta=hessian_delta,
            verbose=verbose,
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
            msg = f"Invalid method '{method}'. Valid methods: {valid_methods}"
            raise ValueError(msg)

        if self.verbose >= 2:
            logger.info(f"Initialized {method} optimizer with Hessian support")
            if adaptive_hessian:
                if hessian_update_freq is None:
                    logger.info(
                        "Adaptive Hessian updates enabled (force-triggered; no periodic updates)",
                    )
                else:
                    logger.info(
                        f"Adaptive Hessian updates enabled (base frequency: every {hessian_update_freq} step(s))",
                    )
            elif hessian_update_freq is None:
                logger.info("Periodic Hessian updates disabled (compute once and reuse)")
            else:
                logger.info(
                    f"Fixed Hessian update frequency: every {hessian_update_freq} step(s)",
                )
            if use_bfgs_update:
                logger.info("BFGS approximate updates enabled between full Hessians")
            logger.info(f"Hessian calculation method: {hessian_method}")

    def _positions_to_x(self, atoms: Atoms | None = None) -> np.ndarray:
        """Convert atoms positions to 1D array for SciPy."""
        if atoms is None:
            atoms = self.atoms
        if atoms is None:  # type: ignore[unreachable]
            msg = "Atoms object is not initialized"
            raise RuntimeError(msg)
        return atoms.get_positions().ravel()  # type: ignore[no-any-return]

    def _x_to_positions(self, x: np.ndarray) -> np.ndarray:
        """Convert 1D array to positions array."""
        return x.reshape(-1, 3)

    def objective(self, x: np.ndarray) -> float:
        """Objective function for minimization (potential energy).

        Parameters
        ----------
        x : np.ndarray
            Flattened position array.

        Returns:
        -------
        float
            Potential energy scaled by alpha.

        """
        self.atoms.set_positions(self._x_to_positions(x))
        energy = self.atoms.get_potential_energy()
        # Scale by alpha (Hessian scaling factor)
        return energy / self.alpha  # type: ignore[no-any-return]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of objective function (negative forces).

        Parameters
        ----------
        x : np.ndarray
            Flattened position array.

        Returns:
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
        return gradient / self.alpha  # type: ignore[no-any-return]

    def hessian_func(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix with adaptive updates and BFGS approximation.

        This uses QME's FrequencyAnalysis to compute the Hessian efficiently.
        Supports:
        - Adaptive update frequency based on convergence behavior
        - BFGS approximate updates between full Hessian calculations
        - Caching and periodic updates

        Parameters
        ----------
        x : np.ndarray
            Flattened position array.

        Returns:
        -------
        np.ndarray
            Hessian matrix (3N x 3N), scaled by alpha.

        """
        current_positions = x.copy()
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
            if current_fmax is not None and self._previous_fmax is not None:  # type: ignore[unreachable]
                if current_fmax > self._previous_fmax * self.force_threshold_ratio:
                    need_full_update = True
                    reason = f"force increase ({current_fmax:.4f} > {self._previous_fmax:.4f} × {self.force_threshold_ratio})"

            # Periodic update based on base frequency
            if (
                not need_full_update
                and self.hessian_update_freq is not None
                and steps_since_full >= self.hessian_update_freq
            ):
                need_full_update = True
                reason = f"periodic (every {self.hessian_update_freq} steps)"
        # Fixed frequency mode
        elif self.hessian_update_freq is not None and steps_since_full >= self.hessian_update_freq:
            need_full_update = True
            reason = f"fixed frequency ({self.hessian_update_freq} steps)"

        if need_full_update:
            # Compute full Hessian
            self.atoms.set_positions(self._x_to_positions(x))
            self.hessian_calls += 1
            self._last_hessian_step = self.nsteps
            self._last_full_hessian_step = self.nsteps

            if self.verbose >= 1:  # Changed to >= 1 to always show
                logger.info(
                    f"Computing full Hessian at step {self.nsteps} "
                    f"(call #{self.hessian_calls}, reason: {reason})",
                )

            # Update FrequencyAnalysis with current atoms state
            self.freq_analysis.atoms = self.atoms
            self.freq_analysis.atoms.calc = self.atoms.calc

            # Compute Hessian using QME's infrastructure
            hessian = self.freq_analysis.calculate_hessian(method=self.hessian_method)

            # Store for potential reuse
            self.hessian = hessian

            if self.verbose >= 2:
                logger.info(f"Full Hessian computed (shape: {hessian.shape})")

            # Reset BFGS state after full Hessian
            self._last_positions = current_positions.copy()
            self._last_gradient = None
            self.bfgs_updates = 0

        elif (  # type: ignore[unreachable]
            self.use_bfgs_update
            and self._last_positions is not None
            and self._last_gradient is not None
        ):
            # BFGS approximate update
            self._last_hessian_step = self.nsteps

            # Get current gradient
            current_gradient = self.gradient(x)

            # Compute differences
            s = current_positions - self._last_positions  # position change
            y = current_gradient - self._last_gradient  # gradient change

            # BFGS update formula: H_new = H + (y⊗y)/(y·s) - (H·s⊗H·s)/(s·H·s)
            sy = np.dot(s, y)

            if sy > 1e-10:  # Ensure positive definiteness
                # Rank-2 update (BFGS formula)
                if self.hessian is None:
                    msg = "Hessian unavailable during BFGS update"
                    raise RuntimeError(msg)
                Hs = np.dot(self.hessian, s)
                sHs = np.dot(s, Hs)

                # Update Hessian
                self.hessian = self.hessian + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs

                self.bfgs_updates += 1
                logger.debug(
                    f"BFGS Hessian update at step {self.nsteps} "
                    f"(#{self.bfgs_updates} since last full, "
                    f"{steps_since_full} steps since full Hessian)",
                )
            else:
                logger.debug(f"Skipping BFGS update (sy = {sy:.2e} too small)")

            # Store current state for next update
            self._last_positions = current_positions.copy()
            self._last_gradient = current_gradient.copy()
        else:
            # Reuse cached Hessian
            if self.verbose >= 1:  # Changed to >= 1 to always show
                logger.info(f"Reusing Hessian (computed {steps_since_full} steps ago)")
            else:
                logger.debug(f"Reusing Hessian (computed {steps_since_full} steps ago)")

        # Scale by alpha - hessian should never be None at this point
        if self.hessian is None:
            msg = "Hessian is None after update logic"
            raise RuntimeError(msg)
        return self.hessian / self.alpha

    def _get_current_fmax(self) -> float | None:
        """Get current maximum force magnitude."""
        try:
            forces = self.atoms.get_forces()
            return np.max(np.abs(forces))  # type: ignore[no-any-return]
        except Exception:
            return None

    def callback(self, x: np.ndarray, **kwargs: Any) -> None:
        """Callback function called by SciPy after each iteration.

        Parameters
        ----------
        x : np.ndarray
            Current position array.

        """
        # Update positions
        self.atoms.set_positions(self._x_to_positions(x))

        # Increment step counter BEFORE logging to match printed step numbers
        if self.nsteps < self.max_steps:
            self.nsteps += 1

        # Get forces and log
        forces = self.atoms.get_forces()
        fmax = np.max(np.abs(forces))

        # Store for adaptive Hessian updates
        self._previous_fmax = fmax

        self.log(forces)
        self.call_observers()

        # Check convergence - converged() expects 1D gradient array
        forces_flat = forces.ravel()
        if self.converged(forces_flat):
            raise ConvergedError

    def run(self, fmax: float = 0.05, steps: int = 100) -> bool:
        """Run the optimization.

        Parameters
        ----------
        fmax : float
            Maximum force convergence criterion (eV/Å).
        steps : int
            Maximum number of optimization steps.

        Returns:
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
            # Count the initial step to match printed step numbers
            self.nsteps += 1

        if self.verbose >= 2:
            logger.info(f"Starting {self.method} optimization")
            logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")
            logger.info(f"Maximum steps: {steps}")

        try:
            # Set up options for SciPy minimize
            # For trust-region methods, we can control convergence more tightly
            options = {
                "maxiter": steps,
                "disp": False,  # We handle logging ourselves
            }

            # Add trust-region specific options for better convergence
            # These help prevent early termination when close to convergence
            if self.method in ("trust-krylov", "trust-ncg", "trust-exact"):
                # gtol: Gradient norm tolerance for convergence (default is 1e-5)
                # Set to very tight value to prevent premature stopping based on gradient norm
                # Our callback handles force-based convergence checking
                options["gtol"] = 1e-8  # Very tight gradient tolerance to allow more iterations
                # initial_tr_radius: Initial trust region radius (default is 1.0)
                # For TS optimization, we may need a larger trust region to allow climbing
                # Try increasing slightly if convergence is problematic
                options["initial_tr_radius"] = 1.0  # Default, but explicit
                # max_tr_radius: Maximum trust region radius (default is infinity)
                # Allow larger trust regions if needed
                options["max_tr_radius"] = 10.0  # Increase from default to allow more exploration

            # Run SciPy optimization
            self._scipy_result = minimize(  # type: ignore[call-overload]
                fun=self.objective,
                x0=x0,
                method=self.method,
                jac=self.gradient,
                hess=self.hessian_func,
                callback=self.callback,
                options=options,
            )

            # Update final positions
            if self._scipy_result is not None:
                self.atoms.set_positions(self._x_to_positions(self._scipy_result.x))
        except ConvergedError:
            if self.verbose >= 1:
                logger.info("Optimization converged!")
            return True

        # Check final convergence
        forces = self.atoms.get_forces()
        forces_flat = forces.ravel()
        converged = self.converged(forces_flat)

        if converged:
            if self.verbose >= 1:
                logger.info("Optimization converged!")
        elif self.verbose >= 1:
            actual_steps = self.nsteps - 1  # Subtract initial step count
            logger.warning(
                f"Optimization stopped after {actual_steps} steps without converging "
                f"(max optimization steps: {steps})"
            )
            logger.warning(f"Final max force: {np.max(np.abs(forces)):.6f} eV/Å")

        return converged

    def get_number_of_steps(self) -> int:
        """Get the number of optimization steps taken."""
        return self.nsteps

    def dump(self, data: Any) -> None:
        """Dump optimizer state (not implemented for SciPy optimizers)."""

    def load(self) -> None:
        """Load optimizer state (not implemented for SciPy optimizers)."""


class TrustKrylov(SciPyHessianOptimizer):
    """Trust-Krylov optimizer for ASE.

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
    hessian_update_freq : Optional[int]
        Base frequency of full Hessian recalculation (in steps). Default is None,
        which only computes the Hessian once at the beginning. Set to an integer
        to recompute every N steps.
    hessian_method : str
        Method for Hessian calculation: 'auto', 'batch', 'finite_differences', 'direct'
    hessian_delta : float
        Step size for finite difference Hessian (Å). Default is 0.01.
    initial_hessian : Optional[np.ndarray]
        Initial Hessian matrix. If None, computed on first step.
    use_bfgs_update : bool
        Use BFGS approximate updates between full Hessians. Default is True.
    adaptive_hessian : bool
        Adapt update frequency based on convergence. Default is False.
    verbose : int
        Verbosity level for optimization output:
        - 0: Quiet (minimal output)
        - 1: Normal (default, shows progress)
        - 2: Verbose (detailed information)
    **kwargs
        Additional arguments passed to Optimizer base class.

    Example:
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
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Trust-Krylov optimizer."""
        super().__init__(
            atoms=atoms,
            method="trust-krylov",
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )


class TrustKrylovTS(TrustKrylov):
    """Trust-Krylov transition state optimizer.

    This variant of :class:`TrustKrylov` modifies the trust-region model
    so that one direction is treated as an ascent direction while all
    orthogonal directions remain minimization directions. The class follows
    a min-mode following approach similar in spirit to Sella: the lowest
    Hessian eigenvector is tracked and reflected so that the optimization
    converges to an index-1 saddle point.

    Notes:
    -----
    The implementation reflects both the gradient and the Hessian along the
    tracked mode. In practice this means the SciPy trust region solver sees a
    locally convex model and can reuse the existing ``trust-krylov`` machinery
    without invasive changes. Additional safeguards make sure no extra
    negative curvature directions creep into the model, which would otherwise
    destabilise the saddle search.

    """

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = False,
        adaptive_hessian: bool = False,
        mode_recompute_interval: int = 1,
        index_tolerance: float = 5e-4,
        min_positive_eigenvalue: float = 4e-3,
        negative_mode_boost: float = 8e-3,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialise the transition-state Trust-Krylov optimizer."""
        super().__init__(
            atoms=atoms,
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )

        self._ts_mode_vector: np.ndarray | None = None
        self._ts_mode_eigenvalue: float | None = None
        self._ts_last_raw_gradient: np.ndarray | None = None
        self._ts_last_raw_hessian: np.ndarray | None = None
        self._ts_last_mode_step: int = -1
        self._ts_mode_recompute_interval = max(1, mode_recompute_interval)
        self._ts_index_tolerance = index_tolerance
        self._ts_min_positive_eig = min_positive_eigenvalue
        self._ts_negative_mode_boost = negative_mode_boost
        self._ts_degrees_of_freedom = atoms.get_positions().size
        self._ts_manual_mode_override = False

        # Step quality tracking for trust region adaptation (inspired by geomeTRIC)
        self._ts_previous_energy: float | None = None
        self._ts_previous_positions: np.ndarray | None = None
        self._ts_previous_gradient: np.ndarray | None = None
        self._ts_previous_hessian: np.ndarray | None = (
            None  # Stabilized Hessian (what trust-krylov uses)
        )
        self._ts_previous_hessian_raw: np.ndarray | None = None  # Raw Hessian (for reference)
        self._ts_step_quality_history: list[float] = []

        # P-RFO parameters (similar to RFO optimizer)
        self._ts_trust_radius: float = 0.01  # Initial trust radius for TS (geomeTRIC default)
        self._ts_max_trust_radius: float = 0.03  # Maximum trust radius
        self._ts_min_trust_radius: float = 0.001  # Minimum trust radius
        self._ts_alpha: float = 1.0  # Trust radius control parameter for P-RFO

    # ------------------------------------------------------------------
    # Public helpers for seeding / inspecting the tracked transition mode
    # ------------------------------------------------------------------
    def set_transition_mode(self, mode: np.ndarray, eigenvalue: float | None = None) -> None:
        """Seed the tracked transition mode manually.

        Parameters
        ----------
        mode
            Vector with the same dimensionality as the atomic coordinates (3N).
        eigenvalue
            Optional curvature estimate along the provided mode. If omitted,
            a small negative curvature is assumed to maintain the saddle model.

        """
        vector = np.asarray(mode, dtype=float).reshape(-1)
        if vector.size != self._ts_degrees_of_freedom:
            msg = f"Expected mode of length {self._ts_degrees_of_freedom}, got {vector.size}"
            raise ValueError(
                msg,
            )

        norm = np.linalg.norm(vector)
        if norm < 1e-12:
            msg = "Mode vector must have non-zero norm"
            raise ValueError(msg)

        self._ts_mode_vector = vector / norm
        if eigenvalue is None:
            eigenvalue = -abs(self._ts_negative_mode_boost)
        self._ts_mode_eigenvalue = float(eigenvalue)
        self._ts_last_mode_step = self.nsteps
        self._ts_manual_mode_override = True

    def get_transition_mode(self) -> np.ndarray | None:
        """Return a copy of the current transition mode vector, if available."""
        if self._ts_mode_vector is None:
            return None
        return self._ts_mode_vector.copy()

    def get_transition_mode_info(self) -> dict[str, Any]:
        """Return diagnostic information about the tracked transition mode."""
        return {
            "mode": None if self._ts_mode_vector is None else self._ts_mode_vector.copy(),
            "eigenvalue": self._ts_mode_eigenvalue,
            "last_update_step": self._ts_last_mode_step,
            "age": None if self._ts_last_mode_step < 0 else self.nsteps - self._ts_last_mode_step,
            "manual_override": self._ts_manual_mode_override,
        }

    # ------------------------------------------------------------------
    # Helper utilities for mode tracking / reflection
    # ------------------------------------------------------------------
    def _should_update_mode(self) -> bool:
        """Determine if the transition mode should be recomputed.

        Avoids unnecessary updates when the mode is stable or when
        we're making good progress.
        """
        if self._ts_mode_vector is None:
            return True

        # Always update if enough steps have passed
        if self.nsteps - self._ts_last_mode_step >= self._ts_mode_recompute_interval:
            self._ts_manual_mode_override = False
            return True

        # Don't update if manual override is active
        if self._ts_manual_mode_override:
            return False

        return False

    def _update_mode_from_hessian(self, hessian: np.ndarray) -> None:
        """Update the tracked negative mode from the latest Hessian."""
        sym_hessian = 0.5 * (hessian + hessian.T)

        # Check condition number to detect numerical issues
        try:
            cond_num = np.linalg.cond(sym_hessian)
            if cond_num > 1e12:
                if self.verbose >= 2:
                    logger.warning(
                        f"Ill-conditioned Hessian (cond={cond_num:.2e}) detected in mode update. "
                        "Using regularization."
                    )
                # Add small regularization to improve condition number
                n = sym_hessian.shape[0]
                reg_factor = np.trace(sym_hessian) / n * 1e-10
                sym_hessian = sym_hessian + reg_factor * np.eye(n)
        except (np.linalg.LinAlgError, ValueError):
            pass  # Continue with original Hessian

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)
        except np.linalg.LinAlgError as exc:
            logger.warning(f"Failed to diagonalize Hessian: {exc}")
            # Fallback: try to use previous mode or gradient
            if self._ts_mode_vector is not None:
                return  # Keep previous mode
            gradient = self._ts_last_raw_gradient
            if gradient is not None and np.linalg.norm(gradient) > 1e-12:
                self._ts_mode_vector = gradient / np.linalg.norm(gradient)
                self._ts_mode_eigenvalue = -self._ts_negative_mode_boost
                self._ts_last_mode_step = self.nsteps
            else:
                self._ts_mode_vector = None
                self._ts_mode_eigenvalue = None
            return

        min_index = int(np.argmin(eigenvalues))
        min_value = float(eigenvalues[min_index])
        mode_vector = eigenvectors[:, min_index].copy()

        if min_value >= -self._ts_index_tolerance:
            # If Hessian is effectively positive definite, we're near a minimum.
            # Use a better approach: find the direction of steepest ascent
            # by projecting gradient onto the subspace of smallest eigenvalues
            gradient = self._ts_last_raw_gradient
            if gradient is None or np.linalg.norm(gradient) < 1e-12:
                # No gradient available, use the smallest eigenvector as escape direction
                if self.verbose >= 2:
                    logger.info(
                        "Positive definite Hessian with zero gradient, using smallest eigenvector"
                    )
                mode_vector = eigenvectors[:, min_index].copy()
                min_value = -self._ts_negative_mode_boost
            else:
                # Project gradient onto eigenvectors with smallest eigenvalues
                # This gives us the direction of maximum ascent in the flattest direction
                n_flat = min(3, len(eigenvalues))  # Consider up to 3 flattest directions
                flat_indices = np.argsort(eigenvalues)[:n_flat]
                flat_subspace = eigenvectors[:, flat_indices]

                # Project gradient onto flat subspace
                grad_proj = flat_subspace.T @ gradient
                if np.linalg.norm(grad_proj) > 1e-12:
                    mode_vector = flat_subspace @ grad_proj
                    mode_vector = mode_vector / np.linalg.norm(mode_vector)
                else:
                    # If projection is zero, use smallest eigenvector
                    mode_vector = eigenvectors[:, min_index].copy()
                min_value = -self._ts_negative_mode_boost

        # Normalize mode vector
        mode_norm = np.linalg.norm(mode_vector)
        if mode_norm < 1e-12:
            logger.warning("Mode vector has zero norm, using fallback")
            # Try to get gradient for fallback
            fallback_gradient = self._ts_last_raw_gradient
            if fallback_gradient is not None and np.linalg.norm(fallback_gradient) > 1e-12:
                mode_vector = fallback_gradient / np.linalg.norm(fallback_gradient)
            else:
                self._ts_mode_vector = None
                self._ts_mode_eigenvalue = None
                return
        mode_vector = mode_vector / np.linalg.norm(mode_vector)

        self._ts_mode_vector = mode_vector
        self._ts_mode_eigenvalue = min_value
        self._ts_last_mode_step = self.nsteps
        self._ts_manual_mode_override = False

    @staticmethod
    def _reflect_along_mode(vector: np.ndarray, mode: np.ndarray) -> np.ndarray:
        """Reflect *vector* along *mode* (Householder reflection)."""
        mode_normalized = mode / np.linalg.norm(mode)
        return vector - 2.0 * np.dot(vector, mode_normalized) * mode_normalized

    def _compute_prfo_step(
        self, gradient: np.ndarray, hessian: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, float]:
        """Compute P-RFO step for transition state optimization.

        Following geomeTRIC's Partitioned RFO (P-RFO) approach:
        1. Diagonalize Hessian to find transition mode (lowest eigenvalue)
        2. For transition mode: solve 2x2 RFO problem, select highest eigenvalue for maximization
        3. For other modes: minimize using projected RFO formula
        4. Combine steps: y = w_tv * y~_tv + sum(w_k * y~_k)

        This provides explicit partitioning unlike the reflection approach.

        Parameters
        ----------
        gradient : np.ndarray
            Raw gradient vector (negative forces)
        hessian : np.ndarray
            Raw Hessian matrix
        alpha : float
            Trust radius control parameter

        Returns:
        -------
        step : np.ndarray
            P-RFO step vector
        lambda_max : float
            Highest eigenvalue from transition mode RFO problem
        """
        n = len(gradient)

        # Step 1: Diagonalize Hessian to find transition mode
        sym_hessian = 0.5 * (hessian + hessian.T)

        try:
            hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(sym_hessian)
        except np.linalg.LinAlgError as exc:
            logger.error(f"Failed to diagonalize Hessian for P-RFO: {exc}")
            raise

        # Find transition mode (lowest eigenvalue)
        tv_index = int(np.argmin(hessian_eigenvalues))
        omega_tv = float(hessian_eigenvalues[tv_index])
        w_tv = hessian_eigenvectors[:, tv_index].copy()

        # Normalize transition mode
        w_tv_norm = np.linalg.norm(w_tv)
        if w_tv_norm < 1e-12:
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-12:
                w_tv = -gradient / grad_norm
            else:
                w_tv = hessian_eigenvectors[:, 0] / np.linalg.norm(hessian_eigenvectors[:, 0])
        else:
            w_tv = w_tv / w_tv_norm

        # Update tracked mode
        self._ts_mode_vector = w_tv.copy()
        self._ts_mode_eigenvalue = omega_tv
        self._ts_last_mode_step = self.nsteps

        # Step 2: Compute step along transition mode using 2x2 RFO problem
        g_tilde_tv = np.dot(gradient, w_tv)

        # Build 2x2 RFO matrix for transition mode
        rfo_tv_matrix = np.array([[0.0, g_tilde_tv], [g_tilde_tv, omega_tv]])
        metric_tv = np.array([[1.0, 0.0], [0.0, alpha]])

        # Solve 2x2 generalized eigenvalue problem
        try:
            from scipy.linalg import eigh as scipy_eigh

            tv_eigenvalues, tv_eigenvectors = scipy_eigh(rfo_tv_matrix, metric_tv)
        except (ImportError, np.linalg.LinAlgError):
            metric_tv_inv = np.linalg.inv(metric_tv)
            standard_tv = metric_tv_inv @ rfo_tv_matrix
            tv_eigenvalues, tv_eigenvectors = np.linalg.eig(standard_tv)
            tv_eigenvalues = np.real(tv_eigenvalues)
            tv_eigenvectors = np.real(tv_eigenvectors)

        # Select HIGHEST eigenvalue for maximization (TS climbing)
        tv_max_idx = int(np.argmax(tv_eigenvalues))
        lambda_tv_max = float(tv_eigenvalues[tv_max_idx])
        v_tv = tv_eigenvectors[:, tv_max_idx]

        # Extract step along transition mode
        v0_tv = v_tv[0]
        if abs(v0_tv) > 1e-12:
            y_tilde_tv = v_tv[1] / v0_tv
        else:
            denominator = omega_tv - alpha * lambda_tv_max
            if abs(denominator) > 1e-12:
                y_tilde_tv = -g_tilde_tv / denominator
            else:
                y_tilde_tv = 0.0

        # Step 3: Compute steps along other modes (minimization)
        tv_min_idx = int(np.argmin(tv_eigenvalues))
        lambda_tv_min = float(tv_eigenvalues[tv_min_idx])

        # Project gradient onto all Hessian eigenvectors
        g_tilde_all = np.dot(hessian_eigenvectors.T, gradient)

        # Compute steps in eigenbasis
        y_tilde_all = np.zeros(n)
        for k in range(n):
            omega_k = hessian_eigenvalues[k]
            g_tilde_k = g_tilde_all[k]

            if k == tv_index:
                y_tilde_all[k] = y_tilde_tv
            else:
                denominator = omega_k - alpha * lambda_tv_min
                if abs(denominator) > 1e-12:
                    y_tilde_all[k] = -g_tilde_k / denominator
                else:
                    y_tilde_all[k] = 0.0

        # Step 4: Transform back from eigenbasis to Cartesian coordinates
        step = np.zeros(n)
        for k in range(n):
            step += hessian_eigenvectors[:, k] * y_tilde_all[k]

        # Apply trust radius constraint
        step_norm = np.linalg.norm(step)
        if step_norm > 1e-12:
            if step_norm > self._ts_trust_radius * 1.01:
                step = step / step_norm * self._ts_trust_radius
                step_norm = self._ts_trust_radius
        else:
            if np.linalg.norm(w_tv) > 1e-12:
                step = w_tv * self._ts_trust_radius * 0.1
            else:
                step = np.zeros(n)

        if self.verbose >= 2:
            step_norm_final = np.linalg.norm(step)
            grad_step_dot = np.dot(gradient, step) if step_norm_final > 1e-12 else 0.0
            logger.debug(
                f"P-RFO step: lambda_tv_max={lambda_tv_max:.6f}, omega_tv={omega_tv:.6f}, "
                f"step_norm={step_norm_final:.6f}, grad·step={grad_step_dot:.6f}, alpha={alpha:.6f}"
            )

        return step, lambda_tv_max

    def _find_optimal_alpha(self, gradient: np.ndarray, hessian: np.ndarray) -> float:
        """Find optimal alpha parameter for P-RFO step to match trust radius.

        Uses bisection to find alpha such that ||step(alpha)|| ≈ trust_radius.
        """
        alpha_min = 1e-4
        alpha_max = 1e4

        if 1e-4 < self._ts_alpha < 1e4:
            alpha = self._ts_alpha
        else:
            alpha = 1.0

        max_iter = 20
        tolerance = 0.01 * self._ts_trust_radius

        for _iteration in range(max_iter):
            try:
                step, _ = self._compute_prfo_step(gradient, hessian, alpha)
                step_norm = np.linalg.norm(step)

                error = abs(step_norm - self._ts_trust_radius)
                if error < tolerance:
                    break

                if step_norm > self._ts_trust_radius:
                    alpha_min = alpha
                    if alpha_max < 1e4:
                        alpha = (alpha + alpha_max) / 2.0
                    else:
                        alpha *= 1.5
                else:
                    alpha_max = alpha
                    if alpha_min > 1e-4:
                        alpha = (alpha_min + alpha) / 2.0
                    else:
                        alpha *= 0.8

                alpha = max(1e-4, min(alpha, 1e4))

                if alpha_min >= alpha_max - 1e-10:
                    break
            except Exception as e:
                if self.verbose >= 2:
                    logger.debug(f"Alpha optimization error: {e}")
                break

        self._ts_alpha = alpha
        return alpha

    def _adjust_trust_radius(self, step_quality: float, step_size: float) -> None:
        """Adjust trust radius based on step quality (geomeTRIC approach).

        Following geomeTRIC's approach:
        - Q ≥ 0.75: Good step, increase by √2
        - 0.50 ≤ Q < 0.75: Okay step, unchanged
        - 0 ≤ Q < 0.50: Poor step, decrease
        - Q < 0: Very poor, decrease

        Parameters
        ----------
        step_quality : float
            Step quality factor Q
        step_size : float
            Norm of the step taken
        """
        if step_quality >= 0.75:
            # Good step
            new_trust = self._ts_trust_radius * np.sqrt(2.0)
            self._ts_trust_radius = min(new_trust, self._ts_max_trust_radius)
            if self.verbose >= 2:
                logger.debug(
                    f"Good step (Q={step_quality:.4f}), increased trust radius to {self._ts_trust_radius:.6f}"
                )
        elif step_quality >= 0.50:
            # Okay step - unchanged
            if self.verbose >= 2:
                logger.debug(
                    f"Okay step (Q={step_quality:.4f}), trust radius unchanged: {self._ts_trust_radius:.6f}"
                )
        elif step_quality >= 0.0:
            # Poor step
            new_trust = 0.5 * min(self._ts_trust_radius, step_size)
            self._ts_trust_radius = max(new_trust, self._ts_min_trust_radius)
            if self.verbose >= 1:
                logger.warning(
                    f"Poor step (Q={step_quality:.4f}), decreased trust radius to {self._ts_trust_radius:.6f}"
                )
        else:
            # Very poor step (Q < 0)
            new_trust = 0.5 * min(self._ts_trust_radius, step_size)
            self._ts_trust_radius = max(new_trust, self._ts_min_trust_radius)
            if self.verbose >= 1:
                logger.warning(
                    f"Very poor step (Q={step_quality:.4f}), decreased trust radius to {self._ts_trust_radius:.6f}"
                )

    def _stabilise_hessian(self, hessian: np.ndarray, primary_mode: np.ndarray) -> np.ndarray:
        """Ensure the reflected Hessian is positive definite except for the flipped mode."""
        sym_hessian = 0.5 * (hessian + hessian.T)

        # Check condition number first
        try:
            cond_num = np.linalg.cond(sym_hessian)
            if cond_num > 1e14:
                # Very ill-conditioned, use more aggressive regularization
                n = sym_hessian.shape[0]
                # Use trace-based regularization to improve condition number
                trace_val = np.trace(sym_hessian)
                if trace_val > 0:
                    reg_factor = trace_val / n * max(1e-9, 1.0 / cond_num)
                else:
                    reg_factor = abs(trace_val) / n * 1e-9
                sym_hessian = sym_hessian + reg_factor * np.eye(n)
                if self.verbose >= 2:
                    logger.debug(
                        f"Applied regularization {reg_factor:.2e} due to condition number {cond_num:.2e}"
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)
        except np.linalg.LinAlgError:
            # Fallback: add diagonal ridge with conservative value
            n = sym_hessian.shape[0]
            trace_val = abs(np.trace(sym_hessian))
            ridge = max(self._ts_min_positive_eig, trace_val / n * 1e-9)
            stabilised = sym_hessian + ridge * np.eye(n)
            return 0.5 * (stabilised + stabilised.T)

        # Ensure only the tracked mode carries the flipped signature.
        stabilised = sym_hessian.copy()
        primary_projection = eigenvectors.T @ primary_mode
        primary_index = int(np.argmax(np.abs(primary_projection)))

        # Only correct eigenvalues that are significantly negative (not near zero)
        significant_threshold = max(self._ts_index_tolerance, 1e-6)
        corrections_made = 0

        for idx, value in enumerate(eigenvalues):
            if idx == primary_index:
                continue
            if value < -significant_threshold:
                # More conservative correction: only lift to small positive value
                correction = (-value) + max(self._ts_min_positive_eig, abs(value) * 0.1)
                vec = eigenvectors[:, idx]
                stabilised += correction * np.outer(vec, vec)
                corrections_made += 1

        if self.verbose >= 2 and corrections_made > 0:
            logger.debug(
                f"Corrected {corrections_made} negative eigenvalues (excluding primary mode)"
            )

        # CRITICAL: For TS optimization, after reflection, the mode should have NEGATIVE curvature
        # in the stabilised Hessian so that trust-krylov will CLIMB along this direction.
        # We cannot make it positive or trust-krylov will minimize instead of climb.
        mode_curvature = float(primary_mode @ (stabilised @ primary_mode))

        # After reflection, if the original Hessian had positive curvature along mode (near minimum),
        # the reflected Hessian should have negative curvature. We need to preserve this for climbing.
        if mode_curvature > -self._ts_index_tolerance:
            # Mode curvature is positive or too close to zero - trust-krylov will minimize, not climb!
            # We need to make it negative to allow climbing. Set it to a small negative value.
            target_curvature = -self._ts_negative_mode_boost
            correction = target_curvature - mode_curvature
            stabilised += correction * np.outer(primary_mode, primary_mode)
            if self.verbose >= 2:
                logger.debug(
                    f"Mode curvature was {mode_curvature:.2e} (too positive), "
                    f"set to {target_curvature:.2e} to enable climbing"
                )
        elif mode_curvature < -10.0 * abs(self._ts_negative_mode_boost):
            # Mode curvature is too negative - might cause numerical issues or instability
            # Limit it to a reasonable negative value
            target_curvature = -10.0 * abs(self._ts_negative_mode_boost)
            correction = target_curvature - mode_curvature
            stabilised += correction * np.outer(primary_mode, primary_mode)
            if self.verbose >= 2:
                logger.debug(
                    f"Mode curvature was {mode_curvature:.2e} (too negative), "
                    f"limited to {target_curvature:.2e}"
                )

        # Final symmetrization and condition check
        stabilised = 0.5 * (stabilised + stabilised.T)

        # Verify final condition number
        try:
            final_cond = np.linalg.cond(stabilised)
            if final_cond > 1e12 and self.verbose >= 1:
                logger.warning(
                    f"Stabilised Hessian still ill-conditioned (cond={final_cond:.2e}). "
                    "Results may be numerically unstable."
                )
        except (np.linalg.LinAlgError, ValueError):
            pass

        return stabilised

    # ------------------------------------------------------------------
    # Overrides for gradient / Hessian that perform the TS reflection
    # ------------------------------------------------------------------
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Calculate gradient with transition state mode reflection."""
        gradient = super().gradient(x)
        self._ts_last_raw_gradient = gradient.copy()

        # Store raw gradient for step quality computation (used in next callback)
        # This captures the gradient at the current position before reflection
        self._ts_previous_gradient = gradient.copy()

        if self._ts_mode_vector is None:
            return gradient

        reflected = self._reflect_along_mode(gradient, self._ts_mode_vector)
        if self.use_bfgs_update:
            self._last_gradient = reflected.copy()
        return reflected

    def hessian_func(self, x: np.ndarray) -> np.ndarray:
        """Calculate Hessian using P-RFO partitioning for transition state optimization.

        Instead of reflection, we compute P-RFO steps explicitly and modify the Hessian
        so SciPy's trust-krylov naturally follows the P-RFO direction.
        """
        raw_hessian = super().hessian_func(x)
        self._ts_last_raw_hessian = raw_hessian.copy()
        self._ts_previous_hessian_raw = raw_hessian.copy()

        # Get raw gradient (will be reflected in gradient() method)
        raw_gradient = super().gradient(x)

        if self._should_update_mode():
            self._update_mode_from_hessian(raw_hessian)

        # If no mode, try to find it from Hessian
        if self._ts_mode_vector is None:
            try:
                sym_hessian = 0.5 * (raw_hessian + raw_hessian.T)
                eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)
                min_index = int(np.argmin(eigenvalues))
                self._ts_mode_vector = eigenvectors[:, min_index].copy()
                self._ts_mode_eigenvalue = float(eigenvalues[min_index])
                self._ts_last_mode_step = self.nsteps
            except Exception:
                if self.verbose >= 1:
                    logger.warning("Failed to find transition mode, using raw Hessian")
                return raw_hessian

        # Compute P-RFO step
        try:
            # Find optimal alpha to match trust radius
            alpha = self._find_optimal_alpha(raw_gradient, raw_hessian)
            prfo_step, lambda_max = self._compute_prfo_step(raw_gradient, raw_hessian, alpha)

            # Transform Hessian so trust-krylov's step is similar to P-RFO step
            # We want the Newton step -H^-1 * g to be close to prfo_step
            # So we modify H so that H_mod * prfo_step ≈ -g
            # Use rank-1 update: H_mod = H + β * (g + H*prfo_step) * prfo_step^T / ||prfo_step||^2

            step_norm = np.linalg.norm(prfo_step)
            if step_norm > 1e-12:
                # Compute what gradient would produce prfo_step
                H_prfo = raw_hessian @ prfo_step
                g_desired = -H_prfo  # Negative because we want -H^-1 * g = prfo_step
                g_error = raw_gradient - g_desired

                # Small rank-1 update to guide toward P-RFO
                beta = 0.5  # Mixing parameter
                if np.linalg.norm(g_error) > 1e-10:
                    modified_hessian = raw_hessian + beta * np.outer(g_error, prfo_step) / (
                        step_norm**2 + 1e-10
                    )
                    modified_hessian = 0.5 * (modified_hessian + modified_hessian.T)
                else:
                    modified_hessian = raw_hessian
            else:
                modified_hessian = raw_hessian

            # Ensure transition mode has negative curvature for climbing
            mode = self._ts_mode_vector / np.linalg.norm(self._ts_mode_vector)
            mode_curvature = float(mode @ (modified_hessian @ mode))
            if mode_curvature > -self._ts_index_tolerance:
                target_curvature = -self._ts_negative_mode_boost
                correction = target_curvature - mode_curvature
                modified_hessian += correction * np.outer(mode, mode)

            # Stabilize: make other negative eigenvalues positive (except transition mode)
            stabilised = self._stabilise_hessian(modified_hessian, mode)

            # Store for step quality computation
            self._ts_previous_hessian = stabilised.copy()

            if self.verbose >= 2:
                logger.debug(
                    f"P-RFO Hessian: step_norm={step_norm:.6f}, "
                    f"lambda_max={lambda_max:.6f}, alpha={alpha:.6f}"
                )

            return stabilised

        except Exception as e:
            if self.verbose >= 1:
                logger.warning(f"P-RFO computation failed: {e}, falling back to reflection")
            # Fallback: use old reflection method
            mode = self._ts_mode_vector / np.linalg.norm(self._ts_mode_vector)
            curvature = float(mode @ (raw_hessian @ mode))
            reflected_hessian = raw_hessian - 2.0 * curvature * np.outer(mode, mode)
            stabilised = self._stabilise_hessian(reflected_hessian, mode)
            self._ts_previous_hessian = stabilised.copy()
            return stabilised

    def _compute_step_quality(
        self,
        actual_energy_change: float,
        step_vector: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
    ) -> float:
        """Compute step quality factor Q based on geomeTRIC approach.

        For TS optimization, both positive and negative energy deviations
        from the predicted change are considered bad (unlike minimization).

        Q = 1 - |ΔE_actual/ΔE_pred - 1|

        Parameters
        ----------
        actual_energy_change : float
            Actual energy change: E_new - E_old
        step_vector : np.ndarray
            Step vector δ = x_new - x_old
        gradient : np.ndarray
            Gradient at previous point (before step)
        hessian : np.ndarray
            Hessian at previous point (before step)

        Returns:
        -------
        float
            Quality factor Q ∈ [-∞, 1]. Higher is better.
            Q ≥ 0.75: Good step
            0.5 ≤ Q < 0.75: Okay step
            0 ≤ Q < 0.5: Poor step
            Q < 0: Very poor step (should be rejected)
        """
        # Predicted energy change: ΔE_pred = 0.5 * δ^T * H * δ + δ^T * g
        H_delta = hessian @ step_vector
        predicted_quadratic = 0.5 * np.dot(step_vector, H_delta)
        predicted_linear = np.dot(step_vector, gradient)
        predicted_change = predicted_quadratic + predicted_linear

        # Handle edge cases
        if abs(predicted_change) < 1e-12:
            # Very small predicted change - quality is based on actual change magnitude
            if abs(actual_energy_change) < 1e-12:
                return 1.0  # Both zero, perfect match
            return -abs(actual_energy_change)  # Negative quality for unexpected change

        # Quality factor: Q = 1 - |ΔE_actual/ΔE_pred - 1|
        ratio = actual_energy_change / predicted_change
        quality = 1.0 - abs(ratio - 1.0)

        return quality

    def callback(self, x: np.ndarray, **kwargs: Any) -> None:
        """Callback function with step quality tracking for TS optimization."""
        # Update positions and increment step counter first
        if self.nsteps < self.max_steps:
            self.nsteps += 1

        # Get current energy and positions
        self.atoms.set_positions(self._x_to_positions(x))
        current_energy = self.atoms.get_potential_energy()
        current_forces = self.atoms.get_forces()
        fmax = np.max(np.abs(current_forces))

        # Compute step quality if we have previous state
        # Use the reflected gradient and stabilized Hessian (what trust-krylov actually used)
        step_quality = None
        if (
            self._ts_previous_energy is not None
            and self._ts_previous_positions is not None
            and self._ts_previous_gradient is not None
            and self._ts_previous_hessian is not None
        ):
            step_vector = x - self._ts_previous_positions
            step_norm = np.linalg.norm(step_vector)
            if step_norm > 1e-12:  # Only compute quality if step is significant
                actual_energy_change = current_energy - self._ts_previous_energy

                # For quality computation, we should use the reflected gradient
                # (what trust-krylov actually used) if we have a mode
                quality_gradient = self._ts_previous_gradient
                if self._ts_mode_vector is not None:
                    # Reflect the gradient to match what trust-krylov used
                    quality_gradient = self._reflect_along_mode(
                        self._ts_previous_gradient, self._ts_mode_vector
                    )

                step_quality = self._compute_step_quality(
                    actual_energy_change,
                    step_vector,
                    quality_gradient,  # Use reflected gradient
                    self._ts_previous_hessian,  # Use stabilized Hessian
                )
                self._ts_step_quality_history.append(step_quality)

                # Adjust trust radius based on step quality (P-RFO approach)
                self._adjust_trust_radius(step_quality, step_norm)

                if self.verbose >= 2:
                    quality_status = (
                        "good"
                        if step_quality >= 0.75
                        else "okay"
                        if step_quality >= 0.5
                        else "poor"
                        if step_quality >= 0.0
                        else "very poor"
                    )
                    logger.debug(
                        f"Step {self.nsteps}: quality Q={step_quality:.4f} ({quality_status}), "
                        f"ΔE_actual={actual_energy_change:.6f} eV, step_norm={step_norm:.6f} Å"
                    )
                elif self.verbose >= 1 and step_quality < 0.0:
                    logger.warning(
                        f"Step {self.nsteps}: very poor quality (Q={step_quality:.4f}), "
                        "consider recomputing Hessian"
                    )

        # Store current state for next step quality computation
        # The gradient and Hessian from this iteration will be stored in gradient/hessian_func
        # methods, so they'll be available for the next callback

        # Store for adaptive Hessian updates
        self._previous_fmax = fmax

        self.log(current_forces)
        self.call_observers()

        # Check convergence
        forces_flat = current_forces.ravel()
        if self.converged(forces_flat):
            raise ConvergedError

        # Store state for next iteration's step quality computation
        # Energy and positions are updated here; gradient/hessian will be computed
        # in the next iteration and stored in gradient/hessian_func methods
        self._ts_previous_energy = current_energy
        self._ts_previous_positions = x.copy()

    def run(self, fmax: float = 0.05, steps: int = 100) -> bool:
        """Run TS optimization with smaller trust radius and step quality tracking.

        Uses smaller initial trust radius (0.01-0.03) as recommended by geomeTRIC
        for TS optimization, which is 0.1× the default minimization values.
        """
        self.fmax = fmax
        self.max_steps = steps + self.nsteps

        # Get initial position
        x0 = self._positions_to_x()

        # Initialize previous state for step quality tracking
        if self.nsteps == 0:
            self._ts_previous_energy = self.atoms.get_potential_energy()
            self._ts_previous_positions = x0.copy()
            forces = self.atoms.get_forces()
            self.log(forces)
            self.call_observers()
            self.nsteps += 1
            # Initial gradient and hessian will be computed on first iteration

        if self.verbose >= 2:
            logger.info(f"Starting {self.method} TS optimization")
            logger.info(f"Convergence criterion: fmax = {fmax} eV/Å")
            logger.info(f"Maximum steps: {steps}")

        try:
            # Set up options for SciPy minimize with TS-specific trust region settings
            # Inspired by geomeTRIC: use much smaller trust radius for TS optimization
            options = {
                "maxiter": steps,
                "disp": False,  # We handle logging ourselves
            }

            # Trust-region options optimized for TS optimization
            # Note: SciPy's trust-krylov doesn't expose direct control over trust radius
            # via options, but we can still use tighter gradient tolerance
            # The smaller steps will come from the conservative Hessian and mode tracking
            options["gtol"] = 1e-8  # Tight gradient tolerance

            if self.verbose >= 2:
                logger.info(
                    "TS optimization: using tight gradient tolerance (gtol=1e-8). "
                    "Conservative steps enforced via mode tracking and Hessian stabilization."
                )

            # Run SciPy optimization
            self._scipy_result = minimize(  # type: ignore[call-overload]
                fun=self.objective,
                x0=x0,
                method=self.method,
                jac=self.gradient,
                hess=self.hessian_func,
                callback=self.callback,
                options=options,
            )

            # Update final positions
            if self._scipy_result is not None:
                self.atoms.set_positions(self._x_to_positions(self._scipy_result.x))

            # Log step quality summary
            if self._ts_step_quality_history and self.verbose >= 1:
                avg_quality = np.mean(self._ts_step_quality_history)
                good_steps = sum(1 for q in self._ts_step_quality_history if q >= 0.75)
                poor_steps = sum(1 for q in self._ts_step_quality_history if q < 0.0)
                if self.verbose >= 2:
                    logger.info(
                        f"Step quality summary: avg={avg_quality:.3f}, "
                        f"good={good_steps}, poor={poor_steps}"
                    )

        except ConvergedError:
            if self.verbose >= 1:
                logger.info("Optimization converged!")
            return True

        # Check final convergence
        forces = self.atoms.get_forces()
        forces_flat = forces.ravel()
        converged = self.converged(forces_flat)

        if converged:
            if self.verbose >= 1:
                logger.info("Optimization converged!")
        elif self.verbose >= 1:
            actual_steps = self.nsteps - 1  # Subtract initial step count
            logger.warning(
                f"Optimization stopped after {actual_steps} steps without converging "
                f"(max optimization steps: {steps})"
            )
            logger.warning(f"Final max force: {np.max(np.abs(forces)):.6f} eV/Å")

        return converged


class TrustNCG(SciPyHessianOptimizer):
    """Trust-NCG (Newton Conjugate Gradient) optimizer for ASE.

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
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Trust-NCG optimizer."""
        super().__init__(
            atoms=atoms,
            method="trust-ncg",
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )


class TrustExact(SciPyHessianOptimizer):
    """Trust-Exact optimizer for ASE.

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
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Trust-Exact optimizer."""
        super().__init__(
            atoms=atoms,
            method="trust-exact",
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )


class NewtonCG(SciPyHessianOptimizer):
    """Newton-CG (Newton Conjugate Gradient) optimizer for ASE.

    Uses SciPy's Newton-CG method, which is Newton's method where
    the linear system is solved using conjugate gradients.

    Parameters
    ----------
    See TrustKrylov for parameter documentation.

    """

    def __init__(
        self,
        atoms: Atoms,
        logfile: IO | str | None = "-",
        trajectory: str | None = None,
        hessian_update_freq: int | None = None,
        hessian_method: str = "auto",
        hessian_delta: float = 0.01,
        initial_hessian: np.ndarray | None = None,
        use_bfgs_update: bool = True,
        adaptive_hessian: bool = False,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize Newton-CG optimizer."""
        super().__init__(
            atoms=atoms,
            method="Newton-CG",
            logfile=logfile,
            trajectory=trajectory,
            hessian_update_freq=hessian_update_freq,
            hessian_method=hessian_method,
            hessian_delta=hessian_delta,
            initial_hessian=initial_hessian,
            use_bfgs_update=use_bfgs_update,
            adaptive_hessian=adaptive_hessian,
            verbose=verbose,
            **kwargs,
        )
