"""TRIC optimizer implementation.

This module contains the core TRIC optimization algorithm with support for both
minima and transition state optimization. The implementation uses internal coordinates
with proper dihedral gradients and basic eigenvalue-following for TS optimization.

Key features:
- Trust-radius optimization with BFGS Hessian updates
- Proper dihedral gradients based on pysisyphus approach
- Basic eigenvalue-following for transition state optimization
- ASE-compatible interface
- Numerical stability handling for degenerate cases
"""

import numpy as np
from typing import Optional, Callable
from ase import Atoms
from ase.optimize.optimize import Optimizer

from .utils import Geometry
from .internal_coords import InternalCoords
from .b_matrix import BMatrixCalculator


class TRICOptimizer(Optimizer):
    """TRIC (Translation-Rotation Internal Coordinates) optimizer.
    
    A completely independent implementation of TRIC optimization
    without external dependencies.
    """
    
    def __init__(self, atoms: Atoms, order: int = 0, 
                 hessian: Optional[np.ndarray] = None,
                 trust_radius: float = 0.3,
                 max_step: float = 0.2,
                 **kwargs):
        """Initialize TRIC optimizer.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object to optimize
        order : int, default 0
            Order of saddle point (0 for minima, 1 for transition state)
        hessian : np.ndarray, optional
            Initial Hessian matrix
        trust_radius : float, default 0.3
            Initial trust radius
        max_step : float, default 0.2
            Maximum step size
        **kwargs
            Additional keyword arguments for ASE Optimizer
        """
        super().__init__(atoms, **kwargs)
        
        self.order = order
        self.trust_radius = trust_radius
        self.max_step = max_step
        self.step_count = 0
        self._converged = False
        
        # Create geometry and internal coordinates
        self.geometry = Geometry.from_atoms(atoms)
        self.internal_coords = InternalCoords(self.geometry)
        self.b_matrix_calc = BMatrixCalculator(self.internal_coords)
        
        # Setup TRIC coordinates (including TR projection)
        self.b_matrix_calc.add_translation_rotation_coords(self.geometry)
        
        # Initialize Hessian
        if hessian is not None:
            self.hessian = hessian.copy()
        else:
            self.hessian = self.b_matrix_calc.calculate_hessian_guess(self.geometry)
        
        # Store previous values for BFGS update
        self.prev_internal_coords = None
        self.prev_gradient = None
        self.prev_hessian = None
        
        # Optimization state
        self.current_energy = None
        self.current_forces = None
        
        # Trust radius adaptation parameters
        self.trust_radius_min = 0.001
        self.trust_radius_max = 1.0
        self.trust_radius_factor = 2.0
        self.rho_threshold_good = 0.75
        self.rho_threshold_bad = 0.25
    
    def run(self, fmax: float = 0.05, steps: int = 1000) -> bool:
        """Run TRIC optimization.
        
        Parameters
        ----------
        fmax : float, default 0.05
            Maximum force threshold for convergence
        steps : int, default 1000
            Maximum number of optimization steps
            
        Returns
        -------
        bool
            True if optimization converged, False otherwise
        """
        self.log(f"Starting TRIC optimization (order={self.order}, fmax={fmax})")
        
        # Get initial internal coordinates
        q = self.internal_coords.eval_geometry(self.geometry)
        
        for step in range(steps):
            self.step_count = step
            
            # Calculate energy and forces
            energy = self.atoms.get_potential_energy()
            forces = self.atoms.get_forces()
            
            # Update geometry
            self.geometry.positions = self.atoms.get_positions()
            
            # Project forces to remove TR components
            projected_forces = self.b_matrix_calc.project_gradient(forces)
            
            # Convert forces to internal coordinates
            internal_forces = self.b_matrix_calc.project_cartesian_forces(
                self.geometry, projected_forces
            )
            
            # Check convergence
            max_force = np.max(np.abs(forces))
            if max_force < fmax:
                self._converged = True
                self.log(f"Optimization converged after {step} steps (fmax={max_force:.6f})")
                break
            
            # Calculate optimization step
            if step == 0:
                # First step: steepest descent using G-matrix
                B = self.b_matrix_calc.calculate_B_matrix(self.geometry)
                G = B @ B.T
                
                # Always use pseudo-inverse to handle potentially singular G-matrix
                G_inv = np.linalg.pinv(G)
                
                # Steepest descent: dq = +G_inv @ g (positive sign for downhill)
                dq = G_inv @ internal_forces
                
                # Apply trust radius
                step_norm = np.linalg.norm(dq)
                if step_norm > self.trust_radius:
                    dq = dq / step_norm * self.trust_radius
                
                # Apply maximum step limit
                step_norm = np.linalg.norm(dq)
                if step_norm > self.max_step:
                    dq = dq / step_norm * self.max_step
            else:
                # Use appropriate step calculation for minima vs TS
                if self.order == 1:  # Transition state optimization
                    # Convert forces to gradients for RFO: g = -f
                    internal_gradient = -internal_forces
                    dq = self._calculate_ts_step(internal_gradient)
                else:  # Minima optimization
                    try:
                        # Newton step: dq = +H_inv @ g (positive sign for downhill)
                        dq = np.linalg.solve(self.hessian, internal_forces)
                        
                        # Apply trust radius
                        step_norm = np.linalg.norm(dq)
                        if step_norm > self.trust_radius:
                            dq = dq / step_norm * self.trust_radius
                        
                        # Apply maximum step limit
                        step_norm = np.linalg.norm(dq)
                        if step_norm > self.max_step:
                            dq = dq / step_norm * self.max_step
                            
                    except np.linalg.LinAlgError:
                        # Fallback to steepest descent
                        B = self.b_matrix_calc.calculate_B_matrix(self.geometry)
                        G = B @ B.T
                        # Always use pseudo-inverse for consistency
                        G_inv = np.linalg.pinv(G)
                        dq = G_inv @ internal_forces
                        
                        # Apply trust radius
                        step_norm = np.linalg.norm(dq)
                        if step_norm > self.trust_radius:
                            dq = dq / step_norm * self.trust_radius
            
            # Store current energy and geometry before step
            energy_before = energy
            prev_positions = self.geometry.positions.copy()
            prev_q = q.copy()
            
            # Update geometry using internal coordinates
            q_new, geom_new = self._update_geometry(q, dq)
            
            # Update atoms object
            self.atoms.set_positions(geom_new.positions)
            self.geometry = geom_new
            q = q_new
            
            # Calculate energy after step
            energy_after = self.atoms.get_potential_energy()
            
            # Trust radius adaptation based on step quality
            step_accepted = self._adapt_trust_radius(energy_before, energy_after, dq, internal_forces)
            
            if step_accepted:
                # Update Hessian using BFGS
                if step > 0:
                    self._update_hessian_bfgs(q, internal_forces)
                
                # Store values for next iteration
                self.prev_internal_coords = q.copy()
                self.prev_gradient = internal_forces.copy()
                
                # Log progress
                self.log(f"Step {step}: Energy={energy_after:.6f}, Max force={max_force:.6f}, Trust radius={self.trust_radius:.4f}")
            else:
                # Reject step and restore previous geometry
                self.atoms.set_positions(prev_positions)
                self.geometry.positions = prev_positions
                q = prev_q
                
                self.log(f"Step {step}: Step rejected, Trust radius={self.trust_radius:.4f}")
        
        if not self._converged:
            self.log(f"Optimization did not converge after {steps} steps")
        
        return self._converged
    
    def converged(self, forces=None):
        """Check if optimization has converged.
        
        Parameters
        ----------
        forces : np.ndarray, optional
            Forces array (for compatibility with ASE optimizers)
            
        Returns
        -------
        bool
            True if converged, False otherwise
        """
        return self._converged
    
    def _update_geometry(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, Geometry]:
        """Update geometry using internal coordinate step."""
        # Transform internal step to Cartesian using correct G-matrix transformation
        cartesian_step = self.b_matrix_calc.project_internal_step(self.geometry, dq)
        
        # Update geometry directly
        new_positions = self.geometry.positions + cartesian_step
        geom_new = Geometry(self.geometry.symbols, new_positions)
        
        # Calculate new internal coordinates
        q_new = self.internal_coords.eval_geometry(geom_new)
        
        return q_new, geom_new
    
    def _update_hessian_bfgs(self, q: np.ndarray, gradient: np.ndarray):
        """Update Hessian using BFGS formula.
        
        Note: The variable 'gradient' is actually internal forces from ASE (= -internal_gradient).
        For BFGS, we need the actual gradient difference y = g_new - g_old.
        """
        if self.prev_internal_coords is None or self.prev_gradient is None:
            return
        
        # BFGS update for Hessian (not inverse Hessian)
        s = q - self.prev_internal_coords  # Step in internal coordinates
        
        # Convert forces to gradients: g = -f
        # y = g_new - g_old = -f_new - (-f_old) = -(f_new - f_old)
        y = -(gradient - self.prev_gradient)  # Gradient difference (fixed sign)
        
        # Skip update if step or gradient change is too small
        if np.linalg.norm(s) < 1e-12 or np.linalg.norm(y) < 1e-12:
            return
        
        # Check curvature condition: y^T * s > 0
        ys = np.dot(y, s)
        
        if ys > 1e-12:  # Only update if curvature condition is satisfied
            # BFGS formula for Hessian (not inverse Hessian)
            # H_new = H - (H*s*s^T*H)/(s^T*H*s) + (y*y^T)/(y^T*s)
            Hs = self.hessian @ s
            sHs = s @ self.hessian @ s
            
            self.hessian = self.hessian - np.outer(Hs, Hs) / sHs + np.outer(y, y) / ys
    
    def _adapt_trust_radius(self, energy_before: float, energy_after: float, 
                           dq: np.ndarray, internal_forces: np.ndarray) -> bool:
        """Adapt trust radius based on step quality.
        
        Parameters
        ----------
        energy_before : float
            Energy before the step
        energy_after : float
            Energy after the step
        dq : np.ndarray
            Internal coordinate step
        internal_forces : np.ndarray
            Internal coordinate forces
            
        Returns
        -------
        bool
            True if step is accepted, False if rejected
        """
        # Calculate predicted energy change using quadratic model
        # ΔE_predicted = -g^T * dq + 0.5 * dq^T * H * dq
        predicted_energy_change = -np.dot(internal_forces, dq)
        
        # Add quadratic term if Hessian is available
        if hasattr(self, 'hessian') and self.hessian is not None:
            quadratic_term = 0.5 * np.dot(dq, self.hessian @ dq)
            predicted_energy_change += quadratic_term
        
        # Actual energy change
        actual_energy_change = energy_after - energy_before
        
        # Calculate rho (step quality ratio)
        if abs(predicted_energy_change) < 1e-12:
            rho = 1.0 if actual_energy_change < 0 else 0.0
        else:
            rho = actual_energy_change / predicted_energy_change
        
        # Adapt trust radius based on rho
        if rho > self.rho_threshold_good:
            # Good step: increase trust radius
            self.trust_radius = min(self.trust_radius * self.trust_radius_factor, self.trust_radius_max)
            step_accepted = True
        elif rho > self.rho_threshold_bad:
            # Acceptable step: keep trust radius
            step_accepted = True
        else:
            # Bad step: decrease trust radius and reject step
            self.trust_radius = max(self.trust_radius / self.trust_radius_factor, self.trust_radius_min)
            step_accepted = False
        
        return step_accepted
    
    def log(self, message: str):
        """Log optimization progress."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"TRIC: {message}")


class TRICTSOptimizer(TRICOptimizer):
    """TRIC optimizer for transition state searches."""
    
    def __init__(self, atoms: Atoms, **kwargs):
        """Initialize TRIC TS optimizer."""
        # Set order=1 for transition state optimization
        kwargs['order'] = 1
        super().__init__(atoms, **kwargs)
    
    def _update_hessian_bfgs(self, q: np.ndarray, gradient: np.ndarray):
        """Update Hessian for transition state optimization."""
        # For TS optimization, use standard BFGS but check eigenvalues
        super()._update_hessian_bfgs(q, gradient)
        
        # Check that Hessian has exactly one negative eigenvalue for TS
        eigenvalues = np.linalg.eigvals(self.hessian)
        negative_eigenvals = eigenvalues[eigenvalues < 0]
        
        if len(negative_eigenvals) != 1:
            self.log(f"Warning: TS Hessian has {len(negative_eigenvals)} negative eigenvalues, expected 1")
    
    def _calculate_ts_step(self, gradient: np.ndarray) -> np.ndarray:
        """Calculate step for transition state optimization using P-RFO.
        
        Parameters
        ----------
        gradient : np.ndarray
            Internal coordinate gradient (not forces, g points uphill)
            
        Returns
        -------
        np.ndarray
            Step in internal coordinates
        """
        try:
            from .rfo import (
                restricted_step_microcycles,
                calculate_ts_mode_indices,
                calculate_min_mode_indices,
                validate_rfo_step
            )
            
            # Diagonalize Hessian
            eigenvalues, eigenvectors = np.linalg.eigh(self.hessian)
            
            # Find TS and minimization mode indices
            ts_indices = calculate_ts_mode_indices(eigenvalues, n_negative=1)
            min_indices = calculate_min_mode_indices(eigenvalues, ts_indices)
            
            # Check that we have exactly one TS mode
            if len(ts_indices) != 1:
                self.log(f"Warning: Found {len(ts_indices)} TS modes, expected 1")
                if len(ts_indices) == 0:
                    self.log("No negative eigenvalues found, using steepest descent")
                    step_size = min(self.trust_radius, self.max_step)
                    return -step_size * gradient / np.linalg.norm(gradient)
            
            # Perform P-RFO with restricted-step micro-cycles
            dq, alpha_history = restricted_step_microcycles(
                eigenvals=eigenvalues,
                eigenvecs=eigenvectors,
                gradient=gradient,
                trust_radius=self.trust_radius,
                max_micro_cycles=getattr(self, 'max_micro_cycles', 25),
                alpha0=getattr(self, 'alpha0', 1.0),
                min_indices=min_indices,
                max_indices=ts_indices
            )
            
            # Validate the step
            is_valid, warnings = validate_rfo_step(dq, gradient, self.hessian, self.trust_radius)
            if warnings:
                for warning in warnings:
                    self.log(f"RFO step warning: {warning}")
            
            # Log alpha history
            if len(alpha_history) > 1:
                self.log(f"P-RFO converged in {len(alpha_history)} micro-cycles, "
                        f"final alpha={alpha_history[-1]:.6f}")
            
            return dq
            
        except ImportError:
            # Fallback to basic eigenvalue-following if RFO module not available
            self.log("Warning: RFO module not available, using basic eigenvalue-following")
            return self._calculate_basic_ts_step(gradient)
        except Exception as e:
            # Fallback to basic eigenvalue-following on any error
            self.log(f"Warning: P-RFO failed ({e}), using basic eigenvalue-following")
            return self._calculate_basic_ts_step(gradient)
    
    def _calculate_basic_ts_step(self, gradient: np.ndarray) -> np.ndarray:
        """Fallback basic eigenvalue-following algorithm.
        
        Parameters
        ----------
        gradient : np.ndarray
            Internal coordinate gradient (not forces, g points uphill)
            
        Returns
        -------
        np.ndarray
            Step in internal coordinates
        """
        try:
            # Diagonalize Hessian to find negative eigenvalue mode
            eigenvalues, eigenvectors = np.linalg.eigh(self.hessian)
            
            # Find the most negative eigenvalue
            min_idx = np.argmin(eigenvalues)
            negative_mode = eigenvectors[:, min_idx]
            negative_eigenval = eigenvalues[min_idx]
            
            # Project gradient onto the negative mode for uphill movement
            # and onto other modes for downhill movement
            grad_parallel = np.dot(gradient, negative_mode) * negative_mode
            grad_perpendicular = gradient - grad_parallel
            
            # For negative eigenvalue mode, go uphill (in gradient direction)
            # For other modes, go downhill (opposite to gradient direction)
            if negative_eigenval < 0:
                # Eigenvalue-following TS search:
                # - Maximize along negative mode: step = +g_parallel / |lambda|
                # - Minimize along positive modes: use Newton step for each mode
                
                dq = np.zeros_like(gradient)
                for i, eigval in enumerate(eigenvalues):
                    mode = eigenvectors[:, i]
                    grad_component = np.dot(gradient, mode)
                    
                    if i == min_idx:  # Negative eigenvalue mode
                        # Move uphill: step in direction of gradient
                        dq += (grad_component / abs(eigval)) * mode
                    else:  # Positive eigenvalue modes
                        # Move downhill: Newton step
                        dq += (-grad_component / max(abs(eigval), 1e-6)) * mode
            else:
                # Fallback to regular Newton step if no negative eigenvalue
                dq = -np.linalg.solve(self.hessian, gradient)
            
            # Apply trust radius
            step_norm = np.linalg.norm(dq)
            if step_norm > self.trust_radius:
                dq = dq / step_norm * self.trust_radius
            
            return dq
            
        except np.linalg.LinAlgError:
            # Fallback to steepest descent
            self.log("Warning: Hessian diagonalization failed, using steepest descent")
            step_size = min(self.trust_radius, self.max_step)
            return -step_size * gradient / np.linalg.norm(gradient)


def create_tric_optimizer(atoms: Atoms, order: int = 0, **kwargs) -> TRICOptimizer:
    """Factory function to create TRIC optimizer.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object to optimize
    order : int, default 0
        Order of saddle point (0 for minima, 1 for TS)
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    TRICOptimizer
        TRIC optimizer instance
    """
    if order == 1:
        return TRICTSOptimizer(atoms, **kwargs)
    else:
        return TRICOptimizer(atoms, order=order, **kwargs)
