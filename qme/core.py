"""
Core optimization functionality combining ASE and SELLA optimizers with UMA potentials.
"""

from typing import Optional, Union, Dict, Any, List
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS, FIRE
from ase.constraints import FixAtoms

try:
    from sella import Sella, IRC
    HAS_SELLA = True
except ImportError:
    HAS_SELLA = False

from .uma_potential import UMAPotential, get_uma_calculator

# Import mock calculator for testing
from .mock_calculator import get_mock_uma_calculator


class QMEOptimizer:
    """
    Main optimizer class that combines ASE and SELLA optimizers with UMA potentials.
    """
    
    AVAILABLE_OPTIMIZERS = {
        'BFGS': BFGS,
        'LBFGS': LBFGS, 
        'FIRE': FIRE,
    }
    
    if HAS_SELLA:
        AVAILABLE_OPTIMIZERS['Sella'] = Sella
    
    def __init__(
        self,
        calculator: Optional[UMAPotential] = None,
        model_name: str = "uma-4m",
        device: Optional[str] = None,
        use_mock: bool = False,
    ):
        """
        Initialize QME optimizer.
        
        Parameters:
        -----------
        calculator : UMAPotential, optional
            Pre-configured UMA calculator. If None, creates one with model_name.
        model_name : str
            UMA model name to use if calculator is None
        device : str, optional
            Device for computations ('cpu', 'cuda')
        use_mock : bool
            Use mock calculator for testing (default: False)
        """
        
        if calculator is None:
            if use_mock:
                self.calculator = get_mock_uma_calculator()
            else:
                try:
                    self.calculator = get_uma_calculator(model_name=model_name, device=device)
                except ImportError as e:
                    print(f"Warning: {e}")
                    print("Falling back to mock calculator for testing.")
                    self.calculator = get_mock_uma_calculator()
        else:
            self.calculator = calculator
            
        self.atoms = None
        self.results = {}
    
    def load_structure(self, structure_file: Union[str, Path]) -> Atoms:
        """
        Load molecular structure from file.
        
        Parameters:
        -----------
        structure_file : str or Path
            Path to structure file (xyz, cif, pdb, etc.)
            
        Returns:
        --------
        Atoms
            Loaded structure
        """
        
        structure_file = Path(structure_file)
        if not structure_file.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_file}")
        
        try:
            atoms = read(structure_file)
            atoms.calc = self.calculator
            self.atoms = atoms
            return atoms
        except Exception as e:
            raise RuntimeError(f"Failed to load structure from {structure_file}: {e}")
    
    def optimize_minimum(
        self,
        atoms: Optional[Atoms] = None,
        optimizer: str = "BFGS",
        fmax: float = 0.01,
        steps: int = 200,
        logfile: Optional[str] = None,
        trajectory: Optional[str] = None,
        constraints: Optional[List] = None,
        **optimizer_kwargs
    ) -> Dict[str, Any]:
        """
        Optimize structure to find minimum energy geometry.
        
        Parameters:
        -----------
        atoms : Atoms, optional
            Structure to optimize. Uses self.atoms if None.
        optimizer : str
            Optimizer name ('BFGS', 'LBFGS', 'FIRE')
        fmax : float
            Force convergence criterion
        steps : int
            Maximum optimization steps
        logfile : str, optional
            Log file path
        trajectory : str, optional
            Trajectory file path
        constraints : list, optional
            List of ASE constraints
        **optimizer_kwargs :
            Additional optimizer arguments
            
        Returns:
        --------
        dict
            Optimization results
        """
        
        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()
            
        atoms.calc = self.calculator
        
        # Apply constraints if provided
        if constraints:
            atoms.set_constraint(constraints)
        
        # Get optimizer class
        if optimizer not in self.AVAILABLE_OPTIMIZERS:
            available = list(self.AVAILABLE_OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer: {optimizer}. Available: {available}")
        
        OptimizerClass = self.AVAILABLE_OPTIMIZERS[optimizer]
        
        # Initialize optimizer
        opt = OptimizerClass(atoms, logfile=logfile, trajectory=trajectory, **optimizer_kwargs)
        
        # Store initial state
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_max_force = np.max(np.abs(initial_forces))
        
        # Run optimization
        converged = opt.run(fmax=fmax, steps=steps)
        
        # Store final state
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        final_max_force = np.max(np.abs(final_forces))
        
        results = {
            'converged': converged,
            'steps_taken': opt.get_number_of_steps(),
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_change': final_energy - initial_energy,
            'initial_max_force': initial_max_force,
            'final_max_force': final_max_force,
            'optimized_atoms': atoms,
            'optimizer_used': optimizer,
        }
        
        self.results['minimum_optimization'] = results
        return results
    
    def find_transition_state(
        self,
        atoms: Optional[Atoms] = None,
        fmax: float = 0.01,
        steps: int = 200,
        logfile: Optional[str] = None,
        trajectory: Optional[str] = None,
        constraints: Optional[List] = None,
        **sella_kwargs
    ) -> Dict[str, Any]:
        """
        Find transition state (saddle point) using SELLA.
        
        Parameters:
        -----------
        atoms : Atoms, optional
            Starting structure for TS search
        fmax : float
            Force convergence criterion
        steps : int
            Maximum optimization steps
        logfile : str, optional
            Log file path
        trajectory : str, optional
            Trajectory file path
        constraints : list, optional
            List of ASE constraints
        **sella_kwargs :
            Additional SELLA arguments
            
        Returns:
        --------
        dict
            TS optimization results
        """
        
        if not HAS_SELLA:
            raise ImportError(
                "SELLA is required for transition state searches. "
                "Install with: pip install sella"
            )
        
        if atoms is None:
            if self.atoms is None:
                raise ValueError("No structure loaded. Use load_structure() first.")
            atoms = self.atoms.copy()
        else:
            atoms = atoms.copy()
            
        atoms.calc = self.calculator
        
        # Apply constraints if provided
        if constraints:
            atoms.set_constraint(constraints)
        
        # Initialize SELLA optimizer for TS search
        opt = Sella(
            atoms, 
            logfile=logfile, 
            trajectory=trajectory,
            **sella_kwargs
        )
        
        # Store initial state
        initial_energy = atoms.get_potential_energy()
        initial_forces = atoms.get_forces()
        initial_max_force = np.max(np.abs(initial_forces))
        
        # Run TS optimization
        converged = opt.run(fmax=fmax, steps=steps)
        
        # Store final state
        final_energy = atoms.get_potential_energy()
        final_forces = atoms.get_forces()
        final_max_force = np.max(np.abs(final_forces))
        
        results = {
            'converged': converged,
            'steps_taken': opt.get_number_of_steps(),
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_change': final_energy - initial_energy,
            'initial_max_force': initial_max_force,
            'final_max_force': final_max_force,
            'ts_atoms': atoms,
            'optimizer_used': 'Sella',
        }
        
        self.results['transition_state_search'] = results
        return results
    
    def save_structure(
        self, 
        atoms: Atoms, 
        output_file: Union[str, Path],
        format: Optional[str] = None
    ):
        """
        Save optimized structure to file.
        
        Parameters:
        -----------
        atoms : Atoms
            Structure to save
        output_file : str or Path
            Output file path
        format : str, optional
            File format (inferred from extension if None)
        """
        
        output_file = Path(output_file)
        
        try:
            write(output_file, atoms, format=format)
        except Exception as e:
            raise RuntimeError(f"Failed to save structure to {output_file}: {e}")
    
    def get_optimization_summary(self) -> str:
        """
        Get summary of optimization results.
        
        Returns:
        --------
        str
            Formatted summary text
        """
        
        summary_lines = ["QME Optimization Summary", "=" * 25]
        
        for calc_type, results in self.results.items():
            summary_lines.extend([
                f"\n{calc_type.replace('_', ' ').title()}:",
                f"  Converged: {results['converged']}",
                f"  Steps: {results['steps_taken']}",
                f"  Initial Energy: {results['initial_energy']:.6f} eV",
                f"  Final Energy: {results['final_energy']:.6f} eV",
                f"  Energy Change: {results['energy_change']:.6f} eV",
                f"  Max Force (initial): {results['initial_max_force']:.4f} eV/Å",
                f"  Max Force (final): {results['final_max_force']:.4f} eV/Å",
                f"  Optimizer: {results['optimizer_used']}"
            ])
        
        return "\n".join(summary_lines)