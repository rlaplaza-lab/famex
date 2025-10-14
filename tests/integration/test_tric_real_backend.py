"""Integration tests for TRIC optimizer with real ML backends.

This module implements the TRIC debug plan to systematically test and validate
the TRIC optimizer against real ML backends (MACE, UMA when available).
Tests use eclipsed ethane as the primary test system.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.optimize.lbfgs import LBFGS
from ase.optimize import BFGS
import tempfile
import os
import logging

from qme.core.tric import create_tric_optimizer, TRICOptimizer, TRICTSOptimizer
from qme.core.local_strategies import _get_local_optimizer_class
from qme.core.explorer import Explorer
from qme.logging_utils import get_qme_logger

logger = get_qme_logger(__name__)


def create_eclipsed_ethane():
    """Create eclipsed ethane geometry (C2H6).
    
    Returns
    -------
    Atoms
        Eclipsed ethane molecule with C-C bond length ~1.54 Å
    """
    # Eclipsed ethane geometry
    # C-C bond length: 1.54 Å (experimental)
    # C-H bond length: 1.09 Å (experimental)
    # H-C-H angle: 109.47° (tetrahedral)
    # Dihedral angle: 0° (eclipsed)
    
    cc_bond = 1.54
    ch_bond = 1.09
    
    # Carbon positions
    c1_pos = np.array([0.0, 0.0, 0.0])
    c2_pos = np.array([cc_bond, 0.0, 0.0])
    
    # Hydrogen positions around C1 (tetrahedral)
    # First H at (1, 1, 1) direction, normalized
    h1_c1 = ch_bond * np.array([1, 1, 1]) / np.sqrt(3)
    h2_c1 = ch_bond * np.array([1, -1, -1]) / np.sqrt(3)
    h3_c1 = ch_bond * np.array([-1, 1, -1]) / np.sqrt(3)
    h4_c1 = ch_bond * np.array([-1, -1, 1]) / np.sqrt(3)
    
    # Hydrogen positions around C2 (eclipsed with C1)
    # Rotated 60° around C-C axis to achieve eclipsed conformation
    theta = np.pi / 3  # 60 degrees
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Rotate C1 hydrogens by 60° around x-axis and translate to C2
    h1_c2 = ch_bond * np.array([1, cos_theta, sin_theta]) / np.sqrt(3) + c2_pos
    h2_c2 = ch_bond * np.array([1, -cos_theta, -sin_theta]) / np.sqrt(3) + c2_pos
    h3_c2 = ch_bond * np.array([-1, cos_theta, -sin_theta]) / np.sqrt(3) + c2_pos
    h4_c2 = ch_bond * np.array([-1, -cos_theta, sin_theta]) / np.sqrt(3) + c2_pos
    
    # Add C1 hydrogens
    h1_c1 += c1_pos
    h2_c1 += c1_pos
    h3_c1 += c1_pos
    h4_c1 += c1_pos
    
    # Combine all positions
    positions = np.array([
        c1_pos,  # C1
        h1_c1,   # H1
        h2_c1,   # H2
        h3_c1,   # H3
        h4_c1,   # H4
        c2_pos,  # C2
        h1_c2,   # H5
        h2_c2,   # H6
        h3_c2,   # H7
        h4_c2,   # H8
    ])
    
    symbols = ['C', 'H', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'H']
    
    atoms = Atoms(symbols=symbols, positions=positions)
    
    # Set charge and spin info for UMA backend
    atoms.info['charge'] = 0
    atoms.info['spin'] = 1
    
    return atoms


def create_ethane_ts_guess():
    """Create a transition state guess for ethane rotation.
    
    This creates a staggered conformation which is a reasonable TS guess
    for the eclipsed -> staggered rotation.
    
    Returns
    -------
    Atoms
        Staggered ethane molecule as TS guess
    """
    # Start with eclipsed conformation and rotate one methyl group
    atoms = create_eclipsed_ethane()
    positions = atoms.get_positions()
    
    # Rotate C2 methyl group by 60° around C-C axis
    cc_bond = 1.54
    c2_pos = np.array([cc_bond, 0.0, 0.0])
    
    # Get C2 hydrogens
    c2_h_indices = [6, 7, 8, 9]  # H5, H6, H7, H8 in our numbering
    
    # Rotate around C-C axis (x-axis) by 60°
    theta = np.pi / 3
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    for i in c2_h_indices:
        # Translate to origin (C2 position)
        pos = positions[i] - c2_pos
        
        # Rotate around x-axis
        y_new = pos[1] * cos_theta - pos[2] * sin_theta
        z_new = pos[1] * sin_theta + pos[2] * cos_theta
        
        # Update position
        positions[i] = np.array([pos[0], y_new, z_new]) + c2_pos
    
    atoms.set_positions(positions)
    return atoms


def calculate_rmsd(atoms1, atoms2):
    """Calculate RMSD between two structures.
    
    Parameters
    ----------
    atoms1, atoms2 : Atoms
        The two structures to compare
        
    Returns
    -------
    float
        RMSD in Angstroms
    """
    pos1 = atoms1.get_positions()
    pos2 = atoms2.get_positions()
    
    # Align structures using Kabsch algorithm (simplified)
    # For now, just calculate raw RMSD
    diff = pos1 - pos2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return rmsd


def run_optimization_with_logging(atoms, optimizer_class, optimizer_name, 
                                fmax=0.05, steps=100, **kwargs):
    """Run optimization with detailed logging.
    
    Parameters
    ----------
    atoms : Atoms
        Structure to optimize
    optimizer_class : type
        Optimizer class to use
    optimizer_name : str
        Name for logging
    fmax : float
        Force convergence threshold
    steps : int
        Maximum steps
    **kwargs
        Additional optimizer arguments
        
    Returns
    -------
    dict
        Results with atoms, energy, steps, converged, trajectory
    """
    logger.info(f"Starting {optimizer_name} optimization")
    logger.info(f"Initial energy: {atoms.get_potential_energy():.6f} eV")
    logger.info(f"Initial max force: {np.max(np.abs(atoms.get_forces())):.6f} eV/Å")
    
    # Make copy to avoid modifying original
    atoms_copy = atoms.copy()
    
    # Create optimizer
    optimizer = optimizer_class(atoms_copy, **kwargs)
    
    # Store trajectory
    trajectory = [atoms_copy.copy()]
    energies = [atoms_copy.get_potential_energy()]
    max_forces = [np.max(np.abs(atoms_copy.get_forces()))]
    
    # Run optimization with step-by-step logging
    try:
        converged = optimizer.run(fmax=fmax, steps=steps)
        
        # Get final results
        final_energy = atoms_copy.get_potential_energy()
        final_max_force = np.max(np.abs(atoms_copy.get_forces()))
        
        # Get step count
        if hasattr(optimizer, 'step_count'):
            steps_taken = optimizer.step_count
        elif hasattr(optimizer, 'get_number_of_steps'):
            steps_taken = optimizer.get_number_of_steps()
        else:
            steps_taken = steps  # Fallback
            
        logger.info(f"{optimizer_name} completed:")
        logger.info(f"  Final energy: {final_energy:.6f} eV")
        logger.info(f"  Final max force: {final_max_force:.6f} eV/Å")
        logger.info(f"  Steps taken: {steps_taken}")
        logger.info(f"  Converged: {converged}")
        
        return {
            'atoms': atoms_copy,
            'energy': final_energy,
            'steps': steps_taken,
            'converged': converged,
            'trajectory': trajectory,
            'energies': energies,
            'max_forces': max_forces,
            'optimizer': optimizer
        }
        
    except Exception as e:
        logger.error(f"{optimizer_name} failed: {e}")
        return {
            'atoms': atoms_copy,
            'energy': float('inf'),
            'steps': 0,
            'converged': False,
            'trajectory': trajectory,
            'energies': energies,
            'max_forces': max_forces,
            'optimizer': optimizer,
            'error': str(e)
        }


class TestTRICRealBackend:
    """Test TRIC optimizer with real UMA backend."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class."""
        # Check available backends
        from qme.backend_availability import get_available_backends, is_backend_available
        
        cls.available_backends = get_available_backends(include_mock=False)
        cls.mace_available = is_backend_available('mace')
        cls.uma_available = is_backend_available('uma')
        
        logger.info(f"Available backends: {cls.available_backends}")
        logger.info(f"MACE available: {cls.mace_available}")
        logger.info(f"UMA available: {cls.uma_available}")
        
        # Use the first available ML backend
        if cls.mace_available:
            cls.test_backend = 'mace'
        elif cls.uma_available:
            cls.test_backend = 'uma'
        else:
            cls.test_backend = None
            logger.warning("No ML backends available - tests will be skipped")
            
        # Create test geometries
        cls.eclipsed_ethane = create_eclipsed_ethane()
        cls.ethane_ts_guess = create_ethane_ts_guess()
        
    def test_ml_backend_availability(self):
        """Test that at least one ML backend is available."""
        assert self.test_backend is not None, f"No ML backends available. Available: {self.available_backends}"
        
    def test_bfgs_minima_baseline(self):
        """Test BFGS minima optimization baseline with ML backend."""
        if self.test_backend is None:
            pytest.skip("No ML backends available")
            
        # Create explorer with available ML backend
        explorer = Explorer(
            atoms=self.eclipsed_ethane,
            backend=self.test_backend,
            target='minima',
            strategy='local',
            local_optimizer='bfgs'
        )
        
        # Run optimization
        result = explorer.run(fmax=0.05, steps=100)
        
        # Store baseline results
        self.bfgs_baseline = {
            'atoms': result['optimized_atoms'],
            'energy': result['optimized_atoms'].get_potential_energy(),
            'steps': result.get('steps_taken', 0),
            'converged': result.get('converged', False)
        }
        
        logger.info(f"BFGS baseline: Energy={self.bfgs_baseline['energy']:.6f} eV, "
                   f"Steps={self.bfgs_baseline['steps']}, Converged={self.bfgs_baseline['converged']}")
        
        # Basic assertions
        assert self.bfgs_baseline['converged'], "BFGS baseline should converge"
        assert self.bfgs_baseline['energy'] < float('inf'), "BFGS baseline should have finite energy"
        
    def test_sella_ts_baseline(self):
        """Test Sella TS optimization baseline with ML backend."""
        if self.test_backend is None:
            pytest.skip("No ML backends available")
            
        # Create explorer with available ML backend for TS search
        explorer = Explorer(
            atoms=self.ethane_ts_guess,
            backend=self.test_backend,
            target='ts',
            strategy='local',
            local_optimizer='sella'
        )
        
        # Run optimization
        result = explorer.run(fmax=0.05, steps=100)
        
        # Store baseline results
        self.sella_baseline = {
            'atoms': result['optimized_atoms'],
            'energy': result['optimized_atoms'].get_potential_energy(),
            'steps': result.get('steps_taken', 0),
            'converged': result.get('converged', False)
        }
        
        logger.info(f"Sella baseline: Energy={self.sella_baseline['energy']:.6f} eV, "
                   f"Steps={self.sella_baseline['steps']}, Converged={self.sella_baseline['converged']}")
        
        # Basic assertions
        # Note: TS optimization may not always converge with this simple test case
        # We'll be more lenient here
        assert self.sella_baseline['energy'] < float('inf'), "Sella baseline should have finite energy"
        
    def test_tric_minima_optimization(self):
        """Test TRIC minima optimization and compare with BFGS baseline."""
        if self.test_backend is None:
            pytest.skip("No ML backends available")
            
        # First run BFGS baseline if not already done
        if not hasattr(self, 'bfgs_baseline'):
            self.test_bfgs_minima_baseline()
            
        # Create explorer with available ML backend for TRIC minima
        explorer = Explorer(
            atoms=self.eclipsed_ethane,
            backend=self.test_backend,
            target='minima',
            strategy='local',
            local_optimizer='tric'
        )
        
        # Run optimization
        result = explorer.run(fmax=0.05, steps=100)
        
        # Store TRIC results
        tric_result = {
            'atoms': result['optimized_atoms'],
            'energy': result['optimized_atoms'].get_potential_energy(),
            'steps': result.get('steps_taken', 0),
            'converged': result.get('converged', False)
        }
        
        logger.info(f"TRIC minima: Energy={tric_result['energy']:.6f} eV, "
                   f"Steps={tric_result['steps']}, Converged={tric_result['converged']}")
        
        # Compare with BFGS baseline
        energy_diff = abs(tric_result['energy'] - self.bfgs_baseline['energy'])
        geometry_rmsd = calculate_rmsd(tric_result['atoms'], self.bfgs_baseline['atoms'])
        
        logger.info(f"Energy difference: {energy_diff:.6f} eV")
        logger.info(f"Geometry RMSD: {geometry_rmsd:.6f} Å")
        
        # Success criteria
        assert energy_diff < 0.001, f"Energy difference too large: {energy_diff:.6f} eV > 0.001 eV"
        assert geometry_rmsd < 0.1, f"Geometry RMSD too large: {geometry_rmsd:.6f} Å > 0.1 Å"
        
        # Store for further analysis
        self.tric_minima_result = tric_result
        
    def test_tric_ts_optimization(self):
        """Test TRIC TS optimization and compare with Sella baseline."""
        if self.test_backend is None:
            pytest.skip("No ML backends available")
            
        # First run Sella baseline if not already done
        if not hasattr(self, 'sella_baseline'):
            self.test_sella_ts_baseline()
            
        # Create explorer with available ML backend for TRIC TS
        explorer = Explorer(
            atoms=self.ethane_ts_guess,
            backend=self.test_backend,
            target='ts',
            strategy='local',
            local_optimizer='tric'
        )
        
        # Run optimization
        result = explorer.run(fmax=0.05, steps=100)
        
        # Store TRIC results
        tric_ts_result = {
            'atoms': result['optimized_atoms'],
            'energy': result['optimized_atoms'].get_potential_energy(),
            'steps': result.get('steps_taken', 0),
            'converged': result.get('converged', False)
        }
        
        logger.info(f"TRIC TS: Energy={tric_ts_result['energy']:.6f} eV, "
                   f"Steps={tric_ts_result['steps']}, Converged={tric_ts_result['converged']}")
        
        # Compare with Sella baseline
        energy_diff = abs(tric_ts_result['energy'] - self.sella_baseline['energy'])
        geometry_rmsd = calculate_rmsd(tric_ts_result['atoms'], self.sella_baseline['atoms'])
        
        logger.info(f"TS Energy difference: {energy_diff:.6f} eV")
        logger.info(f"TS Geometry RMSD: {geometry_rmsd:.6f} Å")
        
        # Success criteria (more lenient for TS)
        assert energy_diff < 0.01, f"TS Energy difference too large: {energy_diff:.6f} eV > 0.01 eV"
        assert geometry_rmsd < 0.2, f"TS Geometry RMSD too large: {geometry_rmsd:.6f} Å > 0.2 Å"
        
        # Store for further analysis
        self.tric_ts_result = tric_ts_result


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
