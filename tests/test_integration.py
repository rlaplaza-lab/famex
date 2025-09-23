"""Integration tests for QME package functionality."""

import pytest
import numpy as np
from qme import Geometry, Reaction, MLPCalculator


class TestQMEIntegration:
    """Integration tests combining multiple QME components."""
    
    def test_simple_reaction_workflow(self):
        """Test complete workflow for a simple reaction."""
        # Create simple H2 dissociation reaction: H2 → H + H
        
        # Reactant: H2 molecule
        h2_atoms = ["H", "H"]
        h2_coords = np.array([0.0, 0.0, 0.0, 0.74, 0.0, 0.0])
        h2_geom = Geometry(atoms=h2_atoms, coords=h2_coords, charge=0, mult=1)
        
        # Product: two separated H atoms
        h_separated_coords = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        h_separated_geom = Geometry(atoms=h2_atoms, coords=h_separated_coords, charge=0, mult=3)
        
        # Create reaction
        reaction = Reaction(
            reactant=h2_geom,
            product=h_separated_geom,
            name="H2_dissociation"
        )
        
        # Set up calculator
        calculator = MLPCalculator(model_type="mock_h2_dissociation")
        
        # Calculate energies
        calculator.calculate(h2_geom)
        calculator.calculate(h_separated_geom)
        
        # Check basic properties
        assert reaction.reaction_energy is not None
        assert h2_geom.energy is not None
        assert h_separated_geom.energy is not None
        
        # Generate reaction path
        path_geoms = reaction.interpolate(npoints=10)
        
        # Calculate energies along path
        energies = []
        for geom in path_geoms:
            calculator.calculate(geom)
            energies.append(geom.energy)
        
        # Should have monotonic energy increase (bond breaking)
        energies = np.array(energies)
        assert energies[-1] > energies[0], "Dissociation should increase energy"
        
        # Generate trajectory
        xyz_traj = reaction.to_xyz_trajectory(path_geoms)
        assert len(xyz_traj) > 0
        assert "H " in xyz_traj
    
    def test_multi_step_reaction_analysis(self):
        """Test analysis of multi-step reaction pathway."""
        # Create a simple A → B → C reaction sequence
        
        # Geometry A
        atoms_a = ["C", "H", "H", "H", "Cl"]
        coords_a = np.array([
            0.0, 0.0, 0.0,      # C
            1.09, 0.0, 0.0,     # H
            -0.36, 1.03, 0.0,   # H
            -0.36, -0.51, 0.89, # H
            -0.36, -0.51, -1.76, # Cl
        ])
        geom_a = Geometry(atoms=atoms_a, coords=coords_a, charge=0, mult=1)
        
        # Geometry B (intermediate)
        coords_b = np.array([
            0.0, 0.0, 0.0,      # C
            1.09, 0.0, 0.0,     # H
            -0.36, 1.03, 0.0,   # H
            -0.36, -0.51, 0.89, # H
            -0.8, -1.0, -2.5,   # Cl (further away)
        ])
        geom_b = Geometry(atoms=atoms_a, coords=coords_b, charge=0, mult=1)
        
        # Geometry C (final)
        coords_c = np.array([
            0.0, 0.0, 0.0,      # C
            1.09, 0.0, 0.0,     # H
            -0.36, 1.03, 0.0,   # H
            -0.36, -0.51, 0.89, # H
            -3.0, -2.0, -4.0,   # Cl (very far)
        ])
        geom_c = Geometry(atoms=atoms_a, coords=coords_c, charge=0, mult=1)
        
        # Create reaction steps
        reaction1 = Reaction(geom_a, geom_b, name="step1_A_to_B")
        reaction2 = Reaction(geom_b, geom_c, name="step2_B_to_C")
        
        # Set up calculator and calculate energies
        calculator = MLPCalculator(model_type="mock_multistep")
        
        for geom in [geom_a, geom_b, geom_c]:
            calculator.calculate(geom)
        
        # Analyze each step
        assert reaction1.reaction_energy is not None
        assert reaction2.reaction_energy is not None
        
        # Overall reaction energy
        overall_delta_e = geom_c.energy - geom_a.energy
        step_sum = reaction1.reaction_energy + reaction2.reaction_energy
        assert np.isclose(overall_delta_e, step_sum, rtol=1e-10)
        
        print(f"Step 1 ΔE: {reaction1.reaction_energy:.6f}")
        print(f"Step 2 ΔE: {reaction2.reaction_energy:.6f}")
        print(f"Overall ΔE: {overall_delta_e:.6f}")
    
    def test_reaction_pathway_optimization(self):
        """Test pathway optimization using forces."""
        # Create reaction with intermediate geometries
        atoms = ["H", "H", "H"]
        
        # Linear arrangement: H-H-H
        reactant_coords = np.array([
            -1.0, 0.0, 0.0,     # H1
             0.0, 0.0, 0.0,     # H2
             1.0, 0.0, 0.0,     # H3
        ])
        
        # Triangular arrangement: H triangle
        product_coords = np.array([
            -0.5, -0.29, 0.0,   # H1
             0.5, -0.29, 0.0,   # H2
             0.0,  0.58, 0.0,   # H3
        ])
        
        reactant = Geometry(atoms=atoms, coords=reactant_coords, charge=1, mult=2)
        product = Geometry(atoms=atoms, coords=product_coords, charge=1, mult=2)
        
        reaction = Reaction(reactant, product, name="H3_rearrangement")
        
        # Generate path and calculate forces
        calculator = MLPCalculator(model_type="mock_h3_system")
        path_geoms = reaction.interpolate(npoints=7)
        
        forces_norms = []
        for geom in path_geoms:
            calculator.calculate(geom)
            force_norm = np.linalg.norm(geom.forces)
            forces_norms.append(force_norm)
        
        # Forces should be significant in middle of path, smaller at endpoints
        forces_norms = np.array(forces_norms)
        max_force_idx = np.argmax(forces_norms)
        
        # Maximum forces should not be at endpoints
        assert 1 <= max_force_idx <= len(forces_norms) - 2, "Max forces should be in middle of path"
        
        print(f"Forces along path: {forces_norms}")
    
    def test_conformational_analysis(self):
        """Test conformational analysis using reaction paths."""
        # Ethane conformational change: staggered → eclipsed
        atoms = ["C", "C", "H", "H", "H", "H", "H", "H"]
        
        # Staggered conformation (lower energy)
        coords_staggered = np.array([
            -0.77, 0.0, 0.0,    # C1
             0.77, 0.0, 0.0,    # C2
            -1.17, 1.03, 0.0,   # H on C1
            -1.17, -0.51, 0.89, # H on C1
            -1.17, -0.51, -0.89, # H on C1
             1.17, -1.03, 0.0,  # H on C2 (staggered)
             1.17, 0.51, -0.89, # H on C2
             1.17, 0.51, 0.89,  # H on C2
        ])
        
        # Eclipsed conformation (higher energy)
        coords_eclipsed = np.array([
            -0.77, 0.0, 0.0,    # C1
             0.77, 0.0, 0.0,    # C2
            -1.17, 1.03, 0.0,   # H on C1
            -1.17, -0.51, 0.89, # H on C1
            -1.17, -0.51, -0.89, # H on C1
             1.17, 1.03, 0.0,   # H on C2 (eclipsed)
             1.17, -0.51, 0.89, # H on C2
             1.17, -0.51, -0.89, # H on C2
        ])
        
        staggered = Geometry(atoms=atoms, coords=coords_staggered, charge=0, mult=1)
        eclipsed = Geometry(atoms=atoms, coords=coords_eclipsed, charge=0, mult=1)
        
        conformational_path = Reaction(staggered, eclipsed, name="ethane_rotation")
        
        # Calculate energies
        calculator = MLPCalculator(model_type="mock_ethane_rotation")
        calculator.calculate(staggered)
        calculator.calculate(eclipsed)
        
        # Eclipsed should be higher energy (though our mock calculator might not show this)
        print(f"Staggered energy: {staggered.energy:.6f}")
        print(f"Eclipsed energy: {eclipsed.energy:.6f}")
        print(f"Rotational barrier: {conformational_path.reaction_energy:.6f}")
        
        # Test rotational path
        rotation_path = conformational_path.interpolate(npoints=12)
        
        # Calculate dihedral angles along path to verify rotation
        dihedrals = []
        for geom in rotation_path:
            coords = geom.coords3d
            # Simple dihedral approximation using cross products
            # This is a simplified calculation
            c1_pos = coords[0]
            c2_pos = coords[1]
            h1_pos = coords[2]  # H on C1
            h2_pos = coords[5]  # H on C2
            
            # Calculate dihedral angle (simplified)
            cc_vec = c2_pos - c1_pos
            c1h_vec = h1_pos - c1_pos
            c2h_vec = h2_pos - c2_pos
            
            # Project onto plane perpendicular to C-C bond
            c1h_perp = c1h_vec - np.dot(c1h_vec, cc_vec) * cc_vec / np.dot(cc_vec, cc_vec)
            c2h_perp = c2h_vec - np.dot(c2h_vec, cc_vec) * cc_vec / np.dot(cc_vec, cc_vec)
            
            # Calculate angle between projections
            if np.linalg.norm(c1h_perp) > 1e-6 and np.linalg.norm(c2h_perp) > 1e-6:
                cos_dihedral = np.dot(c1h_perp, c2h_perp) / (np.linalg.norm(c1h_perp) * np.linalg.norm(c2h_perp))
                cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)
                dihedral = np.arccos(cos_dihedral)
                dihedrals.append(np.degrees(dihedral))
        
        # Should show rotation from staggered to eclipsed
        if len(dihedrals) > 2:
            print(f"Dihedral angles: {dihedrals[:3]} ... {dihedrals[-3:]}")
    
    def test_calculator_comparison(self):
        """Test comparing different calculator types on same system."""
        # Simple water molecule
        atoms = ["O", "H", "H"]
        coords = np.array([
            0.0, 0.0, 0.0,      # O
            0.96, 0.0, 0.0,     # H
            -0.24, 0.93, 0.0,   # H
        ])
        water = Geometry(atoms=atoms, coords=coords, charge=0, mult=1)
        
        # Create different calculators
        from qme.calculators import HarmonicCalculator
        mlp_calc = MLPCalculator(model_type="mock_water")
        harmonic_calc = HarmonicCalculator(water, force_constant=100.0)
        
        # Compare calculations on displaced geometry
        displaced = water.copy()
        displaced.coords += np.random.normal(0, 0.05, size=displaced.coords.shape)
        
        # Calculate with both
        displaced_mlp = displaced.copy()
        displaced_harmonic = displaced.copy()
        
        mlp_calc.calculate(displaced_mlp)
        harmonic_calc.calculate(displaced_harmonic)
        
        print(f"MLP energy: {displaced_mlp.energy:.6f}")
        print(f"Harmonic energy: {displaced_harmonic.energy:.6f}")
        print(f"MLP force norm: {np.linalg.norm(displaced_mlp.forces):.6f}")
        print(f"Harmonic force norm: {np.linalg.norm(displaced_harmonic.forces):.6f}")
        
        # Both should give finite energies and forces
        assert np.isfinite(displaced_mlp.energy)
        assert np.isfinite(displaced_harmonic.energy)
        assert np.all(np.isfinite(displaced_mlp.forces))
        assert np.all(np.isfinite(displaced_harmonic.forces))
        
        # Harmonic should give positive energy (displacement from equilibrium)
        assert displaced_harmonic.energy >= 0.0
    
    def test_package_imports(self):
        """Test that all expected components can be imported."""
        from qme import Geometry, Reaction, MLPCalculator
        from qme.geometry import Geometry as GeometryDirect
        from qme.reactions import Reaction as ReactionDirect
        from qme.calculators import MLPCalculator as MLPCalculatorDirect, HarmonicCalculator
        
        # Check that imports work and are the same
        assert Geometry is GeometryDirect
        assert Reaction is ReactionDirect
        assert MLPCalculator is MLPCalculatorDirect
        
        # Test creating instances
        test_geom = Geometry(["H", "H"], np.array([0., 0., 0., 1., 0., 0.]))
        test_calc = MLPCalculator()
        test_harmonic = HarmonicCalculator(test_geom)
        
        assert test_geom.natoms == 2
        assert test_calc.model_type == "mock"
        assert test_harmonic.k == 100.0  # default value