"""Test proton transfer reaction: HCl + NH3 → NH4+ + Cl-

This is a classic acid-base reaction that demonstrates
proton transfer mechanisms and hydrogen bonding.

Inspired by pysisyphus proton transfer tests.
"""

import pytest
import numpy as np
from qme import Geometry, Reaction, MLPCalculator, HarmonicCalculator


class TestProtonTransferReaction:
    """Test suite for acid-base proton transfer reaction."""
    
    @pytest.fixture
    def proton_transfer_reactants(self):
        """HCl + NH3 reactant complex."""
        # HCl approaching NH3
        atoms = ["N", "H", "H", "H", "H", "Cl"]
        coords = np.array([
            # NH3 
            0.0, 0.0, 0.0,      # N
            1.0, 0.0, 0.0,      # H
            -0.5, 0.87, 0.0,    # H  
            -0.5, -0.87, 0.0,   # H
            # HCl approaching
            3.0, 0.0, 0.0,      # H (will transfer)
            4.5, 0.0, 0.0,      # Cl
        ])
        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)
    
    @pytest.fixture
    def proton_transfer_products(self):
        """NH4+ + Cl- product complex."""
        atoms = ["N", "H", "H", "H", "H", "Cl"]
        coords = np.array([
            # NH4+
            0.0, 0.0, 0.0,      # N
            1.0, 0.0, 0.0,      # H
            -0.33, 0.94, 0.0,   # H  
            -0.33, -0.47, 0.82, # H
            -0.33, -0.47, -0.82, # H (transferred proton)
            # Cl- (farther away)
            -3.0, 0.0, 0.0,     # Cl
        ])
        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)
    
    @pytest.fixture  
    def proton_transfer_ts(self):
        """Transition state with proton partially transferred."""
        atoms = ["N", "H", "H", "H", "H", "Cl"]
        coords = np.array([
            # NH3 with approaching proton
            0.0, 0.0, 0.0,      # N
            1.0, 0.0, 0.0,      # H
            -0.5, 0.87, 0.0,    # H
            -0.5, -0.87, 0.0,   # H
            # Transferring proton (between N and Cl)
            1.5, 0.0, 0.0,      # H (partially transferred)
            3.0, 0.0, 0.0,      # Cl
        ])
        return Geometry(atoms=atoms, coords=coords, charge=0, mult=1)
    
    def test_proton_transfer_setup(self, proton_transfer_reactants, proton_transfer_products):
        """Test basic proton transfer reaction setup."""
        reaction = Reaction(
            reactant=proton_transfer_reactants,
            product=proton_transfer_products,
            name="HCl_NH3_proton_transfer"
        )
        
        assert reaction.name == "HCl_NH3_proton_transfer"
        assert reaction.reactant.natoms == 6
        assert reaction.product.natoms == 6
        assert reaction.reactant.charge == 0
        assert reaction.product.charge == 0
    
    def test_proton_transfer_distances(self, proton_transfer_reactants, proton_transfer_products):
        """Test key distance changes in proton transfer."""
        reaction = Reaction(proton_transfer_reactants, proton_transfer_products, name="PT_distances")
        path_geoms = reaction.interpolate(npoints=15)
        
        nh_distances = []  # N-H(transferring) distance
        hcl_distances = [] # H(transferring)-Cl distance
        
        for geom in path_geoms:
            coords = geom.coords3d
            # N is atom 0, transferring H is atom 4, Cl is atom 5
            nh_dist = np.linalg.norm(coords[0] - coords[4])
            hcl_dist = np.linalg.norm(coords[4] - coords[5])
            
            nh_distances.append(nh_dist)
            hcl_distances.append(hcl_dist)
        
        nh_distances = np.array(nh_distances)
        hcl_distances = np.array(hcl_distances)
        
        # N-H distance should decrease (bond formation)
        assert nh_distances[0] > nh_distances[-1], "N-H distance should decrease"
        
        # H-Cl distance should increase (bond breaking)
        assert hcl_distances[0] < hcl_distances[-1], "H-Cl distance should increase"
        
        # Check reasonable final bond lengths
        assert nh_distances[-1] < 1.5, "Final N-H distance should be reasonable"
        assert hcl_distances[-1] > 2.0, "Final H-Cl distance should show separation"
    
    def test_proton_transfer_linearity(self, proton_transfer_reactants, proton_transfer_products):
        """Test that proton transfer follows approximately linear path."""
        reaction = Reaction(proton_transfer_reactants, proton_transfer_products, name="PT_linear")
        path_geoms = reaction.interpolate(npoints=10)
        
        for geom in path_geoms:
            coords = geom.coords3d
            # Check N-H-Cl angle (should be close to linear)
            n_pos = coords[0]   # N
            h_pos = coords[4]   # transferring H
            cl_pos = coords[5]  # Cl
            
            # Vectors
            nh_vec = h_pos - n_pos
            hcl_vec = cl_pos - h_pos
            
            # Angle between N-H and H-Cl vectors
            if np.linalg.norm(nh_vec) > 1e-6 and np.linalg.norm(hcl_vec) > 1e-6:
                cos_angle = np.dot(nh_vec, hcl_vec) / (np.linalg.norm(nh_vec) * np.linalg.norm(hcl_vec))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Should be close to linear (angle close to π)
                assert angle > np.pi/2, f"N-H-Cl angle should be > 90°, got {np.degrees(angle):.1f}°"
    
    def test_proton_transfer_with_ts(self, proton_transfer_reactants, proton_transfer_products, proton_transfer_ts):
        """Test proton transfer with transition state."""
        reaction = Reaction(
            reactant=proton_transfer_reactants,
            product=proton_transfer_products,
            ts=proton_transfer_ts,
            name="PT_with_TS"
        )
        
        assert reaction.has_ts
        assert reaction.ts.natoms == 6
        
        # Check that TS is geometrically between reactants and products
        ts_coords = reaction.ts.coords3d
        r_coords = proton_transfer_reactants.coords3d  
        p_coords = proton_transfer_products.coords3d
        
        # Transferring proton position should be intermediate
        ts_h_pos = ts_coords[4]
        r_h_pos = r_coords[4]
        p_h_pos = p_coords[4]
        
        # Simple check: TS proton should be between reactant and product positions
        ts_nh_dist = np.linalg.norm(ts_coords[0] - ts_coords[4])  # N-H in TS
        r_nh_dist = np.linalg.norm(r_coords[0] - r_coords[4])     # N-H in reactants
        p_nh_dist = np.linalg.norm(p_coords[0] - p_coords[4])     # N-H in products
        
        assert r_nh_dist > ts_nh_dist > p_nh_dist, "TS should be geometrically intermediate"
    
    def test_proton_transfer_energetics(self, proton_transfer_reactants, proton_transfer_products):
        """Test energetics of proton transfer reaction."""
        calculator = MLPCalculator(model_type="mock_proton_transfer")
        
        calculator.calculate(proton_transfer_reactants)
        calculator.calculate(proton_transfer_products)
        
        reaction = Reaction(
            proton_transfer_reactants, 
            proton_transfer_products, 
            name="PT_energetics"
        )
        
        assert reaction.reaction_energy is not None
        print(f"Proton transfer energy: {reaction.reaction_energy:.6f} Hartree")
    
    def test_proton_transfer_forces(self, proton_transfer_reactants, proton_transfer_products):
        """Test force calculations during proton transfer."""
        calculator = MLPCalculator(model_type="mock_pt_forces")
        reaction = Reaction(proton_transfer_reactants, proton_transfer_products, name="PT_forces")
        
        path_geoms = reaction.interpolate(npoints=7)
        
        for geom in path_geoms:
            calculator.calculate(geom)
            assert geom.forces is not None
            assert len(geom.forces) == 18  # 3 * 6 atoms
            
            # Forces on transferring proton should be significant along path
            h_forces = geom.forces[12:15]  # Forces on atom 4 (transferring H)
            h_force_magnitude = np.linalg.norm(h_forces)
            
            # Transferring proton should experience forces (except at endpoints)
            coords = geom.coords3d
            nh_dist = np.linalg.norm(coords[0] - coords[4])
            
            if 1.2 < nh_dist < 2.5:  # In the transfer region
                assert h_force_magnitude > 0.001, "Transferring proton should experience significant forces"
    
    def test_proton_transfer_harmonic_model(self, proton_transfer_products):
        """Test proton transfer using harmonic calculator around equilibrium."""
        # Use product as equilibrium geometry for harmonic expansion
        harmonic_calc = HarmonicCalculator(
            equilibrium_geometry=proton_transfer_products,
            force_constant=50.0
        )
        
        # Create slightly displaced geometry
        displaced_geom = proton_transfer_products.copy()
        displaced_geom.coords += np.random.normal(0, 0.1, size=displaced_geom.coords.shape)
        
        # Calculate harmonic energy and forces
        harmonic_calc.calculate(displaced_geom)
        
        assert displaced_geom.energy is not None
        assert displaced_geom.forces is not None
        assert displaced_geom.energy > 0  # Should be positive (displacement from equilibrium)
        
        # Forces should point towards equilibrium
        displacement = displaced_geom.coords - proton_transfer_products.coords
        assert np.dot(displacement, displaced_geom.forces) < 0, "Forces should oppose displacement"
    
    def test_proton_transfer_rmsd_analysis(self, proton_transfer_reactants, proton_transfer_products):
        """Test RMSD analysis along proton transfer path."""
        reaction = Reaction(proton_transfer_reactants, proton_transfer_products, name="PT_rmsd")
        path_geoms = reaction.interpolate(npoints=12)
        
        rmsd_from_reactants, rmsd_from_products = reaction.get_rmsd_profile(path_geoms)
        
        # RMSD from reactants should increase
        assert rmsd_from_reactants[0] == 0.0
        assert rmsd_from_reactants[-1] > rmsd_from_reactants[0]
        
        # RMSD from products should decrease  
        assert rmsd_from_products[-1] == 0.0
        assert rmsd_from_products[0] > rmsd_from_products[-1]
        
        # Should show roughly monotonic behavior
        assert np.all(np.diff(rmsd_from_reactants) >= -0.1), "RMSD from reactants should generally increase"
        assert np.all(np.diff(rmsd_from_products) <= 0.1), "RMSD from products should generally decrease"
    
    def test_proton_transfer_hydrogen_bonding(self, proton_transfer_reactants, proton_transfer_products):
        """Test hydrogen bonding interactions during proton transfer."""
        reaction = Reaction(proton_transfer_reactants, proton_transfer_products, name="PT_hbond")
        path_geoms = reaction.interpolate(npoints=15)
        
        # Analyze potential hydrogen bonding during the reaction
        hbond_distances = []  # Distance between N and Cl (potential H-bond acceptors)
        
        for geom in path_geoms:
            coords = geom.coords3d
            n_pos = coords[0]   # N
            cl_pos = coords[5]  # Cl
            
            n_cl_dist = np.linalg.norm(n_pos - cl_pos)
            hbond_distances.append(n_cl_dist)
        
        hbond_distances = np.array(hbond_distances)
        
        # N-Cl distance should first decrease (complex formation) then increase (separation)
        # This suggests hydrogen bonding interactions during the transfer
        min_distance_idx = np.argmin(hbond_distances)
        
        assert min_distance_idx > 0, "Closest approach should not be at the beginning"
        assert min_distance_idx < len(hbond_distances) - 1, "Closest approach should not be at the end"
        assert hbond_distances[min_distance_idx] < 4.0, "Minimum N-Cl distance should suggest interaction"
    
    def test_proton_transfer_mechanism_validation(self, proton_transfer_reactants, proton_transfer_products):
        """Test validation of proton transfer mechanism."""
        reaction = Reaction(proton_transfer_reactants, proton_transfer_products, name="PT_mechanism")
        path_geoms = reaction.interpolate(npoints=20)
        
        # Validate that this follows expected proton transfer mechanism
        # Monitor all N-H and H-Cl distances throughout
        
        for i, geom in enumerate(path_geoms):
            coords = geom.coords3d
            
            # Count number of short N-H distances (< 1.3 Å indicates bonding)
            n_pos = coords[0]
            nh_bonds = 0
            
            for h_idx in [1, 2, 3, 4]:  # All H atoms
                nh_dist = np.linalg.norm(n_pos - coords[h_idx])
                if nh_dist < 1.3:
                    nh_bonds += 1
            
            # At the beginning, should have 3 N-H bonds (NH3)
            if i == 0:
                assert nh_bonds == 3, "Initial state should have 3 N-H bonds"
            
            # At the end, should have 4 N-H bonds (NH4+)  
            if i == len(path_geoms) - 1:
                assert nh_bonds == 4, "Final state should have 4 N-H bonds"
            
            # Throughout the reaction, should have 3 or 4 bonds (no intermediate with 2 or 5)
            assert 3 <= nh_bonds <= 4, f"Should have 3 or 4 N-H bonds, found {nh_bonds} at point {i}"