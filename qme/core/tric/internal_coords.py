"""Internal coordinate definitions for TRIC implementation.

This module implements the internal coordinate system used by TRIC optimizer,
including bonds, angles, and dihedrals with proper gradient calculations.

The dihedral gradient implementation is based on the pysisyphus Torsion class
approach, providing accurate gradients for all internal coordinates.
"""

import numpy as np
from typing import List, Tuple, Union
from abc import ABC, abstractmethod

from .utils import Geometry, angle_between_vectors, dihedral_angle


class InternalCoord(ABC):
    """Abstract base class for internal coordinates."""
    
    def __init__(self, indices: List[int]):
        """Initialize internal coordinate.
        
        Parameters
        ----------
        indices : List[int]
            Atomic indices defining the coordinate
        """
        self.indices = indices
        self.n_atoms = len(indices)
    
    @abstractmethod
    def eval(self, positions: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """Evaluate internal coordinate and its gradients.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions (N, 3)
            
        Returns
        -------
        Tuple[float, List[np.ndarray]]
            (coordinate_value, gradient_list)
        """
        pass
    
    def __len__(self):
        return self.n_atoms
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.indices})"


class Bond(InternalCoord):
    """Bond length coordinate."""
    
    def __init__(self, i: int, j: int):
        """Initialize bond coordinate.
        
        Parameters
        ----------
        i, j : int
            Atomic indices
        """
        super().__init__([i, j])
        self.i = i
        self.j = j
    
    def eval(self, positions: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """Calculate bond length and gradient.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions (N, 3)
            
        Returns
        -------
        Tuple[float, List[np.ndarray]]
            (bond_length, [grad_i, grad_j])
        """
        # Bond vector
        bond_vec = positions[self.i] - positions[self.j]
        bond_length = np.linalg.norm(bond_vec)
        
        if bond_length < 1e-12:
            # Handle zero length bond
            return 0.0, [np.zeros(3), np.zeros(3)]
        
        # Unit vector along bond
        unit_vec = bond_vec / bond_length
        
        # Gradients: grad_i = +unit_vec, grad_j = -unit_vec
        grad_i = unit_vec
        grad_j = -unit_vec
        
        return bond_length, [grad_i, grad_j]


class Angle(InternalCoord):
    """Bond angle coordinate."""
    
    def __init__(self, i: int, j: int, k: int):
        """Initialize angle coordinate.
        
        Parameters
        ----------
        i, j, k : int
            Atomic indices (j is central atom)
        """
        super().__init__([i, j, k])
        self.i = i
        self.j = j
        self.k = k
    
    def eval(self, positions: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """Calculate bond angle and gradient.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions (N, 3)
            
        Returns
        -------
        Tuple[float, List[np.ndarray]]
            (angle_in_radians, [grad_i, grad_j, grad_k])
        """
        # Vectors from central atom
        vec_ji = positions[self.i] - positions[self.j]  # j -> i
        vec_jk = positions[self.k] - positions[self.j]  # j -> k
        
        # Calculate angle
        angle = angle_between_vectors(vec_ji, vec_jk)
        
        # Calculate gradients
        grad_i = np.zeros(3)
        grad_j = np.zeros(3)
        grad_k = np.zeros(3)
        
        # Handle near-linear case
        if np.abs(angle) < 1e-6 or np.abs(angle - np.pi) < 1e-6:
            return angle, [grad_i, grad_j, grad_k]
        
        # Lengths
        r_ji = np.linalg.norm(vec_ji)
        r_jk = np.linalg.norm(vec_jk)
        
        if r_ji < 1e-12 or r_jk < 1e-12:
            return angle, [grad_i, grad_j, grad_k]
        
        # Unit vectors
        unit_ji = vec_ji / r_ji
        unit_jk = vec_jk / r_jk
        
        # Gradient with respect to angle
        dangle_dcos = -1.0 / np.sqrt(1.0 - np.cos(angle)**2)
        
        # Gradients for each atom
        # Atom i
        grad_i = dangle_dcos * (unit_jk - np.cos(angle) * unit_ji) / r_ji
        
        # Atom k  
        grad_k = dangle_dcos * (unit_ji - np.cos(angle) * unit_jk) / r_jk
        
        # Atom j (central atom)
        grad_j = -(grad_i + grad_k)
        
        return angle, [grad_i, grad_j, grad_k]


class Dihedral(InternalCoord):
    """Dihedral angle coordinate."""
    
    def __init__(self, i: int, j: int, k: int, l: int):
        """Initialize dihedral coordinate.
        
        Parameters
        ----------
        i, j, k, l : int
            Atomic indices defining the dihedral angle
        """
        super().__init__([i, j, k, l])
        self.i = i
        self.j = j
        self.k = k
        self.l = l
    
    def eval(self, positions: np.ndarray) -> Tuple[float, List[np.ndarray]]:
        """Calculate dihedral angle and gradient.
        
        Parameters
        ----------
        positions : np.ndarray
            Atomic positions (N, 3)
            
        Returns
        -------
        Tuple[float, List[np.ndarray]]
            (dihedral_angle_in_radians, [grad_i, grad_j, grad_k, grad_l])
        """
        # Calculate dihedral angle
        angle = dihedral_angle(
            positions[self.i], positions[self.j], 
            positions[self.k], positions[self.l]
        )
        
        # Calculate dihedral gradients using pysisyphus-inspired approach
        # Based on Torsion._calculate with gradient=True from pysisyphus
        m, o, p, n = self.i, self.j, self.k, self.l
        
        # Bond vectors (following pysisyphus convention)
        u_dash = positions[m] - positions[o]  # m -> o
        v_dash = positions[n] - positions[p]  # n -> p  
        w_dash = positions[p] - positions[o]  # o -> p
        
        # Bond lengths
        u_norm = np.linalg.norm(u_dash)
        v_norm = np.linalg.norm(v_dash)
        w_norm = np.linalg.norm(w_dash)
        
        # Handle degenerate cases
        if u_norm < 1e-12 or v_norm < 1e-12 or w_norm < 1e-12:
            return angle, [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
        
        # Normalized vectors
        u = u_dash / u_norm
        v = v_dash / v_norm
        w = w_dash / w_norm
        
        # Angles between vectors
        phi_u = np.arccos(np.clip(u.dot(w), -1.0, 1.0))
        phi_v = np.arccos(np.clip(-w.dot(v), -1.0, 1.0))
        
        # Cross products
        uxw = np.cross(u, w)
        vxw = np.cross(v, w)
        
        # Avoid division by zero
        sin_phi_u = np.sin(phi_u)
        sin_phi_v = np.sin(phi_v)
        
        if sin_phi_u < 1e-12 or sin_phi_v < 1e-12:
            return angle, [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
        
        # Gradient terms (following pysisyphus formulas)
        sin2_u = sin_phi_u ** 2
        sin2_v = sin_phi_v ** 2
        
        first_term = uxw / (u_norm * sin2_u)
        second_term = vxw / (v_norm * sin2_v)
        third_term = uxw * np.cos(phi_u) / (w_norm * sin2_u)
        fourth_term = -vxw * np.cos(phi_v) / (w_norm * sin2_v)
        
        # Gradients for each atom
        grad_m = first_term
        grad_n = -second_term
        grad_o = -first_term + third_term - fourth_term
        grad_p = second_term - third_term + fourth_term
        
        return angle, [grad_m, grad_o, grad_p, grad_n]


class InternalCoords:
    """Container for internal coordinates."""
    
    def __init__(self, geometry: Geometry):
        """Initialize internal coordinate system.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
        """
        self.geometry = geometry
        self.coords: List[InternalCoord] = []
        
        # Generate internal coordinates
        self._generate_coordinates()
    
    def _generate_coordinates(self):
        """Generate internal coordinates from geometry using smart connectivity."""
        from .utils import find_angles, find_dihedrals
        from .connectivity import (
            find_bonds_with_connectivity, 
            validate_connectivity,
            create_connectivity_graph
        )
        
        # Find bonds with connectivity checking
        bonds = find_bonds_with_connectivity(self.geometry)
        
        # Validate connectivity
        is_valid, warnings = validate_connectivity(self.geometry, bonds)
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(f"Connectivity warning: {warning}")
        
        # CRITICAL FIX: Use systematic coordinate selection
        # to ensure we get exactly the right number of coordinates
        n_atoms = len(self.geometry)
        total_needed = 3 * n_atoms - 6
        
        # Step 1: Add essential bonds
        essential_bonds = self._select_essential_bonds(bonds)
        for i, j in essential_bonds:
            self.coords.append(Bond(i, j))
        
        # Step 2: Add essential angles
        angles = find_angles(essential_bonds)
        essential_angles = self._select_essential_angles(angles)
        for i, j, k in essential_angles:
            self.coords.append(Angle(i, j, k))
        
        # Step 3: Add essential dihedrals to reach target count
        dihedrals = find_dihedrals(essential_bonds)
        essential_dihedrals = self._select_essential_dihedrals(dihedrals)
        for i, j, k, l in essential_dihedrals:
            self.coords.append(Dihedral(i, j, k, l))
        
        # Final check: ensure we have a reasonable number of coordinates
        current_count = len(self.coords)
        
        # For small molecules, limit coordinates to avoid severe rank deficiency
        # but don't be too aggressive - we need enough coordinates for proper optimization
        if current_count > total_needed:
            # Remove excess coordinates (prioritize bonds > angles > dihedrals)
            excess = current_count - total_needed
            removed = 0
            
            # Remove excess dihedrals first
            dihedral_coords = [c for c in self.coords if c.__class__.__name__ == 'Dihedral']
            for coord in dihedral_coords:
                if removed >= excess:
                    break
                self.coords.remove(coord)
                removed += 1
            
            # Remove excess angles if still needed
            if removed < excess:
                angle_coords = [c for c in self.coords if c.__class__.__name__ == 'Angle']
                for coord in angle_coords:
                    if removed >= excess:
                        break
                    self.coords.remove(coord)
                    removed += 1
    
    def _select_essential_bonds(self, bonds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Select essential bonds to avoid redundancy.
        
        For small molecules, use a spanning tree approach to ensure connectivity
        without redundancy.
        """
        n_atoms = len(self.geometry)
        expected_bonds = max(0, n_atoms - 1)  # Spanning tree for connectivity
        
        if len(bonds) <= expected_bonds:
            return bonds
        
        # For H2O-like molecules, prioritize bonds to central atom
        if n_atoms == 3:
            # Find central atom (most connected)
            connectivity = {}
            for i, j in bonds:
                connectivity[i] = connectivity.get(i, 0) + 1
                connectivity[j] = connectivity.get(j, 0) + 1
            
            central_atom = max(connectivity, key=connectivity.get)
            
            # Keep only bonds to central atom
            essential_bonds = []
            for i, j in bonds:
                if i == central_atom or j == central_atom:
                    essential_bonds.append((i, j))
            
            return essential_bonds[:expected_bonds]
        
        # For larger molecules, use spanning tree
        return bonds[:expected_bonds]
    
    def _select_essential_angles(self, angles: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Select essential angles to avoid redundancy."""
        n_atoms = len(self.geometry)
        expected_angles = max(0, n_atoms - 2)  # Rough estimate
        
        if len(angles) <= expected_angles:
            return angles
        
        # For H2O-like molecules, keep only the most important angle
        if n_atoms == 3:
            # Keep the angle with the central atom in the middle
            for i, j, k in angles:
                # Check if j is the central atom (most connected)
                connectivity = {}
                for bond in self._select_essential_bonds(self._get_all_bonds()):
                    a, b = bond
                    connectivity[a] = connectivity.get(a, 0) + 1
                    connectivity[b] = connectivity.get(b, 0) + 1
                
                if len(connectivity) > 0:
                    central_atom = max(connectivity, key=connectivity.get)
                    if j == central_atom:
                        return [(i, j, k)]
            
            # Fallback: return first angle
            return [angles[0]]
        
        return angles[:expected_angles]
    
    def _select_essential_dihedrals(self, dihedrals: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Select essential dihedrals to avoid redundancy and rank deficiency.
        
        This is the CRITICAL FIX for the singular G-matrix issue.
        We need to limit dihedrals to avoid creating too many redundant coordinates.
        """
        n_atoms = len(self.geometry)
        
        # For a molecule with N atoms, we need 3N-6 internal coordinates total
        # We already have bonds and angles, so limit dihedrals accordingly
        bonds_count = len([c for c in self.coords if c.__class__.__name__ == 'Bond'])
        angles_count = len([c for c in self.coords if c.__class__.__name__ == 'Angle'])
        
        # Total coordinates we should have: 3*N - 6
        total_needed = 3 * n_atoms - 6
        dihedrals_needed = max(0, total_needed - bonds_count - angles_count)
        
        # CRITICAL FIX: Limit dihedrals more aggressively
        # For small molecules, use even fewer dihedrals to ensure full rank
        if n_atoms <= 10:  # Small molecules
            max_dihedrals = min(3, dihedrals_needed, len(dihedrals))
        else:
            max_dihedrals = min(n_atoms - 3, dihedrals_needed, len(dihedrals))
        
        if len(dihedrals) <= max_dihedrals:
            return dihedrals
        
        # For small molecules like ethane, prioritize dihedrals that span the longest chains
        # This helps maintain connectivity and avoids redundant local rotations
        
        # Score dihedrals based on:
        # 1. Distance between end atoms (longer chains preferred)
        # 2. Whether they span different branches
        # 3. Avoid dihedrals that are too similar
        
        dihedral_scores = []
        for dihedral in dihedrals:
            i, j, k, l = dihedral
            score = 0.0
            
            # Distance score - prefer dihedrals that span longer distances
            pos_i = self.geometry.positions[i]
            pos_l = self.geometry.positions[l]
            distance = np.linalg.norm(pos_i - pos_l)
            score += distance * 2.0  # Weight distance heavily
            
            # Connectivity score - prefer dihedrals that connect different branches
            # Count how many atoms are connected to the central atoms j and k
            j_connections = sum(1 for bond in self._get_all_bonds() if j in bond)
            k_connections = sum(1 for bond in self._get_all_bonds() if k in bond)
            score += (j_connections + k_connections) * 0.5
            
            # Penalty for very short dihedrals (likely redundant)
            if distance < 2.0:
                score -= 1.0
                
            dihedral_scores.append((score, dihedral))
        
        # Sort by score (highest first) and take the best ones
        dihedral_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Take the top dihedrals, but ensure we don't exceed the limit
        selected_dihedrals = []
        for score, dihedral in dihedral_scores[:max_dihedrals]:
            selected_dihedrals.append(dihedral)
        
        return selected_dihedrals
    
    def _get_all_bonds(self) -> List[Tuple[int, int]]:
        """Get all bonds from the original connectivity analysis."""
        from .connectivity import find_bonds_with_connectivity
        return find_bonds_with_connectivity(self.geometry)
        
        # Create connectivity graph for analysis
        graph = create_connectivity_graph(self.geometry, bonds)
        
        # Score dihedrals based on connectivity
        dihedral_scores = []
        for dihedral in dihedrals:
            score = self._score_dihedral(dihedral, graph)
            dihedral_scores.append((score, dihedral))
        
        # Sort by score (highest first) and take top dihedrals
        dihedral_scores.sort(key=lambda x: x[0], reverse=True)
        max_dihedrals = min(50, len(dihedral_scores))  # Limit to 50 dihedrals
        
        for score, (i, j, k, l) in dihedral_scores[:max_dihedrals]:
            self.coords.append(Dihedral(i, j, k, l))
    
    def _score_dihedral(self, dihedral: Tuple[int, int, int, int], graph) -> float:
        """Score a dihedral based on connectivity importance.
        
        Parameters
        ----------
        dihedral : Tuple[int, int, int, int]
            Dihedral atom indices (i, j, k, l)
        graph : ConnectivityGraph
            Connectivity graph
            
        Returns
        -------
        float
            Score (higher = more important)
        """
        i, j, k, l = dihedral
        score = 0.0
        
        # Bonus for dihedrals in rings
        if self._is_in_ring(dihedral, graph):
            score += 10.0
        
        # Bonus for dihedrals along backbone (atoms with degree >= 2)
        backbone_atoms = sum(1 for atom in dihedral if graph.get_degree(atom) >= 2)
        score += backbone_atoms * 2.0
        
        # Bonus for dihedrals with central atoms having higher degree
        score += graph.get_degree(j) + graph.get_degree(k)
        
        # Penalty for dihedrals involving terminal atoms (degree 1)
        terminal_atoms = sum(1 for atom in dihedral if graph.get_degree(atom) == 1)
        score -= terminal_atoms * 1.0
        
        return score
    
    def _is_in_ring(self, dihedral: Tuple[int, int, int, int], graph) -> bool:
        """Check if a dihedral is part of a ring.
        
        This is a simplified check - a more sophisticated implementation
        would use proper ring detection algorithms.
        """
        i, j, k, l = dihedral
        
        # Check if there's a path from i to l that doesn't go through j or k
        # This is a heuristic for ring detection
        try:
            # Find shortest path between i and l
            path = graph.find_shortest_path(i, l)
            if path and len(path) > 2:
                # Check if path avoids j and k
                if j not in path[1:-1] and k not in path[1:-1]:
                    return True
        except:
            pass
        
        return False
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx]
    
    def eval_geometry(self, geometry: Geometry) -> np.ndarray:
        """Evaluate all internal coordinates for a geometry.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
            
        Returns
        -------
        np.ndarray
            Array of internal coordinate values
        """
        values = np.zeros(len(self.coords))
        positions = geometry.positions
        
        for i, coord in enumerate(self.coords):
            value, _ = coord.eval(positions)
            values[i] = value
        
        return values
    
    def B_matrix(self, geometry: Geometry) -> np.ndarray:
        """Calculate B-matrix (Jacobian) for coordinate transformation.
        
        Parameters
        ----------
        geometry : Geometry
            Molecular geometry
            
        Returns
        -------
        np.ndarray
            B-matrix of shape (n_internal, 3*n_atoms)
        """
        n_internal = len(self.coords)
        n_atoms = len(geometry)
        
        B = np.zeros((n_internal, 3 * n_atoms))
        positions = geometry.positions
        
        for i, coord in enumerate(self.coords):
            _, gradients = coord.eval(positions)
            
            # Fill B-matrix row
            for j, grad in enumerate(gradients):
                atom_idx = coord.indices[j]
                B[i, 3*atom_idx:3*(atom_idx+1)] = grad
        
        return B
    
    def update_geometry(self, geometry: Geometry, q: np.ndarray, dq: np.ndarray, 
                       B_inv: np.ndarray, max_iter: int = 20, 
                       tolerance: float = 1e-6) -> Tuple[np.ndarray, Geometry]:
        """Update geometry using internal coordinate step.
        
        Parameters
        ----------
        geometry : Geometry
            Current geometry
        q : np.ndarray
            Current internal coordinates
        dq : np.ndarray
            Internal coordinate step
        B_inv : np.ndarray
            Pseudo-inverse of B-matrix
        max_iter : int, default 20
            Maximum iterations
        tolerance : float, default 1e-6
            Convergence tolerance
            
        Returns
        -------
        Tuple[np.ndarray, Geometry]
            (updated_internal_coords, updated_geometry)
        """
        geom = geometry.copy()
        
        for iteration in range(max_iter):
            # Calculate Cartesian step
            cartesian_step = B_inv @ dq
            cartesian_step = cartesian_step.reshape(-1, 3)
            
            # Update geometry
            geom.positions += cartesian_step
            
            # Calculate new internal coordinates
            q_new = self.eval_geometry(geom)
            
            # Calculate residual
            dq_residual = dq - (q_new - q)
            
            # Check convergence
            if np.linalg.norm(dq_residual) < tolerance:
                break
            
            # Update for next iteration
            dq = dq_residual
            q = q_new
        
        return q_new, geom
