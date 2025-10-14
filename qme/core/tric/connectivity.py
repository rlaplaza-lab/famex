"""Connectivity analysis for improved internal coordinate generation.

This module provides graph-based connectivity analysis to ensure proper
bond detection and fragment handling for internal coordinate generation.
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Optional
from collections import defaultdict, deque

from .utils import Geometry, find_bonds


class ConnectivityGraph:
    """Graph representation of molecular connectivity."""
    
    def __init__(self, n_atoms: int):
        """Initialize connectivity graph.
        
        Parameters
        ----------
        n_atoms : int
            Number of atoms in the molecule
        """
        self.n_atoms = n_atoms
        self.edges = defaultdict(set)  # adjacency list
        self.bonds = []  # list of (i, j) tuples
    
    def add_bond(self, i: int, j: int):
        """Add a bond between atoms i and j."""
        if i != j and 0 <= i < self.n_atoms and 0 <= j < self.n_atoms:
            self.edges[i].add(j)
            self.edges[j].add(i)
            self.bonds.append((min(i, j), max(i, j)))
    
    def get_neighbors(self, atom: int) -> Set[int]:
        """Get all neighbors of an atom."""
        return self.edges[atom].copy()
    
    def is_connected(self, i: int, j: int) -> bool:
        """Check if two atoms are connected."""
        return j in self.edges[i]
    
    def get_connected_components(self) -> List[Set[int]]:
        """Find all connected components using DFS.
        
        Returns
        -------
        List[Set[int]]
            List of connected components, each containing atom indices
        """
        visited = set()
        components = []
        
        for atom in range(self.n_atoms):
            if atom not in visited:
                component = set()
                stack = [atom]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        # Add all neighbors to stack
                        stack.extend(self.edges[current])
                
                if component:
                    components.append(component)
        
        return components
    
    def is_fully_connected(self) -> bool:
        """Check if all atoms are in a single connected component."""
        components = self.get_connected_components()
        return len(components) == 1
    
    def get_fragments(self) -> List[Set[int]]:
        """Get all molecular fragments (connected components)."""
        return self.get_connected_components()
    
    def find_shortest_path(self, start: int, end: int) -> Optional[List[int]]:
        """Find shortest path between two atoms using BFS."""
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.edges[current]:
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_bond_count(self) -> int:
        """Get total number of bonds."""
        return len(self.bonds)
    
    def get_degree(self, atom: int) -> int:
        """Get coordination number of an atom."""
        return len(self.edges[atom])


def find_bonds_with_connectivity(
    geometry: Geometry, 
    initial_factor: float = 1.5, 
    max_factor: float = 2.0,
    step_size: float = 0.1
) -> List[Tuple[int, int]]:
    """Find bonds ensuring full connectivity.
    
    This function iteratively increases the covalent radius factor until
    all atoms are connected in a single component.
    
    Parameters
    ----------
    geometry : Geometry
        Molecular geometry
    initial_factor : float, default 1.5
        Initial covalent radius factor
    max_factor : float, default 2.0
        Maximum covalent radius factor
    step_size : float, default 0.1
        Step size for factor increase
        
    Returns
    -------
    List[Tuple[int, int]]
        List of bonds as (i, j) tuples
    """
    factor = initial_factor
    n_atoms = len(geometry)
    
    while factor <= max_factor:
        bonds = find_bonds(geometry, factor)
        
        # Create connectivity graph
        graph = ConnectivityGraph(n_atoms)
        for i, j in bonds:
            graph.add_bond(i, j)
        
        # Check if fully connected
        if graph.is_fully_connected():
            break
        
        # If not connected, increase factor
        factor += step_size
    
    # If still not connected at max factor, add bonds between fragments
    if not graph.is_fully_connected():
        bonds = _connect_fragments(geometry, bonds, max_factor)
    
    return bonds


def _connect_fragments(
    geometry: Geometry, 
    initial_bonds: List[Tuple[int, int]], 
    max_factor: float
) -> List[Tuple[int, int]]:
    """Connect disconnected fragments by adding bonds between nearest atoms.
    
    Parameters
    ----------
    geometry : Geometry
        Molecular geometry
    initial_bonds : List[Tuple[int, int]]
        Initial bonds found
    max_factor : float
        Maximum covalent radius factor
        
    Returns
    -------
    List[Tuple[int, int]]
        Extended list of bonds including fragment connections
    """
    n_atoms = len(geometry)
    positions = geometry.positions
    
    # Create initial graph
    graph = ConnectivityGraph(n_atoms)
    for i, j in initial_bonds:
        graph.add_bond(i, j)
    
    # Get fragments
    fragments = graph.get_fragments()
    
    # If only one fragment, return original bonds
    if len(fragments) <= 1:
        return initial_bonds
    
    bonds = initial_bonds.copy()
    
    # Connect fragments by finding nearest atoms between them
    while len(fragments) > 1:
        min_distance = float('inf')
        best_bond = None
        
        # Find closest pair of atoms between different fragments
        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments[i+1:], i+1):
                for atom1 in frag1:
                    for atom2 in frag2:
                        distance = np.linalg.norm(positions[atom1] - positions[atom2])
                        if distance < min_distance:
                            min_distance = distance
                            best_bond = (atom1, atom2)
        
        # Add the best bond
        if best_bond is not None:
            bonds.append(best_bond)
            graph.add_bond(best_bond[0], best_bond[1])
            # Update fragments
            fragments = graph.get_fragments()
        else:
            break
    
    return bonds


def validate_connectivity(
    geometry: Geometry, 
    bonds: List[Tuple[int, int]], 
    min_bonds_per_atom: int = 1,
    require_tree_structure: bool = False
) -> Tuple[bool, List[str]]:
    """Validate molecular connectivity.
    
    Parameters
    ----------
    geometry : Geometry
        Molecular geometry
    bonds : List[Tuple[int, int]]
        List of bonds
    min_bonds_per_atom : int, default 1
        Minimum number of bonds per atom (except terminal atoms)
    require_tree_structure : bool, default False
        Whether to require exactly N-1 bonds for tree structure
        
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_warnings)
    """
    n_atoms = len(geometry)
    warnings = []
    
    # Create connectivity graph
    graph = ConnectivityGraph(n_atoms)
    for i, j in bonds:
        graph.add_bond(i, j)
    
    # Check full connectivity
    if not graph.is_fully_connected():
        warnings.append("Molecule is not fully connected")
    
    # Check minimum bonds per atom
    for atom in range(n_atoms):
        degree = graph.get_degree(atom)
        if degree < min_bonds_per_atom:
            warnings.append(f"Atom {atom} has only {degree} bonds (minimum: {min_bonds_per_atom})")
    
    # Check tree structure if required
    if require_tree_structure:
        n_bonds = len(bonds)
        expected_bonds = n_atoms - 1
        if n_bonds != expected_bonds:
            warnings.append(f"Expected {expected_bonds} bonds for tree structure, found {n_bonds}")
    
    # Check for reasonable bond count
    n_bonds = len(bonds)
    if n_bonds < n_atoms - 1:
        warnings.append(f"Too few bonds: {n_bonds} < {n_atoms - 1} (minimum for connectivity)")
    elif n_bonds > n_atoms * 3:  # Heuristic: avoid excessive bonds
        warnings.append(f"Very many bonds: {n_bonds} (may indicate over-bonding)")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def find_rings(graph: ConnectivityGraph, max_ring_size: int = 8) -> List[List[int]]:
    """Find rings in the connectivity graph.
    
    Parameters
    ----------
    graph : ConnectivityGraph
        Connectivity graph
    max_ring_size : int, default 8
        Maximum ring size to consider
        
    Returns
    -------
    List[List[int]]
        List of rings, each as a list of atom indices
    """
    rings = []
    
    # Use DFS to find cycles
    for start_atom in range(graph.n_atoms):
        visited = set()
        path = []
        
        def dfs(current, parent):
            visited.add(current)
            path.append(current)
            
            for neighbor in graph.get_neighbors(current):
                if neighbor == parent:
                    continue
                
                if neighbor in path:
                    # Found a cycle
                    ring_start = path.index(neighbor)
                    ring = path[ring_start:]
                    if len(ring) <= max_ring_size and len(ring) >= 3:
                        rings.append(ring.copy())
                elif neighbor not in visited:
                    dfs(neighbor, current)
            
            path.pop()
        
        dfs(start_atom, -1)
    
    # Remove duplicate rings (same atoms, different starting points)
    unique_rings = []
    for ring in rings:
        ring_set = set(ring)
        is_duplicate = False
        for existing_ring in unique_rings:
            if set(existing_ring) == ring_set:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_rings.append(ring)
    
    return unique_rings


def get_backbone_atoms(graph: ConnectivityGraph) -> List[int]:
    """Identify backbone atoms (atoms in the main molecular chain).
    
    Parameters
    ----------
    graph : ConnectivityGraph
        Connectivity graph
        
    Returns
    -------
    List[int]
        List of backbone atom indices
    """
    # Simple heuristic: backbone atoms are those with degree >= 2
    # and are part of the longest path in the molecule
    backbone = []
    
    # Find atoms with degree >= 2
    for atom in range(graph.n_atoms):
        if graph.get_degree(atom) >= 2:
            backbone.append(atom)
    
    # If no atoms with degree >= 2, all atoms are backbone
    if not backbone:
        backbone = list(range(graph.n_atoms))
    
    return backbone


def create_connectivity_graph(geometry: Geometry, bonds: List[Tuple[int, int]]) -> ConnectivityGraph:
    """Create connectivity graph from geometry and bonds.
    
    Parameters
    ----------
    geometry : Geometry
        Molecular geometry
    bonds : List[Tuple[int, int]]
        List of bonds
        
    Returns
    -------
    ConnectivityGraph
        Connectivity graph
    """
    graph = ConnectivityGraph(len(geometry))
    for i, j in bonds:
        graph.add_bond(i, j)
    return graph
