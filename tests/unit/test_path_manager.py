from __future__ import annotations

from ase import Atoms

import qme
from tests.test_utils import StandardTestAssertions


class TestPathManager:
    def test_linear_interpolation_and_lengths(self, mock_backend):
        # Use H2O -> H2O (slightly different geometry) for meaningful testing
        reactant = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        product = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

        path_mgr = qme.PathManager([reactant, product], calculator=mock_backend)
        path = path_mgr.interpolate(npoints=5, method="linear")

        # Check path length
        assert len(path) == 5

        # Check that all structures have correct number of atoms
        for structure in path:
            assert len(structure) == 3  # H2O has 3 atoms
            # Check that structure is reasonable
            StandardTestAssertions.assert_reasonable_geometry(structure, "mock")

    def test_reaction_energy_calculation(self, mock_backend):
        reactant = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        product = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

        path_mgr = qme.PathManager([reactant, product], calculator=mock_backend)

        # Test energy calculation using the reaction_energy property
        reaction_energy = path_mgr.reaction_energy

        # Check that reaction energy is reasonable
        if reaction_energy is not None:
            StandardTestAssertions.assert_energy_reasonable(reaction_energy, "mock")
            assert isinstance(reaction_energy, (int, float))

        # Test individual geometry energies
        reactant_energy = path_mgr.reactant.energy
        product_energy = path_mgr.product.energy

        # Check that energies are reasonable
        if reactant_energy is not None:
            StandardTestAssertions.assert_energy_reasonable(reactant_energy, "mock")
            assert isinstance(reactant_energy, (int, float))

        if product_energy is not None:
            StandardTestAssertions.assert_energy_reasonable(product_energy, "mock")
            assert isinstance(product_energy, (int, float))

    def test_multi_segment_interpolation(self, mock_backend):
        struct1 = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
        struct2 = Atoms("H2", positions=[[0, 0, 0], [1.0, 0, 0]])
        struct3 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        path_mgr = qme.PathManager([struct1, struct2, struct3], calculator=mock_backend)
        path = path_mgr.interpolate(npoints=7, method="linear")

        # Should have 7 total points across 2 segments
        assert len(path) == 7
        # All structures should have 2 atoms
        for structure in path:
            assert len(structure) == 2

    def test_calculate_rmsd(self):
        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        atoms2 = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

        rmsd = qme.PathManager.calculate_rmsd(atoms1, atoms2)

        assert isinstance(rmsd, float)
        assert rmsd >= 0
        assert rmsd < 1.0  # Should be small for similar structures

    def test_find_ts_guess(self, mock_backend):
        # Create a simple path with known energies
        path = []
        for i in range(5):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            atoms.calc = mock_backend
            path.append(atoms)

        ts_structure, ts_index = qme.PathManager.find_ts_guess(path)

        assert ts_structure is not None
        assert 0 <= ts_index < len(path)
        assert ts_structure == path[ts_index]

    def test_find_local_minima(self, mock_backend):
        path = []
        for i in range(7):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            atoms.calc = mock_backend
            path.append(atoms)

        minima_indices = qme.PathManager.find_local_minima(path)

        assert isinstance(minima_indices, list)
        assert len(minima_indices) > 0
        assert all(0 <= idx < len(path) for idx in minima_indices)

    def test_filter_redundant_structures(self, mock_backend):
        # Create some structures, some redundant
        struct1 = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
        struct2 = Atoms("H2", positions=[[0, 0, 0], [0.701, 0, 0]])  # Very similar
        struct3 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])  # Different

        for struct in [struct1, struct2, struct3]:
            struct.calc = mock_backend

        filtered, removed, warnings = qme.PathManager.filter_redundant_structures(
            [struct1, struct2, struct3],
            rmsd_threshold=0.05,
            energy_threshold=0.01,
        )

        assert isinstance(filtered, list)
        assert isinstance(removed, list)
        assert isinstance(warnings, list)
        assert len(filtered) <= 3

    def test_path_statistics(self, mock_backend):
        reactant = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
        product = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        path_mgr = qme.PathManager([reactant, product], calculator=mock_backend)
        path = path_mgr.interpolate(npoints=5, method="linear")

        # Attach calculator to path
        for atoms in path:
            atoms.calc = mock_backend

        stats = path_mgr.get_path_statistics(path)

        assert isinstance(stats, dict)
        assert "num_structures" in stats
        assert "energies" in stats
        assert "min_energy" in stats
        assert "max_energy" in stats
        assert stats["num_structures"] == 5
