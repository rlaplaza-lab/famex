from __future__ import annotations

import pytest
from ase import Atoms

import famex
from tests.test_utils import StandardTestAssertions


class TestPathManager:
    def test_linear_interpolation_and_lengths(self, mock_backend):
        reactant = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        product = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

        path_mgr = famex.PathManager([reactant, product], calculator=mock_backend)
        path = path_mgr.interpolate(npoints=5, method="linear")

        assert len(path) == 5
        for structure in path:
            assert len(structure) == 3
            StandardTestAssertions.assert_reasonable_geometry(structure, "mock")

    def test_reactant_product_energies(self, mock_backend):
        reactant = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        product = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

        path_mgr = famex.PathManager([reactant, product], calculator=mock_backend)

        reactant_energy = path_mgr.reactant.energy
        product_energy = path_mgr.product.energy
        assert reactant_energy is not None
        assert product_energy is not None
        StandardTestAssertions.assert_energy_reasonable(reactant_energy, "mock")
        StandardTestAssertions.assert_energy_reasonable(product_energy, "mock")

    def test_multi_segment_interpolation(self, mock_backend):
        struct1 = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
        struct2 = Atoms("H2", positions=[[0, 0, 0], [1.0, 0, 0]])
        struct3 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])

        path_mgr = famex.PathManager([struct1, struct2, struct3], calculator=mock_backend)
        path = path_mgr.interpolate(npoints=7, method="linear")

        # Should have 7 total points across 2 segments
        assert len(path) == 7
        # All structures should have 2 atoms
        for structure in path:
            assert len(structure) == 2

    def test_calculate_rmsd(self):
        atoms1 = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        atoms2 = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

        rmsd = famex.PathManager.calculate_rmsd(atoms1, atoms2)

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

        ts_structure, ts_index = famex.PathManager.find_ts_guess(path)

        assert ts_structure is not None
        assert 0 <= ts_index < len(path)
        assert ts_structure == path[ts_index]

    def test_filter_redundant_structures(self, mock_backend):
        # Create some structures, some redundant
        struct1 = Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]])
        struct2 = Atoms("H2", positions=[[0, 0, 0], [0.701, 0, 0]])  # Very similar
        struct3 = Atoms("H2", positions=[[0, 0, 0], [1.5, 0, 0]])  # Different

        for struct in [struct1, struct2, struct3]:
            struct.calc = mock_backend

        filtered, removed, warnings = famex.PathManager.filter_redundant_structures(
            [struct1, struct2, struct3],
            rmsd_threshold=0.05,
            energy_threshold=0.01,
        )

        assert isinstance(filtered, list)
        assert isinstance(removed, list)
        assert isinstance(warnings, list)
        assert len(filtered) == 2
        assert len(removed) == 1

    def test_initialization_requires_multiple_structures(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        with pytest.raises(ValueError, match="requires at least 2 structures"):
            famex.PathManager(atoms)
        with pytest.raises(ValueError, match="requires at least 2 structures"):
            famex.PathManager([])

    def test_initialization_rejects_incompatible_structures(self):
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms2 = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        with pytest.raises(ValueError, match="Incompatible atoms"):
            famex.PathManager([atoms1, atoms2])

    def test_short_path_ts_guess(self, mock_backend):
        path = [
            Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]]),
            Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]]),
        ]
        for atoms in path:
            atoms.calc = mock_backend

        ts_structure, ts_index = famex.PathManager.find_ts_guess(path)
        assert ts_structure is not None
        assert 0 <= ts_index < len(path)
