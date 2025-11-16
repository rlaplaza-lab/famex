"""Additional tests for PathManager to improve coverage."""

from __future__ import annotations

import pytest
from ase import Atoms

from qme.io.path_manager import PathManager


class TestPathManagerErrorHandling:
    """Test error handling paths in PathManager."""

    def test_initialization_with_single_structure(self):
        """Test that PathManager raises error with single structure."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

        with pytest.raises(ValueError, match="requires at least 2 structures"):
            PathManager(atoms)

    def test_initialization_with_empty_list(self):
        """Test that PathManager raises error with empty list."""
        with pytest.raises(ValueError, match="requires at least 2 structures"):
            PathManager([])

    def test_get_path_statistics_with_missing_energies(self, mock_backend):
        """Test get_path_statistics handles missing energies gracefully."""
        # Create path with some structures missing calculators
        path = []
        for i in range(5):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            # Only attach calculator to some structures
            if i % 2 == 0:
                atoms.calc = mock_backend
            path.append(atoms)

        path_mgr = PathManager([path[0], path[-1]], calculator=mock_backend)
        stats = path_mgr.get_path_statistics(path)

        # Should handle missing energies gracefully
        assert "num_structures" in stats
        assert stats["num_structures"] == 5
        assert "energies" in stats
        assert len(stats["energies"]) == 5

    def test_get_path_statistics_with_all_nan_energies(self):
        """Test get_path_statistics handles all NaN energies."""
        # Create path without calculators
        path = []
        for i in range(3):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            # No calculator attached
            path.append(atoms)

        path_mgr = PathManager([path[0], path[-1]])
        stats = path_mgr.get_path_statistics(path)

        # Should handle all NaN energies
        assert "num_structures" in stats
        assert stats["num_structures"] == 3
        assert (
            stats["min_energy"] is None or stats["min_energy"] != stats["min_energy"]
        )  # NaN check
        assert (
            stats["max_energy"] is None or stats["max_energy"] != stats["max_energy"]
        )  # NaN check

    def test_get_path_statistics_handles_ts_finding_errors(self, mock_backend):
        """Test get_path_statistics handles TS finding errors gracefully."""
        # Create path that might cause issues in TS finding
        path = []
        for i in range(2):  # Very short path
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            atoms.calc = mock_backend
            path.append(atoms)

        path_mgr = PathManager([path[0], path[-1]], calculator=mock_backend)
        stats = path_mgr.get_path_statistics(path)

        # Should handle TS finding errors gracefully
        assert "ts_index" in stats
        # ts_index might be None if TS finding fails

    def test_get_path_statistics_handles_minima_finding_errors(self, mock_backend):
        """Test get_path_statistics handles minima finding errors gracefully."""
        path = []
        for i in range(2):  # Very short path
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            atoms.calc = mock_backend
            path.append(atoms)

        path_mgr = PathManager([path[0], path[-1]], calculator=mock_backend)
        stats = path_mgr.get_path_statistics(path)

        # Should handle minima finding errors gracefully
        assert "minima_indices" in stats
        assert isinstance(stats["minima_indices"], list)

    def test_get_path_statistics_handles_maxima_finding_errors(self, mock_backend):
        """Test get_path_statistics handles maxima finding errors gracefully."""
        path = []
        for i in range(2):  # Very short path
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            atoms.calc = mock_backend
            path.append(atoms)

        path_mgr = PathManager([path[0], path[-1]], calculator=mock_backend)
        stats = path_mgr.get_path_statistics(path)

        # Should handle maxima finding errors gracefully
        assert "maxima_indices" in stats
        assert isinstance(stats["maxima_indices"], list)

    def test_find_ts_guess_with_short_path(self, mock_backend):
        """Test find_ts_guess with very short path."""
        path = []
        for i in range(2):  # Minimum path length
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            atoms.calc = mock_backend
            path.append(atoms)

        # Should handle short paths
        ts_structure, ts_index = PathManager.find_ts_guess(path)
        assert ts_structure is not None
        assert 0 <= ts_index < len(path)

    def test_find_local_minima_with_short_path(self, mock_backend):
        """Test find_local_minima with very short path."""
        path = []
        for i in range(2):  # Minimum path length
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            atoms.calc = mock_backend
            path.append(atoms)

        # Should handle short paths
        minima_indices = PathManager.find_local_minima(path)
        assert isinstance(minima_indices, list)

    def test_find_local_maxima_with_short_path(self, mock_backend):
        """Test find_local_maxima with very short path."""
        path = []
        for i in range(2):  # Minimum path length
            atoms = Atoms("H2", positions=[[0, 0, 0], [0.7 + i * 0.1, 0, 0]])
            atoms.calc = mock_backend
            path.append(atoms)

        # Should handle short paths
        maxima_indices = PathManager.find_local_maxima(path)
        assert isinstance(maxima_indices, list)

    def test_initialization_with_incompatible_structures(self):
        """Test that PathManager validates structure compatibility."""
        atoms1 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        atoms2 = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])

        # Should raise error for incompatible structures
        with pytest.raises((ValueError, AssertionError)):
            PathManager([atoms1, atoms2])
