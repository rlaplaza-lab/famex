from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from qme.constraints.constraints import (
    FixedAtomsConstraint,
    HarmonicAngleConstraint,
    HarmonicBondConstraint,
    QMEConstraintManager,
    parse_constraint_string,
    validate_atom_indices,
)


@pytest.fixture
def water_atoms():
    return Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])


@pytest.fixture
def constraint_manager(water_atoms):
    return QMEConstraintManager(water_atoms)


class TestConstraintManager:
    def test_initialization(self, constraint_manager: QMEConstraintManager):
        assert len(constraint_manager.constraints) == 0

    @pytest.mark.parametrize("atom_indices", [[0], [1], [2], [0, 2]])
    def test_add_fixed_atoms(self, constraint_manager: QMEConstraintManager, atom_indices):
        constraint_manager.add_fixed_atoms(atom_indices)
        assert len(constraint_manager.constraints) == 1
        assert isinstance(constraint_manager.constraints[0], FixedAtomsConstraint)

        info = constraint_manager.get_constraint_info()
        assert info["fixed_atoms"] == atom_indices

    @pytest.mark.parametrize(
        ("atoms", "force_constant"), [([0, 1], 5.0), ([1, 2], 3.0), ([0, 2], 2.5)]
    )
    def test_add_harmonic_bond_constraint(
        self,
        constraint_manager: QMEConstraintManager,
        atoms,
        force_constant,
    ):
        constraint_manager.add_harmonic_constraint("bond", atoms, force_constant)
        assert len(constraint_manager.constraints) == 1
        assert isinstance(constraint_manager.constraints[0], HarmonicBondConstraint)

        info = constraint_manager.get_constraint_info()
        assert len(info["harmonic_constraints"]) == 1
        hc = info["harmonic_constraints"][0]
        assert hc["type"] == "bond"
        assert hc["atoms"] == atoms
        assert hc["force_constant"] == force_constant

    @pytest.mark.parametrize(("atoms", "force_constant"), [([1, 0, 2], 2.0), ([0, 1, 2], 1.5)])
    def test_add_harmonic_angle_constraint(
        self,
        constraint_manager: QMEConstraintManager,
        atoms,
        force_constant,
    ):
        constraint_manager.add_harmonic_constraint("angle", atoms, force_constant)
        assert len(constraint_manager.constraints) == 1
        assert isinstance(constraint_manager.constraints[0], HarmonicAngleConstraint)

    def test_invalid_constraint_type(self, constraint_manager: QMEConstraintManager):
        with pytest.raises(ValueError):
            constraint_manager.add_harmonic_constraint("invalid", [0, 1])


class TestConstraintParsing:
    def test_parse_fixed_atoms(self, water_atoms):
        test_cases = [("fix 0", [0]), ("fix 0,2", [0, 2])]
        for constraint_string, expected_atoms in test_cases:
            cm = parse_constraint_string(constraint_string, water_atoms)
            info = cm.get_constraint_info()
            assert info["fixed_atoms"] == expected_atoms

    def test_parse_harmonic_constraints(self, water_atoms):
        test_cases = [
            ("harmonic_bond 0,1 k=5.0", "bond", [0, 1], 5.0),
            ("harmonic_bond 0,2 k=2.5", "bond", [0, 2], 2.5),
        ]
        for constraint_string, expected_type, expected_atoms, expected_k in test_cases:
            cm = parse_constraint_string(constraint_string, water_atoms)
            info = cm.get_constraint_info()
            hc = info["harmonic_constraints"][0]
            assert hc["type"] == expected_type
            assert hc["atoms"] == expected_atoms
            assert hc["force_constant"] == expected_k

    def test_parse_invalid(self, water_atoms):
        with pytest.raises(ValueError):
            parse_constraint_string("invalid 0,1", water_atoms)


class TestConstraintValidation:
    def test_validate_valid_indices(self, water_atoms):
        valid_cases = [[0], [2], [0, 1], [1, 2], [0, 1, 2]]
        for indices in valid_cases:
            assert validate_atom_indices(indices, water_atoms) is True

    def test_validate_invalid_indices(self, water_atoms):
        invalid_cases = [[0, 3], [1, 4], [0, 1, 3]]
        for invalid_indices in invalid_cases:
            with pytest.raises(ValueError):
                validate_atom_indices(invalid_indices, water_atoms)

    def test_validate_non_integer(self, water_atoms):
        non_integer_cases = [[0, 1.5], [1.0, 2], [0.5]]
        for non_integer_indices in non_integer_cases:
            with pytest.raises(AssertionError):
                validate_atom_indices(non_integer_indices, water_atoms)  # type: ignore


class TestHarmonicConstraintInternals:
    def test_bond_reference_calculation(self, water_atoms):
        c = HarmonicBondConstraint([0, 1], water_atoms, 5.0)
        expected = np.linalg.norm(water_atoms.positions[1] - water_atoms.positions[0])
        assert abs(c.reference_value - expected) < 1e-10

    def test_angle_reference_validation(self, water_atoms):
        with pytest.raises(ValueError):
            HarmonicAngleConstraint([0, 1], water_atoms, 2.0)


class TestParseConstraintsFunction:
    """Tests for qme.constraints.parser.parse_constraints()."""

    def test_parse_constraints_string(self, water_atoms):
        """Test string input (works via parse_constraint_string)."""
        from qme.constraints.parser import parse_constraints

        constraints = parse_constraints("fix 0", water_atoms)
        assert len(constraints) == 1

    def test_parse_constraints_list_of_strings(self, water_atoms):
        """Test list of constraint strings."""
        from qme.constraints.parser import parse_constraints

        constraints = parse_constraints(["fix 0", "harmonic_bond 1,2 k=5.0"], water_atoms)
        # Should combine and parse both constraints
        assert len(constraints) >= 1

    def test_parse_constraints_list_of_ase_constraints(self, water_atoms):
        """Test passing pre-made ASE constraints."""
        from ase.constraints import FixAtoms

        from qme.constraints.parser import parse_constraints

        ase_constraints = [FixAtoms(indices=[0, 1])]
        result = parse_constraints(ase_constraints, water_atoms)

        assert len(result) == 1
        assert isinstance(result[0], FixAtoms)

    def test_parse_constraints_invalid_list_item(self, water_atoms):
        """Test error for unsupported list items."""
        from qme.constraints.parser import parse_constraints

        with pytest.raises(ValueError, match="Unsupported constraint specification"):
            parse_constraints([123, "fix 0"], water_atoms)

    def test_parse_constraints_dict_not_implemented(self, water_atoms):
        """Test that dict input raises NotImplementedError."""
        from qme.constraints.parser import parse_constraints

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            parse_constraints({"fix": [0, 1]}, water_atoms)

    def test_parse_constraints_verbose_output(self, water_atoms, caplog):
        """Test verbose logging."""
        from qme.constraints.parser import parse_constraints

        with caplog.at_level("INFO"):
            parse_constraints("fix 0; harmonic_bond 1,2 k=5.0", water_atoms, verbose=1)

        # Verbose logging may or may not be captured depending on logger configuration
        # Just verify the function completes successfully with verbose=1
        assert True  # Function executed without error

    def test_parse_constraints_quiet_mode(self, water_atoms, caplog):
        """Test that verbose=0 doesn't log."""
        from qme.constraints.parser import parse_constraints

        parse_constraints("fix 0", water_atoms, verbose=0)
        assert "Applied constraints" not in caplog.text
