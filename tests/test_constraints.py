"""
Tests for the enhanced constraint handling system.
"""

import numpy as np
import pytest
from ase import Atoms

from qme.constraints import (
    FixedAtomsConstraint,
    HarmonicAngleConstraint,
    HarmonicBondConstraint,
    QMEConstraintManager,
    parse_constraint_string,
    validate_atom_indices,
)
from qme.core import QMEOptimizer


class TestConstraintManager:
    """Test the QMEConstraintManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple water molecule
        self.atoms = Atoms(
            "H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]
        )
        self.constraint_manager = QMEConstraintManager(self.atoms)

    def test_initialization(self):
        """Test constraint manager initialization."""
        assert len(self.constraint_manager.constraints) == 0
        assert len(self.constraint_manager.reference_atoms) == 3

    def test_add_fixed_atoms(self):
        """Test adding fixed atoms constraints."""
        self.constraint_manager.add_fixed_atoms([0])
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(self.constraint_manager.constraints[0], FixedAtomsConstraint)

        info = self.constraint_manager.get_constraint_info()
        assert info["fixed_atoms"] == [0]
        assert len(info["harmonic_constraints"]) == 0

    def test_add_harmonic_bond_constraint(self):
        """Test adding harmonic bond constraints."""
        self.constraint_manager.add_harmonic_constraint("bond", [0, 1], 5.0)
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(
            self.constraint_manager.constraints[0], HarmonicBondConstraint
        )

        info = self.constraint_manager.get_constraint_info()
        assert len(info["fixed_atoms"]) == 0
        assert len(info["harmonic_constraints"]) == 1
        assert info["harmonic_constraints"][0]["type"] == "bond"
        assert info["harmonic_constraints"][0]["atoms"] == [0, 1]
        assert info["harmonic_constraints"][0]["force_constant"] == 5.0

    def test_add_harmonic_angle_constraint(self):
        """Test adding harmonic angle constraints."""
        self.constraint_manager.add_harmonic_constraint("angle", [1, 0, 2], 2.0)
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(
            self.constraint_manager.constraints[0], HarmonicAngleConstraint
        )

        info = self.constraint_manager.get_constraint_info()
        assert len(info["fixed_atoms"]) == 0
        assert len(info["harmonic_constraints"]) == 1
        assert info["harmonic_constraints"][0]["type"] == "angle"
        assert info["harmonic_constraints"][0]["atoms"] == [1, 0, 2]

    def test_invalid_constraint_type(self):
        """Test handling of invalid constraint types."""
        with pytest.raises(ValueError, match="Unknown constraint type"):
            self.constraint_manager.add_harmonic_constraint("invalid", [0, 1])

    def test_multiple_constraints(self):
        """Test adding multiple constraints."""
        self.constraint_manager.add_fixed_atoms([0])
        self.constraint_manager.add_harmonic_constraint("bond", [1, 2], 3.0)

        assert len(self.constraint_manager.constraints) == 2

        info = self.constraint_manager.get_constraint_info()
        assert info["fixed_atoms"] == [0]
        assert len(info["harmonic_constraints"]) == 1
        assert info["harmonic_constraints"][0]["type"] == "bond"
        assert info["harmonic_constraints"][0]["atoms"] == [1, 2]


class TestConstraintParsing:
    """Test constraint string parsing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms(
            "H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]
        )

    def test_parse_fixed_atoms(self):
        """Test parsing fixed atoms constraints."""
        constraint_manager = parse_constraint_string("fix 0", self.atoms)
        info = constraint_manager.get_constraint_info()
        assert info["fixed_atoms"] == [0]
        assert len(info["harmonic_constraints"]) == 0

    def test_parse_multiple_fixed_atoms(self):
        """Test parsing multiple fixed atoms."""
        constraint_manager = parse_constraint_string("fix 0,1,2", self.atoms)
        info = constraint_manager.get_constraint_info()
        assert info["fixed_atoms"] == [0, 1, 2]

    def test_parse_harmonic_bond(self):
        """Test parsing harmonic bond constraints."""
        constraint_manager = parse_constraint_string(
            "harmonic_bond 0,1 k=5.0", self.atoms
        )
        info = constraint_manager.get_constraint_info()
        assert len(info["fixed_atoms"]) == 0
        assert len(info["harmonic_constraints"]) == 1
        assert info["harmonic_constraints"][0]["type"] == "bond"
        assert info["harmonic_constraints"][0]["atoms"] == [0, 1]
        assert info["harmonic_constraints"][0]["force_constant"] == 5.0

    def test_parse_harmonic_bond_default_k(self):
        """Test parsing harmonic bond with default force constant."""
        constraint_manager = parse_constraint_string("harmonic_bond 0,1", self.atoms)
        info = constraint_manager.get_constraint_info()
        assert info["harmonic_constraints"][0]["force_constant"] == 10.0  # default

    def test_parse_harmonic_angle(self):
        """Test parsing harmonic angle constraints."""
        constraint_manager = parse_constraint_string(
            "harmonic_angle 1,0,2 k=2.0", self.atoms
        )
        info = constraint_manager.get_constraint_info()
        assert len(info["harmonic_constraints"]) == 1
        assert info["harmonic_constraints"][0]["type"] == "angle"
        assert info["harmonic_constraints"][0]["atoms"] == [1, 0, 2]
        assert info["harmonic_constraints"][0]["force_constant"] == 2.0

    def test_parse_multiple_constraints(self):
        """Test parsing multiple constraints separated by semicolons."""
        constraint_manager = parse_constraint_string(
            "fix 0; harmonic_bond 1,2 k=3.0", self.atoms
        )
        info = constraint_manager.get_constraint_info()
        assert info["fixed_atoms"] == [0]
        assert len(info["harmonic_constraints"]) == 1
        assert info["harmonic_constraints"][0]["type"] == "bond"
        assert info["harmonic_constraints"][0]["atoms"] == [1, 2]
        assert info["harmonic_constraints"][0]["force_constant"] == 3.0

    def test_parse_invalid_constraint(self):
        """Test handling of invalid constraint specifications."""
        with pytest.raises(ValueError):
            parse_constraint_string("invalid_type 0,1", self.atoms)

    def test_parse_malformed_constraint(self):
        """Test handling of malformed constraint specifications."""
        with pytest.raises(ValueError, match="Invalid constraint specification"):
            parse_constraint_string("fix", self.atoms)  # Missing atom indices


class TestConstraintValidation:
    """Test constraint validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms(
            "H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]
        )

    def test_validate_valid_indices(self):
        """Test validation of valid atom indices."""
        assert validate_atom_indices([0, 1, 2], self.atoms) is True
        assert validate_atom_indices([0], self.atoms) is True
        assert validate_atom_indices([2], self.atoms) is True

    def test_validate_invalid_indices(self):
        """Test validation of invalid atom indices."""
        with pytest.raises(ValueError, match="out of range"):
            validate_atom_indices([0, 1, 5], self.atoms)

        with pytest.raises(ValueError, match="out of range"):
            validate_atom_indices([-1], self.atoms)

    def test_validate_non_integer_indices(self):
        """Test validation of non-integer indices."""
        # Test with mixed types that will cause runtime type checking to fail
        invalid_indices = [0, 1.5]  # This will be caught by runtime validation
        with pytest.raises(ValueError, match="must be integer"):
            validate_atom_indices(invalid_indices, self.atoms)  # type: ignore


class TestConstraintIntegration:
    """Test integration of constraints with QMEOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms(
            "H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]
        )
        self.optimizer = QMEOptimizer(backend="mock")

    def test_parse_constraints_method(self):
        """Test the parse_constraints method in QMEOptimizer."""
        constraints = self.optimizer.parse_constraints(
            "fix 0,1", self.atoms, verbose=False
        )
        assert len(constraints) > 0

    def test_optimize_with_constraints_method(self):
        """Test the optimize_with_constraints method."""
        results = self.optimizer.optimize_with_constraints(
            atoms=self.atoms, constraints="fix 0", steps=5, verbose=False
        )

        assert "converged" in results
        assert "constraint_info" in results
        assert len(results["constraint_info"]["fixed_atoms"]) > 0

    def test_backward_compatibility(self):
        """Test that old constraint format still works."""
        # Test with list of pre-made ASE constraints
        from ase.constraints import FixAtoms

        ase_constraints = [FixAtoms(indices=[0])]

        constraints = self.optimizer.parse_constraints(ase_constraints, self.atoms)
        assert constraints == ase_constraints


class TestHarmonicConstraints:
    """Test specific harmonic constraint calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.atoms = Atoms(
            "H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]
        )

    def test_bond_reference_calculation(self):
        """Test that bond reference values are calculated correctly."""
        constraint = HarmonicBondConstraint([0, 1], self.atoms, 5.0)

        # Calculate expected bond length
        pos1 = self.atoms.positions[0]
        pos2 = self.atoms.positions[1]
        expected_length = np.linalg.norm(pos2 - pos1)

        assert abs(constraint.reference_value - expected_length) < 1e-10

    def test_angle_reference_calculation(self):
        """Test that angle reference values are calculated correctly."""
        constraint = HarmonicAngleConstraint([1, 0, 2], self.atoms, 2.0)

        # Calculate expected angle
        pos1 = self.atoms.positions[1]
        pos2 = self.atoms.positions[0]  # central atom
        pos3 = self.atoms.positions[2]

        vec1 = pos1 - pos2
        vec2 = pos3 - pos2

        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        expected_angle = np.arccos(cos_angle)

        assert abs(constraint.reference_value - expected_angle) < 1e-10

    def test_bond_constraint_validation(self):
        """Test bond constraint validation."""
        with pytest.raises(ValueError, match="exactly 2 atoms"):
            HarmonicBondConstraint([0], self.atoms, 5.0)

        with pytest.raises(ValueError, match="exactly 2 atoms"):
            HarmonicBondConstraint([0, 1, 2], self.atoms, 5.0)

    def test_angle_constraint_validation(self):
        """Test angle constraint validation."""
        with pytest.raises(ValueError, match="exactly 3 atoms"):
            HarmonicAngleConstraint([0, 1], self.atoms, 2.0)

        with pytest.raises(ValueError, match="exactly 3 atoms"):
            HarmonicAngleConstraint([0, 1, 2, 3], self.atoms, 2.0)
