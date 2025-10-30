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


class TestConstraintManager:
    def setup_method(self) -> None:
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
        self.constraint_manager = QMEConstraintManager(self.atoms)

    def test_initialization(self) -> None:
        assert len(self.constraint_manager.constraints) == 0

    @pytest.mark.parametrize("atom_indices", [[0], [1], [2], [0, 2]])
    def test_add_fixed_atoms(self, atom_indices) -> None:
        self.constraint_manager.add_fixed_atoms(atom_indices)
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(self.constraint_manager.constraints[0], FixedAtomsConstraint)

        info = self.constraint_manager.get_constraint_info()
        assert info["fixed_atoms"] == atom_indices

    @pytest.mark.parametrize(
        ("atoms", "force_constant"), [([0, 1], 5.0), ([1, 2], 3.0), ([0, 2], 2.5)]
    )
    def test_add_harmonic_bond_constraint(self, atoms, force_constant) -> None:
        self.constraint_manager.add_harmonic_constraint("bond", atoms, force_constant)
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(self.constraint_manager.constraints[0], HarmonicBondConstraint)

        info = self.constraint_manager.get_constraint_info()
        assert len(info["harmonic_constraints"]) == 1
        hc = info["harmonic_constraints"][0]
        assert hc["type"] == "bond"
        assert hc["atoms"] == atoms
        assert hc["force_constant"] == force_constant

    @pytest.mark.parametrize(("atoms", "force_constant"), [([1, 0, 2], 2.0), ([0, 1, 2], 1.5)])
    def test_add_harmonic_angle_constraint(self, atoms, force_constant) -> None:
        self.constraint_manager.add_harmonic_constraint("angle", atoms, force_constant)
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(self.constraint_manager.constraints[0], HarmonicAngleConstraint)

    def test_invalid_constraint_type(self) -> None:
        with pytest.raises(ValueError):
            self.constraint_manager.add_harmonic_constraint("invalid", [0, 1])


class TestConstraintParsing:
    def setup_method(self) -> None:
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])

    def test_parse_fixed_atoms(self) -> None:
        """Test parsing fixed atom constraints with various inputs."""
        test_cases = [("fix 0", [0]), ("fix 0,2", [0, 2])]
        for constraint_string, expected_atoms in test_cases:
            cm = parse_constraint_string(constraint_string, self.atoms)
            info = cm.get_constraint_info()
            assert info["fixed_atoms"] == expected_atoms

    def test_parse_harmonic_constraints(self) -> None:
        """Test parsing harmonic constraints."""
        test_cases = [
            ("harmonic_bond 0,1 k=5.0", "bond", [0, 1], 5.0),
            ("harmonic_bond 0,2 k=2.5", "bond", [0, 2], 2.5),
        ]
        for constraint_string, expected_type, expected_atoms, expected_k in test_cases:
            cm = parse_constraint_string(constraint_string, self.atoms)
            info = cm.get_constraint_info()
            hc = info["harmonic_constraints"][0]
            assert hc["type"] == expected_type
            assert hc["atoms"] == expected_atoms
            assert hc["force_constant"] == expected_k

    def test_parse_invalid(self) -> None:
        with pytest.raises(ValueError):
            parse_constraint_string("invalid 0,1", self.atoms)


class TestConstraintValidation:
    def setup_method(self) -> None:
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])

    def test_validate_valid_indices(self) -> None:
        """Test validation with various valid index combinations."""
        valid_cases = [[0], [2], [0, 1], [1, 2], [0, 1, 2]]
        for indices in valid_cases:
            assert validate_atom_indices(indices, self.atoms) is True

    def test_validate_invalid_indices(self) -> None:
        """Test validation with various invalid index combinations."""
        invalid_cases = [[0, 3], [1, 4], [0, 1, 3]]
        for invalid_indices in invalid_cases:
            with pytest.raises(ValueError):
                validate_atom_indices(invalid_indices, self.atoms)

    def test_validate_non_integer(self) -> None:
        """Test validation rejects non-integer indices."""
        non_integer_cases = [[0, 1.5], [1.0, 2], [0.5]]
        for non_integer_indices in non_integer_cases:
            with pytest.raises(ValueError):
                validate_atom_indices(non_integer_indices, self.atoms)  # type: ignore


class TestHarmonicConstraintInternals:
    def setup_method(self) -> None:
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])

    def test_bond_reference_calculation(self) -> None:
        c = HarmonicBondConstraint([0, 1], self.atoms, 5.0)
        expected = np.linalg.norm(self.atoms.positions[1] - self.atoms.positions[0])
        assert abs(c.reference_value - expected) < 1e-10

    def test_angle_reference_validation(self) -> None:
        with pytest.raises(ValueError):
            HarmonicAngleConstraint([0, 1], self.atoms, 2.0)
