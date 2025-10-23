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

    @pytest.mark.parametrize(
        ("constraint_string", "expected_atoms"),
        [
            ("fix 0", [0]),
            ("fix 1", [1]),
            ("fix 0,2", [0, 2]),
        ],
    )
    def test_parse_fixed_atoms(self, constraint_string, expected_atoms) -> None:
        cm = parse_constraint_string(constraint_string, self.atoms)
        info = cm.get_constraint_info()
        assert info["fixed_atoms"] == expected_atoms

    @pytest.mark.parametrize(
        ("constraint_string", "expected_atoms", "expected_k"),
        [
            ("harmonic_bond 0,1 k=5.0", [0, 1], 5.0),
            ("harmonic_bond 1,2 k=3.0", [1, 2], 3.0),
            ("harmonic_bond 0,2 k=2.5", [0, 2], 2.5),
        ],
    )
    def test_parse_harmonic_bond(self, constraint_string, expected_atoms, expected_k) -> None:
        cm = parse_constraint_string(constraint_string, self.atoms)
        info = cm.get_constraint_info()
        hc = info["harmonic_constraints"][0]
        assert hc["type"] == "bond"
        assert hc["atoms"] == expected_atoms
        assert hc["force_constant"] == expected_k

    def test_parse_invalid(self) -> None:
        with pytest.raises(ValueError):
            parse_constraint_string("invalid 0,1", self.atoms)


class TestConstraintValidation:
    def setup_method(self) -> None:
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])

    @pytest.mark.parametrize("indices", [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]])
    def test_validate_valid_indices(self, indices) -> None:
        assert validate_atom_indices(indices, self.atoms) is True

    @pytest.mark.parametrize("invalid_indices", [[0, 3], [1, 4], [0, 1, 3]])
    def test_validate_invalid_indices(self, invalid_indices) -> None:
        with pytest.raises(ValueError):
            validate_atom_indices(invalid_indices, self.atoms)

    @pytest.mark.parametrize("non_integer_indices", [[0, 1.5], [1.0, 2], [0.5]])
    def test_validate_non_integer(self, non_integer_indices) -> None:
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
