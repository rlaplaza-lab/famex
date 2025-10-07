import numpy as np
import pytest
from ase import Atoms

from qme.core.constraints import (
    FixedAtomsConstraint,
    HarmonicAngleConstraint,
    HarmonicBondConstraint,
    QMEConstraintManager,
    parse_constraint_string,
    validate_atom_indices,
)


class TestConstraintManager:
    def setup_method(self):
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
        self.constraint_manager = QMEConstraintManager(self.atoms)

    def test_initialization(self):
        assert len(self.constraint_manager.constraints) == 0

    def test_add_fixed_atoms(self):
        self.constraint_manager.add_fixed_atoms([0])
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(self.constraint_manager.constraints[0], FixedAtomsConstraint)

        info = self.constraint_manager.get_constraint_info()
        assert info["fixed_atoms"] == [0]

    def test_add_harmonic_bond_constraint(self):
        self.constraint_manager.add_harmonic_constraint("bond", [0, 1], 5.0)
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(self.constraint_manager.constraints[0], HarmonicBondConstraint)

        info = self.constraint_manager.get_constraint_info()
        assert len(info["harmonic_constraints"]) == 1
        hc = info["harmonic_constraints"][0]
        assert hc["type"] == "bond"
        assert hc["atoms"] == [0, 1]
        assert hc["force_constant"] == 5.0

    def test_add_harmonic_angle_constraint(self):
        self.constraint_manager.add_harmonic_constraint("angle", [1, 0, 2], 2.0)
        assert len(self.constraint_manager.constraints) == 1
        assert isinstance(self.constraint_manager.constraints[0], HarmonicAngleConstraint)

    def test_invalid_constraint_type(self):
        with pytest.raises(ValueError):
            self.constraint_manager.add_harmonic_constraint("invalid", [0, 1])


class TestConstraintParsing:
    def setup_method(self):
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])

    def test_parse_fixed_atoms(self):
        cm = parse_constraint_string("fix 0", self.atoms)
        info = cm.get_constraint_info()
        assert info["fixed_atoms"] == [0]

    def test_parse_harmonic_bond(self):
        cm = parse_constraint_string("harmonic_bond 0,1 k=5.0", self.atoms)
        info = cm.get_constraint_info()
        hc = info["harmonic_constraints"][0]
        assert hc["type"] == "bond"
        assert hc["atoms"] == [0, 1]
        assert hc["force_constant"] == 5.0

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            parse_constraint_string("invalid 0,1", self.atoms)


class TestConstraintValidation:
    def setup_method(self):
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])

    def test_validate_valid_indices(self):
        assert validate_atom_indices([0, 1, 2], self.atoms) is True

    def test_validate_invalid_indices(self):
        with pytest.raises(ValueError):
            validate_atom_indices([0, 3], self.atoms)

    def test_validate_non_integer(self):
        with pytest.raises(ValueError):
            validate_atom_indices([0, 1.5], self.atoms)  # type: ignore


class TestHarmonicConstraintInternals:
    def setup_method(self):
        self.atoms = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])

    def test_bond_reference_calculation(self):
        c = HarmonicBondConstraint([0, 1], self.atoms, 5.0)
        expected = np.linalg.norm(self.atoms.positions[1] - self.atoms.positions[0])
        assert abs(c.reference_value - expected) < 1e-10

    def test_angle_reference_validation(self):
        with pytest.raises(ValueError):
            HarmonicAngleConstraint([0, 1], self.atoms, 2.0)
