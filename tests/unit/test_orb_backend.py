"""
Tests for Orb backend integration.
"""

import pytest
from ase import Atoms
from ase.build import molecule

from qme.potentials.orb_potential import get_orb_calculator, OrbPotential
from qme.calculator_registry import calculator_registry


def _orb_backend_available():
    """Check if Orb backend is available."""
    return calculator_registry.is_backend_available("orb")


class TestOrbBackend:
    """Test Orb ML backend."""

    def test_calculator_creation(self):
        """Test that calculator can be created."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        calc = get_orb_calculator()
        assert calc is not None
        assert isinstance(calc, OrbPotential)

    def test_calculator_with_parameters(self):
        """Test calculator creation with custom parameters."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        calc = get_orb_calculator(
            model_name="orb-v3-conservative-omol",
            device="cpu",
            charge=1,
            spin=2
        )
        assert calc.model_name == "orb-v3-conservative-omol"
        assert calc.charge == 1
        assert calc.spin == 2

    def test_energy_calculation(self):
        """Test energy calculation."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        atoms = molecule("H2O")
        calc = get_orb_calculator()
        atoms.calc = calc

        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert not isinstance(energy, complex)

    def test_force_calculation(self):
        """Test force calculation."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        atoms = molecule("H2O")
        calc = get_orb_calculator()
        atoms.calc = calc

        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)
        assert forces.dtype.kind == 'f'  # float type

    def test_charge_and_spin_parameters(self):
        """Test that charge and spin parameters work correctly."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        atoms = molecule("H2O")
        
        # Test neutral molecule
        calc_neutral = get_orb_calculator(charge=0, spin=1)
        atoms.calc = calc_neutral
        energy_neutral = atoms.get_potential_energy()
        
        # Test charged molecule
        calc_charged = get_orb_calculator(charge=1, spin=2)
        atoms.calc = calc_charged
        energy_charged = atoms.get_potential_energy()
        
        # Energies should be different for different charge states
        assert energy_neutral != energy_charged

    def test_model_variants(self):
        """Test different Orb model variants."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        atoms = molecule("H2O")
        
        # Test default model
        calc_omol = get_orb_calculator(model_name="orb-v3-conservative-omol")
        atoms.calc = calc_omol
        energy_omol = atoms.get_potential_energy()
        
        # Test materials model (should still work for molecules)
        calc_omat = get_orb_calculator(model_name="orb-v3-conservative-inf-omat")
        atoms.calc = calc_omat
        energy_omat = atoms.get_potential_energy()
        
        # Both should give reasonable energies
        assert isinstance(energy_omol, float)
        assert isinstance(energy_omat, float)

    def test_set_charge_and_spin(self):
        """Test setting charge and spin after creation."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        calc = get_orb_calculator()
        
        calc.set_charge(2)
        calc.set_spin(3)
        
        assert calc.charge == 2
        assert calc.spin == 3

    def test_integration_with_explorer(self):
        """Test that Orb backend works with QME Explorer."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        from qme.core.explorer import Explorer
        
        atoms = molecule("H2O")
        
        explorer = Explorer(
            atoms=atoms,
            backend="orb",
            default_charge=0,
            default_spin=1,
            device="cpu"
        )
        
        # Manually attach calculator to test direct energy calculation
        explorer._create_and_attach_calculator(explorer.atoms_list[0])
        
        # Test that we can get energy
        energy = explorer.atoms_list[0].get_potential_energy()
        assert isinstance(energy, float)

    def test_calculator_registry_integration(self):
        """Test that Orb backend is properly registered."""
        # Should be in available backends list (if dependencies available)
        available_backends = calculator_registry.get_available_backends()
        if _orb_backend_available():
            assert "orb" in available_backends
        else:
            # If not available, should still be in the registry
            assert "orb" in calculator_registry._lazy_registry

    def test_backend_name(self):
        """Test that backend name is correct."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        calc = get_orb_calculator()
        assert calc._get_backend_name() == "orb"

    def test_implemented_properties(self):
        """Test that correct properties are implemented."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        calc = get_orb_calculator()
        assert "energy" in calc.implemented_properties
        assert "forces" in calc.implemented_properties

    @pytest.mark.skipif(
        not _orb_backend_available(),
        reason="Orb backend not available"
    )
    def test_optimization_workflow(self):
        """Test basic optimization workflow."""
        from qme.core.explorer import Explorer
        
        atoms = molecule("H2O")
        
        # Create explorer with Orb backend
        explorer = Explorer(
            atoms=atoms,
            backend="orb",
            default_charge=0,
            default_spin=1,
            device="cpu"
        )
        
        # Run a short optimization
        result = explorer.run(mode="minima", steps=5)
        
        # Result is a list containing a dictionary
        assert isinstance(result, list)
        assert len(result) == 1
        result_dict = result[0]
        
        assert "optimized_atoms" in result_dict
        assert "steps_taken" in result_dict
        assert "converged" in result_dict
        assert isinstance(result_dict["optimized_atoms"], type(atoms))

    def test_error_handling_missing_dependencies(self):
        """Test error handling when orb-models is not available."""
        # This test is tricky since we can't easily mock the dependency
        # But we can test that the error message is informative
        try:
            calc = get_orb_calculator()
            # If we get here, orb-models is available
            assert calc is not None
        except ImportError as e:
            # Should have informative error message
            assert "orb-models" in str(e) or "orb" in str(e)

    def test_model_name_aliases(self):
        """Test that model name aliases work."""
        if not _orb_backend_available():
            pytest.skip("Orb backend not available")
        
        # Test aliases
        aliases = [
            "orb-v3-omol",
            "orb-v3-omat", 
            "omol",
            "omat"
        ]
        
        atoms = molecule("H2O")
        
        for alias in aliases:
            calc = get_orb_calculator(model_name=alias)
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            assert isinstance(energy, float)
