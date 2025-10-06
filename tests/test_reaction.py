"""
Test reaction pathway functionality.

This module tests the reaction pathway interpolation and analysis
capabilities of QME.
"""

from ase import Atoms

import qme
from tests.test_utils import StandardTestAssertions


class TestReactionPathway:
    """Test reaction pathway interpolation and analysis."""

    def test_linear_interpolation_and_lengths(self):
        """Test linear interpolation between reactant and product."""
        # Use H2O -> H2O (slightly different geometry) for meaningful testing
        reactant = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        product = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

        calc = qme.MockCalculator(backend="mock")
        reaction = qme.Reaction(reactant, product, calculator=calc)
        path = reaction.interpolate(npoints=5, method="linear")

        # Check path length
        assert len(path) == 5

        # Check that all structures have correct number of atoms
        for structure in path:
            assert len(structure) == 3  # H2O has 3 atoms
            # Check that structure is reasonable
            StandardTestAssertions.assert_reasonable_geometry(structure, "mock")

    def test_reaction_energy_calculation(self):
        """Test energy calculation along reaction pathway."""
        reactant = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
        product = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

        calc = qme.MockCalculator(backend="mock")
        reaction = qme.Reaction(reactant, product, calculator=calc)

        # Test energy calculation using the reaction_energy property
        reaction_energy = reaction.reaction_energy

        # Check that reaction energy is reasonable
        if reaction_energy is not None:
            StandardTestAssertions.assert_energy_reasonable(reaction_energy, "mock")
            assert isinstance(reaction_energy, (int, float))

        # Test individual geometry energies
        reactant_energy = reaction.reactant.energy
        product_energy = reaction.product.energy

        # Check that energies are reasonable
        if reactant_energy is not None:
            StandardTestAssertions.assert_energy_reasonable(reactant_energy, "mock")
            assert isinstance(reactant_energy, (int, float))

        if product_energy is not None:
            StandardTestAssertions.assert_energy_reasonable(product_energy, "mock")
            assert isinstance(product_energy, (int, float))
