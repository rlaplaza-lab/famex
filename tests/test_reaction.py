from ase import Atoms

import qme


def test_reaction_linear_interpolation_and_lengths():
    # Use H2O -> H2O (slightly different geometry) for more meaningful testing
    r = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
    p = Atoms("H2O", positions=[[0, 0, 0], [1.0, 0, 0], [-0.3, 0.9, 0]])

    calc = qme.MockCalculator(backend="mock")
    rxn = qme.Reaction(r, p, calculator=calc)
    path = rxn.interpolate(npoints=5, method="linear")
    assert len(path) == 5
    for g in path:
        assert len(g) == 3  # H2O has 3 atoms
