from ase import Atoms

import qme


def test_reaction_linear_interpolation_and_lengths():
    r = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    p = Atoms("H2", positions=[[0, 0, 0], [2.00, 0, 0]])

    calc = qme.MockCalculator(backend="mock")
    rxn = qme.Reaction(r, p, calculator=calc)
    path = rxn.interpolate(npoints=5, method="linear")
    assert len(path) == 5
    for g in path:
        assert len(g) == 2
