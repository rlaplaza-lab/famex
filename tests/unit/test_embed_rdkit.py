"""RDKit ETKDGv3 embedding. Skipped unless the rdkit extra is installed."""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

import famex
from famex.cli import main
from famex.io.geometry import Geometry

pytest.importorskip("rdkit")


def _min_distance(atoms: Geometry) -> float:
    pos = np.asarray(atoms.positions, dtype=float)
    delta = pos[:, None, :] - pos[None, :, :]
    dmat = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dmat, np.inf)
    return float(dmat.min())


class TestRDKitEmbedder:
    def test_water_and_ethanol(self) -> None:
        water = famex.smiles_to_atoms("O", seed=0, embedder="rdkit")
        assert isinstance(water, Geometry)
        assert water.info["embedder"] == "rdkit"
        assert list(water.get_chemical_symbols()) == ["O", "H", "H"]
        oh = float(np.linalg.norm(water.positions[0] - water.positions[1]))
        assert 0.9 < oh < 1.15

        ethanol = famex.smiles_to_atoms("CCO", seed=0, embedder="rdkit")
        assert isinstance(ethanol, Geometry)
        assert len(ethanol) == 9
        assert _min_distance(ethanol) > 0.7

    def test_charge_and_repeatability(self) -> None:
        ammonium = famex.smiles_to_atoms("[NH4+]", seed=0, embedder="rdkit")
        assert isinstance(ammonium, Geometry)
        assert len(ammonium) == 5
        assert ammonium.charge == 1
        assert ammonium.mult == 1

        first = famex.smiles_to_atoms("CCO", seed=2, embedder="rdkit")
        second = famex.smiles_to_atoms("CCO", seed=2, embedder="rdkit")
        assert isinstance(first, Geometry)
        assert isinstance(second, Geometry)
        assert np.allclose(first.positions, second.positions)

    def test_multiple_conformers_and_helpers(self) -> None:
        conformers = famex.smiles_to_atoms("CCO", n_conf=2, seed=0, embedder="rdkit")
        assert isinstance(conformers, list)
        assert len(conformers) == 2
        assert all(len(geom) == 9 for geom in conformers)

        geom = Geometry.from_smiles("O", seed=0, embedder="rdkit")
        assert geom.info["embedder"] == "rdkit"
        explorer = famex.Explorer.from_smiles("O", backend="mock", seed=0, embedder="rdkit")
        assert len(explorer.atoms_list[0]) == 3

    def test_nickel_carbonyl_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid SMILES"):
            famex.smiles_to_atoms("[Ni](C#[O])(C#[O])(C#[O])C#[O]", embedder="rdkit")


class TestRDKitCLI:
    def test_minima_accepts_rdkit_embedder(self) -> None:
        dry = CliRunner().invoke(
            main,
            [
                "minima",
                "--strategy",
                "local",
                "O",
                "--embedder",
                "rdkit",
                "--backend",
                "mock",
                "--dry-run",
            ],
        )
        assert dry.exit_code == 0, dry.output

        bad = CliRunner().invoke(
            main,
            [
                "minima",
                "--strategy",
                "local",
                "O",
                "--embedder",
                "obabel",
                "--backend",
                "mock",
                "--dry-run",
            ],
        )
        assert bad.exit_code != 0
