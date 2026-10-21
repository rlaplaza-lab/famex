"""Tests for RDKit-free SMILES distance-geometry embedding."""

from __future__ import annotations

import math

import numpy as np
import pytest
from click.testing import CliRunner
from scipy.linalg import orthogonal_procrustes

import famex
from famex.backends.availability import is_backend_available
from famex.cli import main
from famex.cli.cli_helpers import load_structure_or_smiles, write_atoms
from famex.embed.bounds import _smooth, torsion_distance
from famex.embed.cleanup import _UFFModel
from famex.embed.dgeom import _classical_mds, embed_coordinates
from famex.embed.parse import parse_smiles
from famex.embed.typing import AtomParams, angle_force_constant, assign_parameters, bond_length
from famex.io.geometry import Geometry

pytest.importorskip("pysmiles")


def _symbols(atoms: Geometry) -> list[str]:
    return list(atoms.get_chemical_symbols())


def _dist(atoms: Geometry, i: int, j: int) -> float:
    return float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))


def _min_distance(atoms: Geometry) -> float:
    pos = np.asarray(atoms.positions, dtype=float)
    delta = pos[:, None, :] - pos[None, :, :]
    dmat = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dmat, np.inf)
    return float(dmat.min())


def _closed_form_bond(r_i: float, chi_i: float, r_j: float, chi_j: float, order: float) -> float:
    lam = 0.1332
    rij = r_i + r_j
    if abs(order - 1.0) > 1e-8:
        rij -= lam * (r_i + r_j) * math.log(order)
    denom = chi_i * r_i + chi_j * r_j
    ren = r_i * r_j * (math.sqrt(chi_i) - math.sqrt(chi_j)) ** 2 / denom
    return rij - ren


def _closed_form_angle_k(
    z_i: float, z_k: float, r_ij: float, r_jk: float, theta_deg: float
) -> float:
    k_scale = 664.12
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    rik2 = r_ij * r_ij + r_jk * r_jk - 2.0 * r_ij * r_jk * cos_t
    rik = math.sqrt(rik2)
    return (
        k_scale
        * z_i
        * z_k
        / rik**5
        * r_ij
        * r_jk
        * (3.0 * r_ij * r_jk * (1.0 - cos_t * cos_t) - rik2 * cos_t)
    )


def _parsed_volume(geom: Geometry, center: int, neighbors: tuple[int, int, int, int]) -> float:
    pos = geom.positions
    a, b, c = neighbors[:3]
    v1 = pos[a] - pos[center]
    v2 = pos[b] - pos[center]
    v3 = pos[c] - pos[center]
    return float(np.dot(v1, np.cross(v2, v3)))


def _assert_numeric_gradient(smiles: str) -> None:
    graph = parse_smiles(smiles)
    params = assign_parameters(graph)
    coords = embed_coordinates(graph, params, np.random.default_rng(0), deterministic=True)
    model = _UFFModel(graph, params)
    x = np.asarray(coords, dtype=float).reshape(-1).copy()
    _energy, grad = model.energy_grad(x)
    eps = 1e-6
    numeric = np.zeros_like(grad)
    for index in range(len(x)):
        x[index] += eps
        energy_plus, _g = model.energy_grad(x)
        x[index] -= 2.0 * eps
        energy_minus, _g = model.energy_grad(x)
        x[index] += eps
        numeric[index] = (energy_plus - energy_minus) / (2.0 * eps)
    assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-4)


class TestUFFParameters:
    def test_single_bond_lengths(self) -> None:
        carbon = AtomParams("C_3", 6, 0.757, 109.471, 3.851, 0.105, 1.912, 5.343, 4)
        alkene = AtomParams("C_2", 6, 0.732, 120.0, 3.851, 0.105, 1.912, 5.343, 3)
        hydrogen = AtomParams("H_", 1, 0.354, 180.0, 2.886, 0.044, 0.712, 4.528, 1)
        oxygen = AtomParams("O_3", 8, 0.658, 104.51, 3.500, 0.060, 2.300, 8.741, 2)
        assert bond_length(carbon, carbon, 1.0) == pytest.approx(
            _closed_form_bond(0.757, 5.343, 0.757, 5.343, 1.0)
        )
        assert bond_length(alkene, alkene, 2.0) == pytest.approx(
            _closed_form_bond(0.732, 5.343, 0.732, 5.343, 2.0)
        )
        assert bond_length(oxygen, hydrogen, 1.0) == pytest.approx(
            _closed_form_bond(0.658, 8.741, 0.354, 4.528, 1.0)
        )

    def test_linear_angle_constant(self) -> None:
        linear = angle_force_constant(1.912, 1.912, 1.2, 1.2, 180.0)
        near = angle_force_constant(1.912, 1.912, 1.2, 1.2, 179.0)
        assert linear == pytest.approx(_closed_form_angle_k(1.912, 1.912, 1.2, 1.2, 180.0))
        assert near == pytest.approx(_closed_form_angle_k(1.912, 1.912, 1.2, 1.2, 179.0))
        assert abs(linear - near) < 1.0

    def test_torsion_distances(self) -> None:
        cis = torsion_distance(1.5, 1.34, 1.5, 120.0, 120.0, 0.0)
        trans = torsion_distance(1.5, 1.34, 1.5, 120.0, 120.0, math.pi)
        assert cis == pytest.approx(2.84)
        assert trans == pytest.approx(math.sqrt((1.5 * math.sqrt(3.0)) ** 2 + 2.84**2))

    def test_classical_mds_recovers_points(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.4, 1.2, 0.0],
                [0.2, 0.3, 1.1],
            ]
        )
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        embedded = _classical_mds(dist)
        left = embedded - embedded.mean(axis=0)
        right = points - points.mean(axis=0)
        rotation, _scale = orthogonal_procrustes(left, right)
        assert np.allclose(left @ rotation, right, atol=1e-8)

    def test_smoothing_raises_triangle_violation(self) -> None:
        lower = np.zeros((3, 3))
        upper = np.full((3, 3), 10.0)
        np.fill_diagonal(upper, 0.0)
        lower[0, 1] = lower[1, 0] = 4.0
        upper[0, 1] = upper[1, 0] = 4.0
        upper[1, 2] = upper[2, 1] = 1.0
        _smooth(lower, upper)
        assert lower[0, 2] == pytest.approx(3.0)
        assert lower[2, 0] == pytest.approx(3.0)


class TestEmbedMolecules:
    def test_water_geometry(self) -> None:
        geom = famex.smiles_to_atoms("O", seed=0)
        assert isinstance(geom, Geometry)
        assert _symbols(geom) == ["O", "H", "H"]
        assert geom.charge == 0
        assert geom.mult == 1
        hydrogens = [i for i, symbol in enumerate(_symbols(geom)) if symbol == "H"]
        for hydrogen in hydrogens:
            assert 0.90 < _dist(geom, 0, hydrogen) < 1.12
        angle = float(geom.get_angle(hydrogens[0], 0, hydrogens[1]))
        assert 95.0 < angle < 115.0
        assert _min_distance(geom) > 0.7

    def test_ethanol_and_benzene(self) -> None:
        ethanol = famex.smiles_to_atoms("CCO", seed=1)
        assert isinstance(ethanol, Geometry)
        symbols = _symbols(ethanol)
        assert symbols.count("C") == 2
        assert symbols.count("O") == 1
        assert symbols.count("H") == 6
        carbons = [i for i, symbol in enumerate(symbols) if symbol == "C"]
        oxygen = symbols.index("O")
        assert 1.40 < _dist(ethanol, carbons[0], carbons[1]) < 1.70
        carbon_oxygen = min(_dist(ethanol, carbon, oxygen) for carbon in carbons)
        assert 1.25 < carbon_oxygen < 1.60
        assert _min_distance(ethanol) > 0.6

        benzene = famex.smiles_to_atoms("c1ccccc1", seed=0)
        assert isinstance(benzene, Geometry)
        b_symbols = _symbols(benzene)
        assert b_symbols.count("C") == 6
        assert b_symbols.count("H") == 6
        b_carbons = [i for i, symbol in enumerate(b_symbols) if symbol == "C"]
        cc = [
            _dist(benzene, i, j)
            for left, i in enumerate(b_carbons)
            for j in b_carbons[left + 1 :]
            if _dist(benzene, i, j) < 1.7
        ]
        assert len(cc) == 6
        assert all(1.28 < distance < 1.55 for distance in cc)
        coords = benzene.positions[b_carbons]
        centered = coords - coords.mean(axis=0)
        _u, _s, vh = np.linalg.svd(centered)
        normal = vh[-1]
        rms = float(np.sqrt(np.mean(np.square(centered @ normal))))
        assert rms < 0.30
        assert _min_distance(benzene) > 0.6

    def test_salt_and_noble_gas(self) -> None:
        salt = famex.smiles_to_atoms("[Na+].[Cl-]", seed=0)
        assert isinstance(salt, Geometry)
        assert salt.charge == 0
        assert len(salt) == 2
        separation = _dist(salt, 0, 1)
        assert 1.8 < separation < 6.0

        xenon = famex.smiles_to_atoms("[Xe]", seed=0)
        assert isinstance(xenon, Geometry)
        assert _symbols(xenon) == ["Xe"]

    def test_nickel_carbonyl(self) -> None:
        geom = famex.smiles_to_atoms("[Ni](C#[O])(C#[O])(C#[O])C#[O]", seed=0)
        assert isinstance(geom, Geometry)
        symbols = _symbols(geom)
        assert symbols.count("Ni") == 1
        assert symbols.count("C") == 4
        assert symbols.count("O") == 4
        assert "H" not in symbols
        nickel = symbols.index("Ni")
        carbons = [i for i, symbol in enumerate(symbols) if symbol == "C"]
        oxygens = [i for i, symbol in enumerate(symbols) if symbol == "O"]
        for carbon in carbons:
            assert 1.45 < _dist(geom, nickel, carbon) < 2.40
            oxygen = min(oxygens, key=lambda index: _dist(geom, carbon, index))
            assert 1.00 < _dist(geom, carbon, oxygen) < 1.35
            assert float(geom.get_angle(nickel, carbon, oxygen)) > 150.0
        assert _min_distance(geom) > 0.6

    def test_charge_spin_and_multiple_conformers(self) -> None:
        ammonium = famex.smiles_to_atoms("[NH4+]", seed=0)
        assert isinstance(ammonium, Geometry)
        assert ammonium.charge == 1
        assert ammonium.info["spin"] == 1
        assert len(ammonium) == 5

        methyl = famex.smiles_to_atoms("[CH3]", seed=0)
        assert isinstance(methyl, Geometry)
        assert methyl.mult == 2
        assert len(methyl) == 4

        conformers = famex.smiles_to_atoms("CCO", n_conf=2, seed=0)
        assert isinstance(conformers, list)
        assert len(conformers) == 2

    def test_tetrahedral_and_double_bond_stereo(self) -> None:
        clockwise = parse_smiles("[C@H](Br)(Cl)F")
        counter = parse_smiles("[C@@H](Br)(Cl)F")
        assert len(clockwise.chiral) == 1
        assert len(counter.chiral) == 1

        embedded_at = famex.smiles_to_atoms("[C@H](Br)(Cl)F", seed=1)
        embedded_at_at = famex.smiles_to_atoms("[C@@H](Br)(Cl)F", seed=1)
        assert isinstance(embedded_at, Geometry)
        assert isinstance(embedded_at_at, Geometry)

        def volume(geom: Geometry) -> float:
            symbols = _symbols(geom)
            center = symbols.index("C")
            bromine = symbols.index("Br")
            chlorine = symbols.index("Cl")
            fluorine = symbols.index("F")
            pos = geom.positions
            v1 = pos[bromine] - pos[center]
            v2 = pos[chlorine] - pos[center]
            v3 = pos[fluorine] - pos[center]
            return float(np.dot(v1, np.cross(v2, v3)))

        assert volume(embedded_at) * volume(embedded_at_at) < 0.0
        assert (
            _parsed_volume(embedded_at, clockwise.chiral[0].center, clockwise.chiral[0].neighbors)
            > 0.0
        )
        assert (
            _parsed_volume(embedded_at_at, counter.chiral[0].center, counter.chiral[0].neighbors)
            > 0.0
        )

        trans = parse_smiles(r"F/C=C/F")
        cis = parse_smiles(r"F/C=C\F")
        assert trans.ez and trans.ez[0].kind == "trans"
        assert cis.ez and cis.ez[0].kind == "cis"
        trans_geom = famex.smiles_to_atoms(r"F/C=C/F", seed=0)
        cis_geom = famex.smiles_to_atoms(r"F/C=C\F", seed=0)
        assert isinstance(trans_geom, Geometry)
        assert isinstance(cis_geom, Geometry)

        def fluorine_distance(geom: Geometry) -> float:
            fluorines = [i for i, symbol in enumerate(_symbols(geom)) if symbol == "F"]
            return _dist(geom, fluorines[0], fluorines[1])

        assert fluorine_distance(trans_geom) > fluorine_distance(cis_geom) + 0.4

    def test_three_stereocenters(self) -> None:
        smiles = "C[C@H](F)[C@H](F)[C@H](F)C"
        graph = parse_smiles(smiles)
        for seed in range(4):
            geom = famex.smiles_to_atoms(smiles, seed=seed)
            assert isinstance(geom, Geometry)
            symbols = _symbols(geom)
            for spec in graph.chiral:
                assert _parsed_volume(geom, spec.center, spec.neighbors) > 0.2
                hydrogens = [nb for nb in spec.neighbors if symbols[nb] == "H"]
                assert hydrogens
                for hydrogen in hydrogens:
                    assert 0.95 < _dist(geom, spec.center, hydrogen) < 1.25

    def test_diastereomers_keep_opposite_configuration(self) -> None:
        same = "C[C@H](F)[C@H](F)C"
        other = "C[C@H](F)[C@@H](F)C"
        spec = parse_smiles(same).chiral[1]
        same_geom = famex.smiles_to_atoms(same, seed=0)
        other_geom = famex.smiles_to_atoms(other, seed=0)
        assert isinstance(same_geom, Geometry)
        assert isinstance(other_geom, Geometry)
        assert _parsed_volume(same_geom, spec.center, spec.neighbors) > 0.2
        assert _parsed_volume(other_geom, spec.center, spec.neighbors) < -0.2

    def test_invalid_smiles_and_repeatability(self) -> None:
        with pytest.raises(ValueError, match="Invalid SMILES"):
            famex.smiles_to_atoms("[C")
        first = famex.smiles_to_atoms("CCO", seed=3)
        second = famex.smiles_to_atoms("CCO", seed=3)
        assert isinstance(first, Geometry)
        assert isinstance(second, Geometry)
        assert np.allclose(first.positions, second.positions)

    @pytest.mark.parametrize("smiles", ["O", "CCCC", "O=C=O", "[S](F)(F)(F)(F)(F)F"])
    def test_cleanup_gradient(self, smiles: str) -> None:
        _assert_numeric_gradient(smiles)

    def test_geometry_and_explorer_helpers(self) -> None:
        geom = Geometry.from_smiles("O", seed=0)
        assert len(geom) == 3
        assert geom.info["embedder"] == "pysmiles"
        explorer = famex.Explorer.from_smiles("O", backend="mock", seed=0)
        assert len(explorer.atoms_list[0]) == 3
        with pytest.raises(ValueError, match="embedder"):
            famex.smiles_to_atoms("O", embedder="obabel")


class TestEmbedCLI:
    def test_smiles_api_writes_xyz(self, tmp_path) -> None:
        out = tmp_path / "ethanol.xyz"
        geom = famex.smiles_to_atoms("CCO", seed=0)
        write_atoms(geom, str(out))
        assert out.exists()
        loaded = load_structure_or_smiles(str(out))
        assert len(loaded) == 9

    def test_minima_accepts_smiles_and_rejects_missing_files(self) -> None:
        runner = CliRunner()
        dry = runner.invoke(
            main,
            ["minima", "--strategy", "local", "O", "--backend", "mock", "--dry-run"],
        )
        assert dry.exit_code == 0, dry.output
        assert "Target: minima" in dry.output

        interpolated = runner.invoke(
            main,
            [
                "minima",
                "--strategy",
                "interpolate",
                "CCO",
                "--product",
                "COC",
                "--backend",
                "mock",
                "--dry-run",
            ],
        )
        assert interpolated.exit_code == 0, interpolated.output

        missing = runner.invoke(
            main,
            ["minima", "no_such_molecule.xyz", "--backend", "mock", "--dry-run"],
        )
        assert missing.exit_code != 0
        assert "File not found" in missing.output

        ts_dry = runner.invoke(
            main,
            ["ts", "--strategy", "local", "O", "--backend", "mock", "--dry-run"],
        )
        assert ts_dry.exit_code == 0, ts_dry.output

        mismatch = runner.invoke(
            main,
            [
                "ts",
                "--strategy",
                "interpolate",
                "O",
                "--product",
                "CCO",
                "--backend",
                "mock",
                "--dry-run",
            ],
        )
        assert mismatch.exit_code != 0
        assert "Atom counts do not match" in mismatch.output

        path = runner.invoke(
            main,
            [
                "path",
                "--strategy",
                "interpolate",
                "CCO",
                "COC",
                "--backend",
                "mock",
                "--dry-run",
            ],
        )
        assert path.exit_code == 0, path.output

        no_embed = runner.invoke(main, ["embed", "CCO"])
        assert no_embed.exit_code != 0
        no_fetch = runner.invoke(main, ["fetch", "water"])
        assert no_fetch.exit_code != 0

    @pytest.mark.slow
    def test_minima_smiles_when_mlip_present(self, tmp_path) -> None:
        if is_backend_available("uma"):
            backend = "uma"
        elif is_backend_available("pet"):
            backend = "pet"
        else:
            pytest.skip("neither UMA nor PET is installed")

        out = tmp_path / "water.xyz"
        result = CliRunner().invoke(
            main,
            [
                "minima",
                "--strategy",
                "local",
                "O",
                "--output",
                str(out),
                "--backend",
                backend,
                "--device",
                "cpu",
                "--fmax",
                "0.05",
                "--steps",
                "100",
            ],
        )
        assert result.exit_code == 0, result.output
        loaded = load_structure_or_smiles(str(out))
        assert len(loaded) == 3
        oh = float(np.linalg.norm(loaded.positions[0] - loaded.positions[1]))
        assert 0.85 < oh < 1.15

    def test_existing_file_is_not_parsed_as_smiles(self, tmp_path) -> None:
        xyz = tmp_path / "water.xyz"
        xyz.write_text("3\ncharge=0 spin=1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n")
        atoms = load_structure_or_smiles(str(xyz))
        assert len(atoms) == 3
        assert atoms.get_chemical_symbols()[0] == "O"
