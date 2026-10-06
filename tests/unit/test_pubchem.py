"""PubChem import. HTTP is mocked; one fallback test embeds a SMILES string."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import famex
from famex.cli import main
from famex.cli.cli_helpers import load_structure_or_smiles
from famex.io.geometry import Geometry
from famex.io.pubchem import fetch_pubchem, parse_pubchem_query

_WATER_SDF = """
water
     test

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    1.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
"""

_NAME_JSON = {"IdentifierList": {"CID": [962, 222]}}
_PROPS_JSON = {
    "PropertyTable": {
        "Properties": [
            {"CID": 962, "Charge": 0, "Title": "Water", "IsomericSMILES": "O"},
        ]
    }
}


class _Response:
    def __init__(self, status: int, text: str = "", payload: dict | None = None) -> None:
        self.status_code = status
        self.text = text
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.headers: dict[str, str] = {}
        self._responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, timeout: float | None = None) -> _Response:
        self.calls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected request {url}")
        return self._responses.pop(0)


def _patch_session(monkeypatch: pytest.MonkeyPatch, responses: list[_Response]) -> _Session:
    session = _Session(responses)
    monkeypatch.setattr("famex.io.pubchem.requests.Session", lambda: session)
    return session


class TestPubChemQuery:
    def test_prefixes(self) -> None:
        assert parse_pubchem_query("aspirin") == ("name", "aspirin")
        assert parse_pubchem_query("  pubchem:Aspirin ") == ("name", "Aspirin")
        assert parse_pubchem_query("cid:2244") == ("cid", "2244")
        assert parse_pubchem_query("pubchem:cid:2244") == ("cid", "2244")
        assert parse_pubchem_query("2244") == ("cid", "2244")
        with pytest.raises(ValueError, match="empty"):
            parse_pubchem_query("pubchem:")


class TestPubChemDownload:
    def test_uses_3d_conformer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_session(
            monkeypatch,
            [
                _Response(200, payload=_NAME_JSON),
                _Response(200, payload=_PROPS_JSON),
                _Response(200, text=_WATER_SDF),
            ],
        )
        geom = fetch_pubchem("water")
        assert list(geom.get_chemical_symbols()) == ["O", "H", "H"]
        assert geom.info["pubchem_cid"] == 962
        assert geom.info["pubchem_title"] == "Water"
        assert geom.info["coordinates"] == "pubchem-3d"
        assert geom.info["smiles"] == "O"
        assert geom.charge == 0
        assert geom.mult == 1

    def test_retries_while_pubchem_is_busy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("famex.io.pubchem.time.sleep", lambda _delay: None)
        session = _patch_session(
            monkeypatch,
            [
                _Response(202, text="waiting"),
                _Response(200, payload={"IdentifierList": {"CID": [962]}}),
                _Response(200, payload=_PROPS_JSON),
                _Response(200, text=_WATER_SDF),
            ],
        )
        geom = fetch_pubchem("water")
        assert geom.info["coordinates"] == "pubchem-3d"
        assert session.calls[0] == session.calls[1]

    def test_gives_up_when_pubchem_stays_busy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("famex.io.pubchem.time.sleep", lambda _delay: None)
        _patch_session(
            monkeypatch,
            [_Response(202, text="waiting") for _ in range(4)],
        )
        with pytest.raises(ValueError, match="still busy"):
            fetch_pubchem("water")

    def test_missing_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_session(monkeypatch, [_Response(404, text="Status: 404 PUGREST.NotFound")])
        with pytest.raises(ValueError, match="no compound"):
            fetch_pubchem("not-a-real-compound")

    def test_embeds_when_3d_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pysmiles")
        _patch_session(
            monkeypatch,
            [
                _Response(200, payload={"IdentifierList": {"CID": [962]}}),
                _Response(200, payload=_PROPS_JSON),
                _Response(404, text="No conformers"),
            ],
        )
        geom = fetch_pubchem("water")
        assert len(geom) == 3
        assert geom.info["coordinates"] == "embedded"
        assert geom.info["pubchem_cid"] == 962
        assert geom.info["embedder"] == "pysmiles"


class TestPubChemAPI:
    def test_helpers_and_cli(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        geom = Geometry(
            atoms=["O", "H", "H"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            charge=0,
            mult=1,
        )
        geom.info["pubchem_cid"] = 962
        geom.info["pubchem_title"] = "Water"
        geom.info["coordinates"] = "pubchem-3d"
        geom.info["source"] = "pubchem"

        monkeypatch.setattr("famex.io.pubchem.fetch_pubchem", lambda *args, **kwargs: geom)

        loaded = Geometry.from_pubchem("water")
        assert loaded.info["pubchem_cid"] == 962
        explorer = famex.Explorer.from_pubchem("water", backend="mock", target="minima")
        assert len(explorer.atoms_list[0]) == 3

        out = tmp_path / "water.xyz"
        result = CliRunner().invoke(main, ["fetch", "water", "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "CID 962" in result.output
        assert out.exists()

        dry = CliRunner().invoke(
            main,
            ["minima", "--strategy", "local", "pubchem:water", "--backend", "mock", "--dry-run"],
        )
        assert dry.exit_code == 0, dry.output

        pytest.importorskip("pysmiles")
        paired = CliRunner().invoke(
            main,
            [
                "minima",
                "--strategy",
                "interpolate",
                "O",
                "--product",
                "cid:962",
                "--backend",
                "mock",
                "--dry-run",
            ],
        )
        assert paired.exit_code == 0, paired.output

        mismatched = CliRunner().invoke(
            main,
            [
                "path",
                "--strategy",
                "neb",
                "CCO",
                "pubchem:water",
                "--backend",
                "mock",
                "--dry-run",
            ],
        )
        assert mismatched.exit_code != 0
        assert "Atom counts do not match" in mismatched.output

    def test_smiles_input_does_not_call_pubchem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pysmiles")

        def fail(*args, **kwargs):
            raise AssertionError("PubChem was called for a SMILES string")

        monkeypatch.setattr("famex.io.pubchem.fetch_pubchem", fail)
        atoms = load_structure_or_smiles("O")
        assert len(atoms) == 3
