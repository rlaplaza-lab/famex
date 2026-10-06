"""Download a starting geometry from PubChem.

PubChem PUG REST is the only database wired up today. A computed 3D conformer
is used when PubChem has one. Otherwise the deposited SMILES is embedded with
the selected SMILES embedder.
"""

from __future__ import annotations

import time
from io import StringIO
from typing import Any, Literal
from urllib.parse import quote

import requests
from ase.io import read as ase_read

from famex.io.geometry import Geometry
from famex.utils.logging import get_famex_logger

logger = get_famex_logger(__name__)

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_USER_AGENT = "famex (https://github.com/rlaplaza-lab/famex)"
_TIMEOUT_S = 60
_RETRY_DELAYS_S = (1.0, 2.0, 4.0)
SOURCES = ("pubchem",)
QueryKind = Literal["cid", "name"]


def parse_pubchem_query(query: str) -> tuple[QueryKind, str]:
    """Split a PubChem name or CID, including a ``pubchem:`` prefix."""
    text = query.strip()
    if text.lower().startswith("pubchem:"):
        text = text.split(":", 1)[1].strip()
    if not text:
        raise ValueError("PubChem query is empty")
    if text.lower().startswith("cid:"):
        text = text.split(":", 1)[1].strip()
    if text.isdigit():
        return "cid", text
    return "name", text


def fetch_structure(
    query: str,
    *,
    source: str = "pubchem",
    embedder: str = "pysmiles",
) -> Geometry:
    """Download one 3D geometry from a structure database.

    ``source`` is ``pubchem`` for now. Other names raise ``ValueError``.
    """
    if source == "pubchem":
        return fetch_pubchem(query, embedder=embedder)
    available = ", ".join(SOURCES)
    raise ValueError(f"Unknown structure database {source!r}. Available: {available}")


def fetch_pubchem(query: str, *, embedder: str = "pysmiles") -> Geometry:
    """Download a PubChem compound by name or CID.

    Parameters
    ----------
    query : str
        Compound name, numeric CID, or a ``pubchem:`` / ``cid:`` prefixed form.
    embedder : {'pysmiles', 'rdkit'}, default 'pysmiles'
        Used only when PubChem has no computed 3D conformer.

    Returns
    -------
    Geometry
        PubChem 3D coordinates when available. ``info['coordinates']`` is
        ``pubchem-3d`` or ``embedded``.
    """
    kind, value = parse_pubchem_query(query)
    session = requests.Session()
    session.headers["User-Agent"] = _USER_AGENT
    try:
        cid = _resolve_cid(session, kind, value)
        props = _properties(session, cid)
        sdf = _conformer_sdf(session, cid)
    except requests.RequestException as exc:
        raise ValueError(f"PubChem request failed for {query!r}: {exc}") from exc

    title = str(props.get("Title") or "")
    smiles = _smiles_from_properties(props)
    charge = int(props.get("Charge") or 0)
    if sdf is not None:
        geom = _geometry_from_sdf(sdf, cid=cid, charge=charge)
        _stamp(geom, cid=cid, title=title, smiles=smiles, coordinates="pubchem-3d")
        return geom

    if not smiles:
        raise ValueError(f"PubChem CID {cid} has no 3D conformer and no SMILES")
    logger.warning(
        "PubChem CID %s has no 3D conformer; embedding the deposited SMILES with %s",
        cid,
        embedder,
    )
    # Optional embed extra, needed only for compounds without a 3D record.
    from famex.embed.api import smiles_to_atoms

    embedded = smiles_to_atoms(smiles, n_conf=1, seed=0, embedder=embedder)
    if isinstance(embedded, list):
        if not embedded:
            raise ValueError(f"SMILES from PubChem CID {cid} produced no conformers")
        geom = embedded[0]
    else:
        geom = embedded
    _stamp(geom, cid=cid, title=title, smiles=smiles, coordinates="embedded")
    return geom


def _resolve_cid(session: requests.Session, kind: QueryKind, value: str) -> int:
    if kind == "cid":
        return int(value)
    url = f"{_BASE}/compound/name/{quote(value, safe='')}/cids/JSON"
    response = _get(session, url)
    if response.status_code == 404:
        raise ValueError(f"PubChem has no compound named {value!r}")
    _raise_for_status(response, value)
    payload = _json_object(response)
    cids = payload.get("IdentifierList", {}).get("CID", [])
    if not isinstance(cids, list) or not cids:
        raise ValueError(f"PubChem has no compound named {value!r}")
    if len(cids) > 1:
        logger.info(
            "PubChem name %r matched %s compounds; using CID %s",
            value,
            len(cids),
            cids[0],
        )
    return int(cids[0])


def _properties(session: requests.Session, cid: int) -> dict[str, Any]:
    url = f"{_BASE}/compound/cid/{cid}/property/Title,Charge,IsomericSMILES,CanonicalSMILES/JSON"
    response = _get(session, url)
    if response.status_code == 404:
        raise ValueError(f"PubChem has no compound with CID {cid}")
    _raise_for_status(response, str(cid))
    payload = _json_object(response)
    rows = payload.get("PropertyTable", {}).get("Properties", [])
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ValueError(f"PubChem returned no properties for CID {cid}")
    row: dict[str, Any] = rows[0]
    return row


def _conformer_sdf(session: requests.Session, cid: int) -> str | None:
    url = f"{_BASE}/compound/cid/{cid}/SDF?record_type=3d"
    response = _get(session, url)
    if response.status_code == 404:
        return None
    _raise_for_status(response, str(cid))
    text = response.text.strip()
    if not text:
        return None
    return text


def _smiles_from_properties(props: dict[str, Any]) -> str:
    for key in ("IsomericSMILES", "SMILES", "CanonicalSMILES", "ConnectivitySMILES"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _geometry_from_sdf(sdf: str, *, cid: int, charge: int) -> Geometry:
    try:
        loaded = ase_read(StringIO(sdf), format="sdf")
    except Exception as exc:
        raise ValueError(f"Could not read the PubChem 3D record for CID {cid}") from exc
    atoms = loaded[-1] if isinstance(loaded, list) else loaded
    if atoms is None or len(atoms) == 0:
        raise ValueError(f"PubChem 3D record for CID {cid} is empty")
    numbers = [int(number) for number in atoms.get_atomic_numbers()]
    mult = 2 if (sum(numbers) - charge) % 2 else 1
    return Geometry(ase_atoms=atoms, charge=charge, mult=mult)


def _stamp(geom: Geometry, *, cid: int, title: str, smiles: str, coordinates: str) -> None:
    geom.info["source"] = "pubchem"
    geom.info["pubchem_cid"] = int(cid)
    geom.info["pubchem_title"] = title
    geom.info["coordinates"] = coordinates
    geom.info["charge"] = int(geom.charge)
    geom.info["spin"] = int(geom.mult)
    if smiles:
        geom.info["smiles"] = smiles


def _get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=_TIMEOUT_S)
    for delay in _RETRY_DELAYS_S:
        if response.status_code != 202:
            break
        time.sleep(delay)
        response = session.get(url, timeout=_TIMEOUT_S)
    return response


def _json_object(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            f"PubChem returned a non-JSON response (HTTP {response.status_code})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("PubChem returned an unexpected response")
    return payload


def _raise_for_status(response: requests.Response, query: str) -> None:
    if response.status_code < 400:
        return
    snippet = " ".join(response.text.split())[:240]
    raise ValueError(
        f"PubChem lookup failed for {query!r} (HTTP {response.status_code}): {snippet}"
    )
