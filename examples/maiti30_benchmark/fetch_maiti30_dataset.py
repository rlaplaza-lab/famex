#!/usr/bin/env python3
"""Download Maiti-30 geometries from ioChem-BD (BSC) into maiti30_dataset/.

Source: http://dx.doi.org/10.19061/iochem-bd-6-566
Author listing: Maiti, Shoubhik on https://iochem-bd.bsc.es/browse
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from html import unescape
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (compatible; famex-dataset-fetch/1.0)"}
ORIGIN = "https://iochem-bd.bsc.es"
BROWSE = f"{ORIGIN}/browse"
OUT = Path(__file__).resolve().parent / "maiti30_dataset"


def fetch(url: str, retries: int = 4) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Failed {url}: {last}")


def author_pages() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for offset in range(0, 1200, 100):
        url = f"{BROWSE}/browse?type=author&value=Maiti%2C+Shoubhik&rpp=100&offset={offset}"
        text = fetch(url).decode("utf-8", "replace")
        found = re.findall(r'href="/browse/handle/100/(\d+)">([^<]+)</a>', text)
        if not found:
            break
        for h, t in found:
            items.append((h, unescape(t)))
    return items


def parse_item(handle: str) -> dict:
    html = fetch(f"{BROWSE}/handle/100/{handle}").decode("utf-8", "replace")
    m = re.search(r"cml2xyzUrl = '([^']+)'", html)
    if not m:
        raise RuntimeError(f"No cml2xyz for {handle}")
    cml2xyz = m.group(1).replace(":443", "")
    inps = re.findall(r'href="(/browse/bitstream/100/\d+/\d+/[^"]+\.inp)"', html)
    if not inps:
        raise RuntimeError(f"No inp for {handle}")
    inp_text = fetch(ORIGIN + inps[0]).decode("utf-8", "replace")
    cm = re.search(r"\*xyzfile\s+(-?\d+)\s+(\d+)", inp_text)
    if not cm:
        cm = re.search(r"\*\s*xyz\s+(-?\d+)\s+(\d+)", inp_text)
    if not cm:
        raise RuntimeError(f"No charge/mult in {inps[0]}")
    return {
        "handle": handle,
        "cml2xyz": cml2xyz,
        "charge": int(cm.group(1)),
        "spin": int(cm.group(2)),
        "inp_name": Path(unescape(inps[0])).name,
    }


def write_xyz(path: Path, xyz_bytes: bytes, charge: int, spin: int) -> None:
    lines = xyz_bytes.decode("utf-8", "replace").strip().splitlines()
    if len(lines) >= 2:
        lines[1] = f"charge={charge} multiplicity={spin}"
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = author_pages()
    rxn_map: dict[int, dict[str, str]] = {}
    for handle, title in items:
        if "/main_benchmark/dataset_preparation/minima_from_irc" in title:
            m = re.search(r"reaction-(\d+)-minim_([FB])\b", title)
            if not m:
                continue
            n, side = int(m.group(1)), m.group(2)
            role = "reactant" if side == "F" else "product"
            rxn_map.setdefault(n, {})[role] = handle
        elif "/main_benchmark/dataset_preparation/ts_optimisation" in title:
            m = re.search(r"reaction-(\d+)-ts-opt\b", title)
            if not m:
                continue
            rxn_map.setdefault(int(m.group(1)), {})["ts"] = handle

    missing = [
        (n, sorted({"reactant", "product", "ts"} - set(rxn_map.get(n, {}))))
        for n in range(1, 31)
        if set(rxn_map.get(n, {})) != {"reactant", "product", "ts"}
    ]
    if missing:
        raise SystemExit(f"Missing handles: {missing}")

    meta: dict[str, dict] = {}
    for n in range(1, 31):
        rid = f"reaction_{n:03d}"
        print(f"Downloading {rid}...")
        entry: dict = {}
        for role, handle in sorted(rxn_map[n].items()):
            info = parse_item(handle)
            xyz = fetch(info["cml2xyz"])
            write_xyz(OUT / f"{rid}_{role}.xyz", xyz, info["charge"], info["spin"])
            entry[role] = info
            time.sleep(0.1)
        meta[rid] = {
            "charge": entry["reactant"]["charge"],
            "spin": entry["reactant"]["spin"],
            "roles": {r: {k: v for k, v in entry[r].items() if k != "inp_text"} for r in entry},
        }
    (OUT / "charges.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(list(OUT.glob('reaction_*.xyz')))} xyz files to {OUT}")


if __name__ == "__main__":
    main()
