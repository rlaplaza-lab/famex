"""Integration coverage ported from the former example demos.

Runs against UMA with the bundled ACAB geometries. Marked slow; skipped when
UMA is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from click.testing import CliRunner

import famex
from famex.analysis import ThermodynamicProperties
from famex.backends.availability import is_backend_available
from famex.cli import main

EXAMPLE_FILES = Path(__file__).resolve().parents[2] / "examples" / "example_files"
REACTANT = EXAMPLE_FILES / "A_C_A_B_A_C_reactant.xyz"
PRODUCT = EXAMPLE_FILES / "A_C_A_B_A_C_product.xyz"
TS = EXAMPLE_FILES / "A_C_A_B_A_C_ts.xyz"
GSM_REACTANT = EXAMPLE_FILES / "zimmermann_002_reactant.xyz"
GSM_PRODUCT = EXAMPLE_FILES / "zimmermann_002_product.xyz"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available"),
]


def _assert_frequency_json(json_path: Path, *, expect_ts: bool) -> None:
    assert json_path.exists(), f"missing frequency JSON: {json_path}"
    data = json.loads(json_path.read_text())
    assert "frequency_analysis" in data
    freq = data["frequency_analysis"]
    assert isinstance(freq, dict)
    for key in ("frequencies", "zero_point_energy", "thermodynamic_properties"):
        assert key in freq
    frequencies = np.asarray(freq["frequencies"], dtype=float)
    assert frequencies.size > 0
    assert not np.any(np.isnan(frequencies))
    assert not np.all(np.abs(frequencies) < 1e-6)
    assert np.all(np.abs(frequencies) <= 10000)
    zpe = float(freq["zero_point_energy"])
    assert 0.0 <= zpe <= 100.0
    thermo = freq["thermodynamic_properties"]
    assert "temperature" in thermo
    assert "entropy" in thermo
    if expect_ts:
        assert "is_ts" in freq
        assert "ts_analysis" in freq
        assert "n_imaginary_frequencies" in freq["ts_analysis"]


@pytest.mark.parametrize(
    ("args", "output_name", "expect_freq", "expect_ts"),
    [
        (
            [
                "minima",
                "--strategy",
                "local",
                str(REACTANT),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--freq",
                "--output",
                "opt.xyz",
            ],
            "opt.xyz",
            True,
            False,
        ),
        (
            [
                "ts",
                "--strategy",
                "local",
                str(TS),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--freq",
                "--output",
                "tsopt.xyz",
            ],
            "tsopt.xyz",
            True,
            True,
        ),
        (
            [
                "minima",
                "--strategy",
                "interpolate",
                str(REACTANT),
                "--product",
                str(PRODUCT),
                "--backend",
                "uma",
                "--steps",
                "100",
                "--npoints",
                "3",
                "--interp",
                "geodesic",
                "--freq",
                "--output",
                "twoended.xyz",
            ],
            "twoended.xyz",
            False,  # multi-structure minima do not always emit a frequency JSON
            False,
        ),
        (
            [
                "ts",
                "--strategy",
                "local",
                str(TS),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--freq",
                "--require-ts",
                "--output",
                "ts_require.xyz",
            ],
            "ts_require.xyz",
            True,
            True,
        ),
        (
            [
                "ts",
                "--strategy",
                "growing_string",
                str(GSM_REACTANT),
                "--product",
                str(GSM_PRODUCT),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--npoints",
                "11",
                "--step-size",
                "0.1",
                "--distance-threshold",
                "0.5",
                "--freq",
                "--require-ts",
                "--output",
                "ts_gsm.xyz",
            ],
            "ts_gsm.xyz",
            True,
            True,
        ),
        (
            [
                "path",
                "--strategy",
                "interpolate",
                str(REACTANT),
                str(PRODUCT),
                "--backend",
                "uma",
                "--npoints",
                "5",
                "--interp",
                "geodesic",
                "--output",
                "interp.xyz",
            ],
            "interp.xyz",
            False,
            False,
        ),
        (
            [
                "path",
                "--strategy",
                "neb",
                str(REACTANT),
                str(PRODUCT),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--npoints",
                "5",
                "--spring-constant",
                "0.5",
                "--output",
                "neb.xyz",
            ],
            "neb.xyz",
            False,
            False,
        ),
        (
            [
                "path",
                "--strategy",
                "cineb",
                str(REACTANT),
                str(PRODUCT),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--npoints",
                "5",
                "--spring-constant",
                "0.5",
                "--output",
                "cineb.xyz",
            ],
            "cineb.xyz",
            False,
            False,
        ),
        (
            [
                "path",
                "--strategy",
                "irc",
                str(TS),
                "--backend",
                "uma",
                "--steps",
                "50",
                "--step-size",
                "0.1",
                "--direction",
                "both",
                "--output",
                "irc.xyz",
            ],
            "irc.xyz",
            False,
            False,
        ),
        (
            [
                "ts",
                "--strategy",
                "local",
                str(TS),
                "--backend",
                "uma",
                "--local-optimizer",
                "rfo",
                "--steps",
                "200",
                "--freq",
                "--output",
                "ts_rfo.xyz",
            ],
            "ts_rfo.xyz",
            True,
            True,
        ),
        (
            [
                "minima",
                "--strategy",
                "local",
                str(REACTANT),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--force-finite-diff-hessian",
                "--freq",
                "--output",
                "opt_fd.xyz",
            ],
            "opt_fd.xyz",
            True,
            False,
        ),
        (
            [
                "ts",
                "--strategy",
                "local",
                str(TS),
                "--backend",
                "uma",
                "--steps",
                "100",
                "--force-finite-diff-hessian",
                "--freq",
                "--output",
                "ts_fd.xyz",
            ],
            "ts_fd.xyz",
            True,
            True,
        ),
        (
            [
                "minima",
                "--strategy",
                "local",
                str(REACTANT),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--temperature",
                "500.0",
                "--freq",
                "--output",
                "opt_t500.xyz",
            ],
            "opt_t500.xyz",
            True,
            False,
        ),
        (
            [
                "ts",
                "--strategy",
                "local",
                str(TS),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--constraints",
                "fix 0",
                "--freq",
                "--output",
                "ts_fix.xyz",
            ],
            "ts_fix.xyz",
            True,
            True,
        ),
        (
            [
                "minima",
                "--strategy",
                "local",
                str(REACTANT),
                "--backend",
                "uma",
                "--steps",
                "200",
                "--constraints",
                "fix 0",
                "--freq",
                "--output",
                "opt_fix.xyz",
            ],
            "opt_fix.xyz",
            True,
            False,
        ),
    ],
    ids=[
        "minima_local_freq",
        "ts_local_freq",
        "minima_interpolate_freq",
        "ts_local_require_ts",
        "ts_growing_string_freq",
        "path_interpolate",
        "path_neb",
        "path_cineb",
        "path_irc",
        "ts_rfo_freq",
        "minima_fdhess_freq",
        "ts_fdhess_freq",
        "minima_temp500_freq",
        "ts_constrained_freq",
        "minima_constrained_freq",
    ],
)
def test_cli_example_scenarios(
    tmp_path: Path,
    args: list[str],
    output_name: str,
    expect_freq: bool,
    expect_ts: bool,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        result = runner.invoke(main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        out = Path(cwd) / output_name
        assert out.exists(), f"missing output {out}"
        if expect_freq:
            _assert_frequency_json(out.with_suffix(".json"), expect_ts=expect_ts)


def test_irc_api_writes_trajectory(tmp_path: Path) -> None:
    ts = read(TS)
    explorer = famex.Explorer(atoms=ts, backend="uma", target="path", strategy="irc")
    result = explorer.run(steps=50, step_size=0.1, fmax=0.05, direction="both")
    trajectory = result["trajectory"]
    assert len(trajectory) > 1
    out = tmp_path / "irc.xyz"
    explorer.save_trajectory(trajectory, str(out))
    assert out.exists()
    frames = read(out, index=":")
    assert len(frames) == len(trajectory)


def test_growing_string_api_finds_ts(tmp_path: Path) -> None:
    reactant = read(GSM_REACTANT)
    product = read(GSM_PRODUCT)
    explorer = famex.Explorer(
        atoms=[reactant, product],
        backend="uma",
        target="ts",
        strategy="growing_string",
    )
    result = explorer.run(
        npoints=11,
        fmax=0.05,
        steps=200,
        step_size=0.1,
        distance_threshold=0.5,
        calculate_frequencies=True,
        require_ts=True,
    )
    assert "optimized_atoms" in result
    assert len(result["optimized_atoms"]) == len(reactant)
    freq = result.get("frequency_analysis") or {}
    assert freq.get("is_ts") is True
    n_imag = (freq.get("ts_analysis") or {}).get("n_imaginary_frequencies")
    assert n_imag == 1
    out = tmp_path / "gsm_ts.xyz"
    explorer.save_structure(result["optimized_atoms"], str(out))
    assert out.exists()


def test_thermochemistry_from_uma_frequencies() -> None:
    atoms = read(REACTANT)
    explorer = famex.Explorer(atoms=atoms, backend="uma", target="minima", strategy="local")
    result = explorer.run(fmax=0.05, steps=200, calculate_frequencies=True)
    optimized = result["optimized_atoms"]
    freq = result["frequency_analysis"]
    props = ThermodynamicProperties(
        freq["frequencies"],
        optimized,
        temperature=298.15,
        method="rrho",
        solvent="none",
    ).calculate_complete_thermodynamics(energy=optimized.get_potential_energy())
    assert np.isfinite(props["gibbs_free_energy"])
