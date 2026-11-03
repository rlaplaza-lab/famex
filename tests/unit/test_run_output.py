"""Tests for target-based output names and geometry-history recording."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from famex.core.explorer import Explorer
from famex.io.modes import modes_output_dir
from famex.io.run_output import (
    default_output_name,
    history_output_name,
    select_default_frames,
    write_run_outputs,
)
from famex.io.xyz_io import read_xyz_with_metadata


def test_default_names_drop_local_and_keep_strategy() -> None:
    assert default_output_name("water", "minima", "local") == "water.minima.xyz"
    assert default_output_name("water", "ts", "local") == "water.ts.xyz"
    assert default_output_name("water", "ts", "growing_string") == "water.ts.growing_string.xyz"
    assert default_output_name("reactant", "path", "neb") == "reactant.path.neb.xyz"
    assert default_output_name("ts", "path", "irc") == "ts.path.irc.xyz"


def test_mode_directory_sits_beside_the_geometry_file() -> None:
    assert modes_output_dir("water.minima.xyz") == Path("water.minima_modes")
    assert modes_output_dir("/tmp/custom/out.xyz") == Path("/tmp/custom/out_modes")


def test_history_names_append_tag_to_last_token() -> None:
    assert history_output_name("water", "minima", "local", "trajectory") == (
        "water.minima_trajectory.xyz"
    )
    assert history_output_name("water", "ts", "growing_string", "path") == (
        "water.ts.growing_string_path.xyz"
    )
    assert history_output_name("water", "ts", "cineb", "trajectory") == (
        "water.ts.cineb_trajectory.xyz"
    )
    assert history_output_name("ts", "path", "irc", "forward") == "ts.path.irc_forward.xyz"
    assert history_output_name("ts", "path", "irc", "reverse") == "ts.path.irc_reverse.xyz"
    assert history_output_name("r", "path", "neb", "trajectory") == "r.path.neb_trajectory.xyz"


def test_ts_default_is_the_single_saddle_not_the_band() -> None:
    saddle = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    band = [saddle, saddle.copy(), saddle.copy()]
    result = {"optimized_atoms": saddle, "trajectory": band, "search_path": band}

    selected = select_default_frames(result, "ts")

    assert selected is saddle


def test_path_default_is_the_end_to_end_trajectory() -> None:
    reactant = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    product = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    path = [reactant, product]
    result = {"optimized_atoms": reactant, "trajectory": path}

    assert select_default_frames(result, "path") == path


def test_write_run_outputs_places_history_beside_custom_output(tmp_path) -> None:
    saddle = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]])
    saddle.info = {"charge": 0, "spin": 1}
    step = saddle.copy()
    step.info = {"charge": 0, "spin": 1, "step": 0}
    band = [saddle.copy(), saddle.copy()]
    for index, image in enumerate(band):
        image.info = {"charge": 0, "spin": 1, "image": index}
    result = {
        "optimized_atoms": saddle,
        "optimization_trajectory": [step],
        "search_path": band,
        "trajectory": band,
    }
    custom = tmp_path / "custom.xyz"

    written = write_run_outputs(
        result,
        str(tmp_path / "water"),
        target="ts",
        strategy="growing_string",
        output=str(custom),
        history=True,
    )

    assert written[0] == str(custom)
    assert custom.exists()
    assert (tmp_path / "water.ts.growing_string_trajectory.xyz").exists()
    assert (tmp_path / "water.ts.growing_string_path.xyz").exists()
    frames = read_xyz_with_metadata(tmp_path / "water.ts.growing_string_path.xyz", frame="all")
    assert isinstance(frames, list)
    assert len(frames) == 2


def test_irc_history_writes_only_branches_that_exist(tmp_path) -> None:
    ts = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]])
    forward = [ts.copy()]
    result = {
        "optimized_atoms": ts,
        "trajectory": [ts, *forward],
        "forward_path": forward,
        "backward_path": [],
    }

    written = write_run_outputs(
        result,
        "ts",
        target="path",
        strategy="irc",
        output=str(tmp_path / "ts.path.irc.xyz"),
        history=True,
    )

    assert (tmp_path / "ts.path.irc.xyz").exists()
    assert (tmp_path / "ts.path.irc_forward.xyz").exists()
    assert not (tmp_path / "ts.path.irc_reverse.xyz").exists()
    assert len(written) == 2


def test_record_history_local_minima_keeps_more_than_the_final_structure(
    h2_molecule,
) -> None:
    atoms = h2_molecule.copy()
    positions = atoms.get_positions()
    positions[1, 0] += 0.4
    atoms.set_positions(positions)
    explorer = Explorer(atoms, backend="mock", target="minima", strategy="local")

    plain = explorer.run(steps=2, fmax=1e-8)
    recorded = explorer.run(steps=2, fmax=1e-8, record_history=True)

    assert "optimization_trajectory" not in plain
    frames = recorded["optimization_trajectory"]
    assert isinstance(frames, list)
    assert len(frames) > 1
    assert all(isinstance(frame, Atoms) for frame in frames)
