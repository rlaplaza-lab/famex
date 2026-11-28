"""Choose and write the default geometry for a run, plus optional history."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from ase import Atoms


class RunOutputResult(NamedTuple):
    written_paths: list[str]
    frame_count: int


def default_output_name(stem: str, target: str, strategy: str) -> str:
    """Return the default artifact name for a target and strategy.

    ``local`` omits the strategy token: ``{stem}.minima.xyz``, ``{stem}.ts.xyz``.
    Every other strategy keeps its registry name: ``{stem}.ts.growing_string.xyz``.
    """
    if strategy == "local":
        return f"{stem}.{target}.xyz"
    return f"{stem}.{target}.{strategy}.xyz"


def history_output_name(stem: str, target: str, strategy: str, tag: str) -> str:
    """Return an optional-file name with ``tag`` on the last token.

    Examples: ``water.minima_trajectory.xyz``, ``water.ts.growing_string_path.xyz``,
    ``ts.path.irc_forward.xyz``.
    """
    if strategy == "local":
        return f"{stem}.{target}_{tag}.xyz"
    return f"{stem}.{target}.{strategy}_{tag}.xyz"


def select_default_frames(result: dict[str, Any], target: str) -> Atoms | list[Atoms]:
    """Select the geometry the target writes by default.

    A transition state is always one structure. A path is the end-to-end
    trajectory. Minima is the optimized structure, or one frame per image
    when interpolation optimized several minima.
    """
    optimized = result.get("optimized_atoms")
    if target == "minima":
        return _minima_frames(optimized)
    if target == "ts":
        return _single_structure(optimized)
    if target == "path":
        return _path_frames(result.get("trajectory", optimized))
    msg = f"Unknown target: {target}"
    raise ValueError(msg)


def write_run_outputs(
    result: dict[str, Any],
    stem: str,
    *,
    target: str,
    strategy: str,
    output: str | Path | None = None,
    history: bool = False,
) -> RunOutputResult:
    """Write the default artifact and, when asked, geometry history.

    History files are optimizer steps, a TS search path, NEB iteration frames,
    or the IRC forward and reverse branches. ``output`` replaces only the
    default file. History files use the canonical names and are placed in
    that file's directory.

    Returns written paths (default file first) and the default artifact frame count.
    """
    if output is not None:
        default_path = str(Path(output))
    else:
        default_path = default_output_name(stem, target, strategy)
    written = [default_path]
    default_frames = select_default_frames(result, target)
    frame_count = len(default_frames) if isinstance(default_frames, list) else 1
    _write_frames(default_frames, default_path)

    if not history:
        return RunOutputResult(written, frame_count)

    if target == "path" and strategy == "irc":
        for tag, key in (("forward", "forward_path"), ("reverse", "backward_path")):
            branch = _frame_list(result.get(key))
            if branch is None:
                continue
            path = _place_history(default_path, stem, target, strategy, tag)
            _write_frames(branch, path)
            written.append(path)
        return RunOutputResult(written, frame_count)

    trajectory = _frame_list(result.get("optimization_trajectory"))
    if trajectory is not None:
        path = _place_history(default_path, stem, target, strategy, "trajectory")
        _write_frames(trajectory, path)
        written.append(path)

    if target == "ts":
        search = result.get("search_path")
        if strategy == "growing_string" and _frame_list(search) is None:
            search = result.get("trajectory")
        band = _frame_list(search)
        if band is not None:
            path = _place_history(default_path, stem, target, strategy, "path")
            _write_frames(band, path)
            written.append(path)

    return RunOutputResult(written, frame_count)


def _place_history(
    default_path: str | Path,
    stem: str,
    target: str,
    strategy: str,
    tag: str,
) -> str:
    filename = Path(history_output_name(stem, target, strategy, tag)).name
    return str(Path(default_path).parent / filename)


def _write_frames(frames: Atoms | list[Atoms], path: str | Path) -> None:
    from famex.core.file_io import write_atoms_safely, write_trajectory_safely

    if isinstance(frames, Atoms):
        write_atoms_safely(frames, path)
        return
    write_trajectory_safely(frames, path)


def _minima_frames(value: Any) -> Atoms | list[Atoms]:
    if isinstance(value, Atoms):
        return value
    frames = _frame_list(value)
    if frames is None:
        msg = "Minima output must be one or more structures"
        raise TypeError(msg)
    if len(frames) == 1:
        return frames[0]
    return frames


def _single_structure(value: Any) -> Atoms:
    if isinstance(value, Atoms):
        return value
    frames = _frame_list(value)
    if frames is not None and len(frames) == 1:
        return frames[0]
    msg = "TS output must be a single structure"
    raise TypeError(msg)


def _path_frames(value: Any) -> list[Atoms]:
    if isinstance(value, Atoms):
        return [value]
    frames = _frame_list(value)
    if frames is None:
        msg = "Path output must be a reactant-to-product trajectory"
        raise TypeError(msg)
    return frames


def _frame_list(value: Any) -> list[Atoms] | None:
    if isinstance(value, list) and value and all(isinstance(item, Atoms) for item in value):
        return value
    return None


__all__ = [
    "RunOutputResult",
    "default_output_name",
    "history_output_name",
    "select_default_frames",
    "write_run_outputs",
]
