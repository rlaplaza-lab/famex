"""Whitelist pass-through keyword arguments for strategy runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LOCAL_OPTIMIZATION_RUN_KEYS: frozenset[str] = frozenset(
    {
        "local_optimizer_name",
        "verbose",
        "temperature",
        "record_history",
        "require_ts",
        "initial_hessian",
    }
)

NEB_OPTIMIZER_KEYS: frozenset[str] = frozenset(
    {
        "dt",
        "maxstep",
        "dtmax",
        "Nmin",
        "finc",
        "fdec",
        "astart",
        "fa",
        "record_history",
        "spring_constant_lower",
        "rmsd_threshold",
        "energy_threshold",
    }
)

GSM_RUN_KEYS: frozenset[str] = frozenset(
    {
        "npoints",
        "max_images",
        "fmax",
        "steps",
        "perp_thresh",
        "perp_thresh_relaxed",
        "reparam_every",
        "reparam_every_full",
        "optimize_endpoints",
        "refine_ts",
        "local_optimizer_name",
        "reparam_check",
        "distance_threshold",
        "step_size",
        "temperature",
        "record_history",
        "require_ts",
        "ts_refinement_fmax",
        "ts_refinement_steps",
        "ts_refinement_candidates",
    }
)

PATH_INTERPOLATE_RUN_KEYS: frozenset[str] = frozenset({"record_history", "verbose"})

TS_GUESS_RUN_KEYS: frozenset[str] = frozenset(
    {
        "local_optimizer_name",
        "record_history",
        "verbose",
        "temperature",
        "cleanup_frequencies",
        "path_fmax",
        "require_ts",
        "initial_hessian",
    }
)

IRC_RUN_KEYS: frozenset[str] = frozenset(
    {
        "local_optimizer_name",
        "verbose",
        "imag_threshold",
        "record_history",
    }
)

DHS_RUN_KEYS: frozenset[str] = frozenset(
    {
        "record_history",
        "max_macro_steps",
        "local_optimizer_name",
        "verbose",
        "temperature",
    }
)


def pick_pass_through_kwargs(kwargs: Mapping[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    """Return only keyword arguments that are allowed for downstream pass-through."""
    return {key: value for key, value in kwargs.items() if key in allowed}


__all__ = [
    "DHS_RUN_KEYS",
    "GSM_RUN_KEYS",
    "IRC_RUN_KEYS",
    "LOCAL_OPTIMIZATION_RUN_KEYS",
    "NEB_OPTIMIZER_KEYS",
    "PATH_INTERPOLATE_RUN_KEYS",
    "TS_GUESS_RUN_KEYS",
    "pick_pass_through_kwargs",
]
