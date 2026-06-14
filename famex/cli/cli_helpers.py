"""CLI helper functions for FAMEX."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from famex.io.xyz_io import read_xyz_with_metadata, write_xyz_with_metadata


def parse_kv_pairs(pairs: list[str]) -> dict[str, object]:
    """Parse key=value pairs from CLI into a dict with best-effort typing."""
    result: dict[str, object] = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in ("true", "false"):
            coerced: object = value.lower() == "true"
        else:
            try:
                coerced = float(value) if "." in value else int(value)
            except ValueError:
                coerced = value
        result[key] = coerced
    return result


def load_atoms_from_xyz(path: str) -> Atoms:
    """Load atoms from an XYZ file."""
    if path.lower().endswith(".xyz"):
        geom = read_xyz_with_metadata(path, frame="last")
        if isinstance(geom, list):
            geom = geom[-1]
        return geom

    atoms_result = ase_read(path)
    if isinstance(atoms_result, list):
        atoms = atoms_result[-1]
    else:
        atoms = atoms_result
    return atoms


def load_path_structures(structures: tuple[str, ...]) -> list[Atoms]:
    """Load structures for path optimization from variadic file inputs."""
    if not structures:
        raise ValueError("At least one structure file must be provided")

    atoms_list: list[Atoms] = []

    if len(structures) > 1:
        for path in structures:
            atoms = load_atoms_from_xyz(path)
            atoms_list.append(atoms)
        return atoms_list

    single_path = structures[0]

    if single_path.lower().endswith(".xyz"):
        try:
            geom_result = read_xyz_with_metadata(single_path, frame="all")
            if isinstance(geom_result, list):
                atoms_list.extend(geom_result)
                return atoms_list
            else:
                atoms_list.append(geom_result)
                return atoms_list
        except Exception:
            pass

    atoms = load_atoms_from_xyz(single_path)
    atoms_list.append(atoms)
    return atoms_list


def _coerce_to_atoms(obj: Any) -> Atoms:
    """Best-effort conversion of various result shapes into an ASE Atoms."""
    if isinstance(obj, Atoms):
        return obj
    if isinstance(obj, dict) and "optimized_atoms" in obj:
        return obj["optimized_atoms"]  # type: ignore[no-any-return]
    if isinstance(obj, list | tuple) and obj and isinstance(obj[0], Atoms):
        return obj[0]
    if isinstance(obj, str) and os.path.exists(obj):
        result = ase_read(obj)
        if isinstance(result, list):
            return result[0]
        return result
    msg = f"Cannot coerce object of type {type(obj)} to ASE Atoms"
    raise TypeError(msg)


def write_atoms(
    atoms: Atoms | list[Atoms] | dict[str, Any] | str,
    out_path: str | None,
) -> str | None:
    """Write atoms or trajectory to a file."""
    if not out_path:
        return None

    if ".." in out_path or "\x00" in out_path:
        raise ValueError(f"Unsafe output path detected: {out_path}")

    parent_dir = Path(out_path).parent
    if str(parent_dir) != ".":
        parent_dir.mkdir(parents=True, exist_ok=True)

    if out_path.lower().endswith(".xyz"):
        if isinstance(atoms, list) and atoms and isinstance(atoms[0], Atoms):
            write_xyz_with_metadata(atoms, out_path)
            return out_path

        atoms_obj = _coerce_to_atoms(atoms)
        write_xyz_with_metadata(atoms_obj, out_path)
        return out_path

    if isinstance(atoms, list) and atoms and isinstance(atoms[0], Atoms):
        ase_write(out_path, atoms)
        return out_path

    atoms_obj = _coerce_to_atoms(atoms)
    ase_write(out_path, atoms_obj)
    return out_path


def validate_frequency_analysis(frequency_analysis: Any) -> list[str]:
    """Validate frequency analysis result structure and return list of warnings."""
    warnings = []

    if not isinstance(frequency_analysis, dict):
        warnings.append(f"Frequency analysis is not a dictionary (got {type(frequency_analysis)})")
        return warnings

    required_keys = ["frequencies", "zero_point_energy", "thermodynamic_properties"]
    for key in required_keys:
        if key not in frequency_analysis:
            warnings.append(f"Missing required key: {key}")

    frequencies = frequency_analysis.get("frequencies", [])
    if not isinstance(frequencies, list | tuple):
        warnings.append(f"Frequencies should be a list/tuple, got {type(frequencies)}")
    elif len(frequencies) == 0:
        warnings.append("Frequencies list is empty")
    else:
        try:
            freq_array = np.array(frequencies)
            if np.any(np.isnan(freq_array)):
                warnings.append("Frequencies contain NaN values")
            if np.all(np.abs(freq_array) < 1e-6):
                warnings.append("All frequencies are near zero (may indicate calculation error)")
        except (ValueError, TypeError):
            warnings.append("Frequencies contain invalid values")

    zpe = frequency_analysis.get("zero_point_energy")
    if zpe is None:
        warnings.append("Zero-point energy is missing")
    elif not isinstance(zpe, int | float):
        warnings.append(f"Zero-point energy should be numeric, got {type(zpe)}")
    elif np.isnan(zpe) or np.isinf(zpe):
        warnings.append("Zero-point energy is NaN or infinite")

    thermo = frequency_analysis.get("thermodynamic_properties", {})
    if not isinstance(thermo, dict):
        warnings.append(f"Thermodynamic properties should be a dict, got {type(thermo)}")
    elif "temperature" not in thermo:
        warnings.append("Temperature missing from thermodynamic properties")

    return warnings


def print_frequency_summary(frequency_analysis: dict[str, Any], target: str = "minima") -> None:
    """Print a formatted summary of frequency analysis results."""
    if not frequency_analysis:
        click.echo("Warning: Frequency analysis dictionary is empty", err=True)
        return

    warnings = validate_frequency_analysis(frequency_analysis)
    if warnings:
        click.echo("Warning: Frequency analysis validation issues:", err=True)
        for warning in warnings:
            click.echo(f"  - {warning}", err=True)

    frequencies = frequency_analysis.get("frequencies", [])
    n_modes = len(frequencies) if isinstance(frequencies, list | tuple) else 0
    zpe = frequency_analysis.get("zero_point_energy", 0.0)

    thermo = frequency_analysis.get("thermodynamic_properties", {})
    if not isinstance(thermo, dict):
        thermo = {}
    temperature = thermo.get("temperature", 298.15)

    entropy = thermo.get("entropy", 0.0)
    if isinstance(entropy, int | float) and isinstance(temperature, int | float):
        free_energy_correction = -entropy * temperature / 1000.0  # Convert K to eV/K
    else:
        free_energy_correction = 0.0

    click.echo(f"Frequency analysis completed ({n_modes} modes):")
    if isinstance(zpe, int | float) and not (np.isnan(zpe) or np.isinf(zpe)):
        click.echo(f"  Zero-point energy: {zpe:.4f} eV")
    else:
        click.echo("  Zero-point energy: N/A")

    if isinstance(free_energy_correction, int | float) and not (
        np.isnan(free_energy_correction) or np.isinf(free_energy_correction)
    ):
        click.echo(
            f"  Free energy correction ({temperature:.1f} K): {free_energy_correction:.4f} eV"
        )
    else:
        click.echo(f"  Free energy correction ({temperature:.1f} K): N/A")

    if target == "ts":
        is_ts = frequency_analysis.get("is_ts", False)
        ts_analysis = frequency_analysis.get("ts_analysis", {})
        if not isinstance(ts_analysis, dict):
            ts_analysis = {}
        n_imaginary = ts_analysis.get("n_imaginary_frequencies", 0)
        if is_ts:
            click.echo(f"  Status: ✓ Valid transition state ({n_imaginary} imaginary frequency)")
        else:
            click.echo(
                f"  Status: ✗ Invalid transition state ({n_imaginary} imaginary frequencies)",
            )
    elif target == "path":
        is_minimum = frequency_analysis.get("is_minimum", False)
        is_ts = frequency_analysis.get("is_ts", False)
        if is_ts:
            ts_analysis = frequency_analysis.get("ts_analysis", {})
            if not isinstance(ts_analysis, dict):
                ts_analysis = {}
            n_imaginary = ts_analysis.get("n_imaginary_frequencies", 0)
            click.echo(f"  Status: Transition state ({n_imaginary} imaginary frequency)")
        elif is_minimum:
            click.echo("  Status: ✓ Minimum")
        else:
            click.echo("  Status: Unknown (validation not performed)")
    else:
        is_minimum = frequency_analysis.get("is_minimum", False)
        minima_analysis = frequency_analysis.get("minima_analysis", {})
        if not isinstance(minima_analysis, dict):
            minima_analysis = {}
        n_imaginary = minima_analysis.get("n_significant_imaginary_frequencies", 0)
        if is_minimum:
            click.echo(f"  Status: ✓ Valid minimum ({n_imaginary} imaginary frequencies)")
        else:
            click.echo(
                f"  Status: ✗ Invalid minimum ({n_imaginary} significant imaginary frequencies)",
            )

    if frequencies and isinstance(frequencies, list | tuple) and len(frequencies) > 0:
        try:
            real_freqs = [f for f in frequencies if np.real(f) > 0]
            imaginary_freqs = [f for f in frequencies if np.real(f) < 0]

            if real_freqs:
                lowest_real = real_freqs[:3]
                freq_str = ", ".join(f"{np.real(f):.1f}" for f in lowest_real)
                click.echo(f"  Lowest frequencies: {freq_str} cm⁻¹")

            if imaginary_freqs:
                imag_str = ", ".join(f"{np.real(f):.1f}" for f in imaginary_freqs)
                click.echo(f"  Imaginary frequencies: {imag_str} cm⁻¹")
        except (ValueError, TypeError) as e:
            click.echo(f"  Warning: Could not display frequencies: {e}", err=True)


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable type."""
    if isinstance(value, np.ndarray) or hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (AttributeError, ValueError, TypeError):
            pass

    if isinstance(value, np.integer | np.floating):
        return float(value) if isinstance(value, np.floating) else int(value)

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass

    return str(value)


def _serialize_frequency_analysis(freq_analysis: Any) -> dict[str, Any] | None:
    """Serialize frequency analysis dictionary to JSON-compatible format."""
    if not isinstance(freq_analysis, dict):
        click.echo(
            f"Warning: frequency_analysis is not a dict (got {type(freq_analysis)}), skipping",
            err=True,
        )
        return None

    warnings = validate_frequency_analysis(freq_analysis)
    if warnings:
        click.echo(
            "Warning: Frequency analysis validation issues before saving JSON:",
            err=True,
        )
        for warning in warnings:
            click.echo(f"  - {warning}", err=True)

    freq_data: dict[str, Any] = {}
    for k, v in freq_analysis.items():
        serialized = _serialize_value(v)
        if serialized is not None:
            freq_data[k] = serialized

    return freq_data


def save_results_json(results: Any, output_path: str) -> None:
    """Save complete results dictionary to JSON file."""
    json_path = os.path.splitext(output_path)[0] + ".json"

    if not isinstance(results, dict):
        msg = f"Results must be a dictionary, got {type(results)}"
        raise ValueError(msg)

    results_dict: dict[str, Any] = results

    serializable_results: dict[str, Any] = {}
    for key, value in results_dict.items():
        if key == "optimized_atoms":
            continue

        if key == "frequency_analysis" and value:
            freq_data = _serialize_frequency_analysis(value)
            if freq_data is not None:
                serializable_results[key] = freq_data
        else:
            serialized = _serialize_value(value)
            if serialized is not None:
                serializable_results[key] = serialized

    if "frequency_analysis" in results_dict and "frequency_analysis" not in serializable_results:
        click.echo(
            "Warning: frequency_analysis was present in results but could not be serialized",
            err=True,
        )

    try:
        with open(json_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
        click.echo(f"Results saved to: {json_path}")
    except OSError as e:
        msg = f"Failed to write JSON file to {json_path}: {e}"
        raise OSError(msg) from e
    except (TypeError, ValueError) as e:
        msg = f"Failed to serialize results to JSON: {e}"
        raise ValueError(msg) from e


__all__ = [
    "load_atoms_from_xyz",
    "load_path_structures",
    "parse_kv_pairs",
    "print_frequency_summary",
    "save_results_json",
    "validate_frequency_analysis",
    "write_atoms",
]
