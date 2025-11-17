"""CLI helper functions for QME.

This module provides utility functions for the QME command line interface,
including file I/O operations, argument parsing, and data formatting.
"""

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

from qme.io.xyz_io import read_xyz_with_metadata, write_xyz_with_metadata


def parse_kv_pairs(pairs: list[str]) -> dict[str, object]:
    """Parse key=value pairs from CLI into a dict with best-effort typing.

    Parameters
    ----------
    pairs : List[str]
        List of key=value strings from CLI arguments

    Returns
    -------
    Dict[str, object]
        Dictionary with parsed key-value pairs. Values are automatically
        converted to appropriate types (bool, int, float, or str).

    Examples
    --------
    >>> parse_kv_pairs(["k=5.0", "steps=100", "verbose=true"])
    {'k': 5.0, 'steps': 100, 'verbose': True}

    """
    result: dict[str, object] = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Try to coerce value types
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
    """Load atoms from an XYZ file.

    Parameters
    ----------
    path : str
        Path to the XYZ file

    Returns
    -------
    Atoms
        ASE Atoms object. If the file contains multiple frames,
        returns the last frame.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the file format is invalid

    """
    # Use custom XYZ reader for .xyz files to preserve metadata
    if path.lower().endswith(".xyz"):
        geom = read_xyz_with_metadata(path, frame="last")
        # Handle case where read_xyz_with_metadata returns a list
        if isinstance(geom, list):
            geom = geom[-1]  # Take the last frame
        # Convert Geometry to Atoms for CLI compatibility
        atoms = Atoms(
            symbols=geom.get_chemical_symbols(),
            positions=geom.get_positions(),
            cell=geom.get_cell(),
            pbc=geom.get_pbc(),
        )
        # Copy over info including charge/spin
        if hasattr(geom, "info") and geom.info:
            atoms.info = dict(geom.info)
        return atoms

    # Use ASE for non-XYZ files
    atoms_result = ase_read(path)
    if isinstance(atoms_result, list):
        # If multiple frames in XYZ, take the last one by default
        atoms = atoms_result[-1]
    else:
        atoms = atoms_result
    return atoms


def load_path_structures(structures: tuple[str, ...]) -> list[Atoms]:
    """Load structures for path optimization from variadic file inputs.

    Handles multiple input methods:
    - Multiple positional arguments: load each file as a structure
    - Single multi-frame XYZ: load all frames as structures
    - Single single-frame XYZ: load as single structure

    Parameters
    ----------
    structures : tuple[str, ...]
        Tuple of file paths. Can be:
        - Multiple files: each file becomes one structure
        - Single file: if multi-frame XYZ, all frames become structures
        - Single file: if single-frame XYZ, becomes one structure

    Returns
    -------
    list[Atoms]
        List of Atoms objects ready for path optimization

    Raises
    ------
    FileNotFoundError
        If any file doesn't exist
    ValueError
        If file format is invalid or no structures found

    Examples
    --------
    >>> # Multiple files
    >>> load_path_structures(("reactant.xyz", "product.xyz"))
    [Atoms(...), Atoms(...)]

    >>> # Single multi-frame file
    >>> load_path_structures(("path_guess.xyz",))  # Contains 5 frames
    [Atoms(...), Atoms(...), Atoms(...), Atoms(...), Atoms(...)]

    >>> # Single single-frame file
    >>> load_path_structures(("ts.xyz",))
    [Atoms(...)]

    """
    if not structures:
        raise ValueError("At least one structure file must be provided")

    atoms_list: list[Atoms] = []

    # If multiple files provided, load each as a structure
    if len(structures) > 1:
        for path in structures:
            atoms = load_atoms_from_xyz(path)
            atoms_list.append(atoms)
        return atoms_list

    # Single file: check if it's multi-frame
    single_path = structures[0]

    # Try to read as multi-frame first (for XYZ files)
    if single_path.lower().endswith(".xyz"):
        try:
            # Try reading all frames
            geom_list = read_xyz_with_metadata(single_path, frame="all")
            if isinstance(geom_list, list) and len(geom_list) > 1:
                # Multi-frame file: convert all frames to Atoms
                for geom in geom_list:
                    atoms = Atoms(
                        symbols=geom.get_chemical_symbols(),
                        positions=geom.get_positions(),
                        cell=geom.get_cell(),
                        pbc=geom.get_pbc(),
                    )
                    if hasattr(geom, "info") and geom.info:
                        atoms.info = dict(geom.info)
                    atoms_list.append(atoms)
                return atoms_list
            elif isinstance(geom_list, list) and len(geom_list) == 1:
                # Single frame: convert to Atoms
                geom = geom_list[0]
                atoms = Atoms(
                    symbols=geom.get_chemical_symbols(),
                    positions=geom.get_positions(),
                    cell=geom.get_cell(),
                    pbc=geom.get_pbc(),
                )
                if hasattr(geom, "info") and geom.info:
                    atoms.info = dict(geom.info)
                atoms_list.append(atoms)
                return atoms_list
            else:
                # Single Geometry object
                geom = geom_list
                atoms = Atoms(
                    symbols=geom.get_chemical_symbols(),
                    positions=geom.get_positions(),
                    cell=geom.get_cell(),
                    pbc=geom.get_pbc(),
                )
                if hasattr(geom, "info") and geom.info:
                    atoms.info = dict(geom.info)
                atoms_list.append(atoms)
                return atoms_list
        except Exception:
            # Fall back to regular loading if multi-frame read fails
            pass

    # Fallback: use regular single-file loading
    atoms = load_atoms_from_xyz(single_path)
    atoms_list.append(atoms)
    return atoms_list


def _coerce_to_atoms(obj: Any) -> Atoms:
    """Best-effort conversion of various result shapes into an ASE Atoms.

    Parameters
    ----------
    obj : Any
        Object to convert to Atoms. Can be:
        - ASE Atoms object (returned as-is)
        - Dictionary with 'optimized_atoms' key
        - List/tuple of Atoms objects (first one returned)

    Returns
    -------
    Atoms
        ASE Atoms object

    Raises
    ------
    ValueError
        If obj cannot be converted to Atoms

    """
    if isinstance(obj, Atoms):
        return obj
    # Strategy dict result
    if isinstance(obj, dict) and "optimized_atoms" in obj:
        # Dict values are Any, but we know optimized_atoms is Atoms | list[Atoms] at runtime
        return obj["optimized_atoms"]  # type: ignore[no-any-return]
    # List/tuple of Atoms (take first)
    if isinstance(obj, list | tuple) and obj and isinstance(obj[0], Atoms):
        return obj[0]
    # Path string to an XYZ; try to read
    if isinstance(obj, str) and os.path.exists(obj):
        result = ase_read(obj)
        if isinstance(result, list):
            return result[0]  # Return first structure if multiple
        return result
    msg = f"Cannot coerce object of type {type(obj)} to ASE Atoms"
    raise TypeError(msg)


def write_atoms(
    atoms: Atoms | list[Atoms] | dict[str, Any] | str,
    out_path: str | None,
) -> str | None:
    """Write atoms or trajectory to a file.

    Parameters
    ----------
    atoms : Atoms or List[Atoms] or Any
        ASE Atoms object, list of Atoms objects (trajectory), or other result to write
    out_path : str | None
        Output file path. If None, no file is written.

    Returns
    -------
    str | None
        Output path if file was written, None otherwise

    Raises
    ------
    ValueError
        If atoms cannot be converted to valid structure or path is unsafe
    OSError
        If file cannot be written

    """
    if not out_path:
        return None

    # SECURITY: Validate path doesn't contain traversal patterns
    # Allow absolute paths but reject suspicious patterns
    if ".." in out_path or "\x00" in out_path:
        raise ValueError(f"Unsafe output path detected: {out_path}")

    # Ensure output directory exists
    parent_dir = Path(out_path).parent
    if str(parent_dir) != ".":
        parent_dir.mkdir(parents=True, exist_ok=True)

    # Use custom XYZ writer for .xyz files to preserve metadata
    if out_path.lower().endswith(".xyz"):
        # Handle lists of Atoms objects (trajectories from NEB/CI-NEB)
        if isinstance(atoms, list) and atoms and isinstance(atoms[0], Atoms):
            write_xyz_with_metadata(atoms, out_path)
            return out_path

        # Handle single Atoms object or other results
        atoms_obj = _coerce_to_atoms(atoms)
        write_xyz_with_metadata(atoms_obj, out_path)
        return out_path

    # Use ASE for non-XYZ files
    # Handle lists of Atoms objects (trajectories from NEB/CI-NEB)
    if isinstance(atoms, list) and atoms and isinstance(atoms[0], Atoms):
        ase_write(out_path, atoms)
        return out_path

    # Handle single Atoms object or other results
    atoms_obj = _coerce_to_atoms(atoms)
    ase_write(out_path, atoms_obj)
    return out_path


def validate_frequency_analysis(frequency_analysis: Any) -> list[str]:
    """Validate frequency analysis result structure and return list of warnings.

    Parameters
    ----------
    frequency_analysis : dict[str, Any]
        Frequency analysis results dictionary

    Returns
    -------
    list[str]
        List of warning messages (empty if no issues found)

    """
    warnings = []

    if not isinstance(frequency_analysis, dict):
        warnings.append(f"Frequency analysis is not a dictionary (got {type(frequency_analysis)})")
        return warnings

    # Check for required keys
    required_keys = ["frequencies", "zero_point_energy", "thermodynamic_properties"]
    for key in required_keys:
        if key not in frequency_analysis:
            warnings.append(f"Missing required key: {key}")

    # Validate frequencies
    frequencies = frequency_analysis.get("frequencies", [])
    if not isinstance(frequencies, list | tuple):
        warnings.append(f"Frequencies should be a list/tuple, got {type(frequencies)}")
    elif len(frequencies) == 0:
        warnings.append("Frequencies list is empty")
    else:
        # Check for NaN or invalid values
        try:
            freq_array = np.array(frequencies)
            if np.any(np.isnan(freq_array)):
                warnings.append("Frequencies contain NaN values")
            if np.all(np.abs(freq_array) < 1e-6):
                warnings.append("All frequencies are near zero (may indicate calculation error)")
        except (ValueError, TypeError):
            warnings.append("Frequencies contain invalid values")

    # Validate zero-point energy
    zpe = frequency_analysis.get("zero_point_energy")
    if zpe is None:
        warnings.append("Zero-point energy is missing")
    elif not isinstance(zpe, int | float):
        warnings.append(f"Zero-point energy should be numeric, got {type(zpe)}")
    elif np.isnan(zpe) or np.isinf(zpe):
        warnings.append("Zero-point energy is NaN or infinite")

    # Validate thermodynamic properties
    thermo = frequency_analysis.get("thermodynamic_properties", {})
    if not isinstance(thermo, dict):
        warnings.append(f"Thermodynamic properties should be a dict, got {type(thermo)}")
    elif "temperature" not in thermo:
        warnings.append("Temperature missing from thermodynamic properties")

    return warnings


def print_frequency_summary(frequency_analysis: dict[str, Any], target: str = "minima") -> None:
    """Print a formatted summary of frequency analysis results.

    This function validates the frequency analysis results and handles edge cases
    gracefully, providing informative output even when some data is missing or invalid.

    Parameters
    ----------
    frequency_analysis : dict[str, Any]
        Frequency analysis results from Explorer.calculate_frequencies()
    target : str, default="minima"
        Target type ("minima", "ts", or "path") for appropriate validation messages

    """
    if not frequency_analysis:
        click.echo("Warning: Frequency analysis dictionary is empty", err=True)
        return

    # Validate frequency analysis structure
    warnings = validate_frequency_analysis(frequency_analysis)
    if warnings:
        click.echo("Warning: Frequency analysis validation issues:", err=True)
        for warning in warnings:
            click.echo(f"  - {warning}", err=True)

    frequencies = frequency_analysis.get("frequencies", [])
    n_modes = len(frequencies) if isinstance(frequencies, list | tuple) else 0
    zpe = frequency_analysis.get("zero_point_energy", 0.0)

    # Safely extract thermodynamic properties
    thermo = frequency_analysis.get("thermodynamic_properties", {})
    if not isinstance(thermo, dict):
        thermo = {}
    temperature = thermo.get("temperature", 298.15)

    # Calculate free energy correction (G - E) = H - TS - E = -TS (for ideal gas)
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

    # Validation status
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
        # For path strategies, show both minima and TS analysis if available
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
    else:  # minima
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

    # Show key frequencies
    if frequencies and isinstance(frequencies, list | tuple) and len(frequencies) > 0:
        try:
            # Handle complex frequencies by taking real part for comparison
            # Show first few real frequencies and any imaginary ones
            real_freqs = [f for f in frequencies if np.real(f) > 0]
            imaginary_freqs = [f for f in frequencies if np.real(f) < 0]

            if real_freqs:
                lowest_real = real_freqs[:3]  # First 3 real frequencies
                freq_str = ", ".join(f"{np.real(f):.1f}" for f in lowest_real)
                click.echo(f"  Lowest frequencies: {freq_str} cm⁻¹")

            if imaginary_freqs:
                # Show all imaginary frequencies
                imag_str = ", ".join(f"{np.real(f):.1f}" for f in imaginary_freqs)
                click.echo(f"  Imaginary frequencies: {imag_str} cm⁻¹")
        except (ValueError, TypeError) as e:
            click.echo(f"  Warning: Could not display frequencies: {e}", err=True)


def save_results_json(results: Any, output_path: str) -> None:
    """Save complete results dictionary to JSON file.

    This function validates the results structure, especially frequency analysis,
    before saving to ensure the JSON output is complete and valid.

    Parameters
    ----------
    results : dict[str, Any]
        Complete results dictionary from Explorer.run()
    output_path : str
        Path to save JSON file (will replace .xyz extension with .json)

    Raises
    ------
    ValueError
        If results structure is invalid
    OSError
        If file cannot be written

    """
    # Convert output path to JSON
    json_path = os.path.splitext(output_path)[0] + ".json"

    # Validate that results is a dictionary
    if not isinstance(results, dict):
        msg = f"Results must be a dictionary, got {type(results)}"
        raise ValueError(msg)

    # Type narrowing: results is now dict[str, Any]
    results_dict: dict[str, Any] = results

    # Create a serializable version of results
    serializable_results: dict[str, Any] = {}
    for key, value in results_dict.items():
        if key == "optimized_atoms":
            # Skip Atoms objects - they're saved separately as XYZ
            continue
        if key == "frequency_analysis" and value:
            # Validate frequency analysis structure before serialization
            if not isinstance(value, dict):
                click.echo(
                    f"Warning: frequency_analysis is not a dict (got {type(value)}), skipping",
                    err=True,
                )
                continue

            # Validate frequency analysis using validation function
            warnings = validate_frequency_analysis(value)
            if warnings:
                click.echo(
                    "Warning: Frequency analysis validation issues before saving JSON:",
                    err=True,
                )
                for warning in warnings:
                    click.echo(f"  - {warning}", err=True)

            # Ensure frequency analysis is serializable
            freq_data = dict(value)
            # Convert numpy arrays to lists
            for k, v in freq_data.items():
                if hasattr(v, "tolist"):
                    try:
                        freq_data[k] = v.tolist()
                    except (AttributeError, ValueError):
                        # If tolist() fails, try converting to Python type
                        try:
                            if np.isscalar(v):
                                # Type narrowing: v is a scalar, try to convert to float
                                try:
                                    freq_data[k] = float(v)  # type: ignore[arg-type]
                                except (ValueError, TypeError):
                                    freq_data[k] = str(v)
                            else:
                                freq_data[k] = str(v)
                        except (ValueError, TypeError):
                            freq_data[k] = str(v)
                elif isinstance(v, np.ndarray):
                    freq_data[k] = v.tolist()
                elif isinstance(v, np.integer | np.floating):
                    freq_data[k] = float(v) if isinstance(v, np.floating) else int(v)

            serializable_results[key] = freq_data
        else:
            # Try to make other values serializable
            try:
                if isinstance(value, np.ndarray) or hasattr(value, "tolist"):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, np.integer | np.floating):
                    serializable_results[key] = (
                        float(value) if isinstance(value, np.floating) else int(value)
                    )
                else:
                    serializable_results[key] = value
            except (TypeError, AttributeError, ValueError):
                # Skip non-serializable values
                continue

    # Verify that frequency_analysis was included if it exists in results_dict
    if "frequency_analysis" in results_dict and "frequency_analysis" not in serializable_results:
        click.echo(
            "Warning: frequency_analysis was present in results but could not be serialized",
            err=True,
        )

    # Save to JSON
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
