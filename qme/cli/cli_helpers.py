"""CLI helper functions for QME.

This module provides utility functions for the QME command line interface,
including file I/O operations, argument parsing, and data formatting.
"""

import json
import os
from typing import Any

import click
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

    Returns:
    -------
    Dict[str, object]
        Dictionary with parsed key-value pairs. Values are automatically
        converted to appropriate types (bool, int, float, or str).

    Examples:
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

    Returns:
    -------
    Atoms
        ASE Atoms object. If the file contains multiple frames,
        returns the last frame.

    Raises:
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the file format is invalid

    """
    # Use custom XYZ reader for .xyz files to preserve metadata
    if path.lower().endswith(".xyz"):
        geom = read_xyz_with_metadata(path, frame="last")
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
    atoms = ase_read(path)
    if isinstance(atoms, list):
        # If multiple frames in XYZ, take the last one by default
        atoms = atoms[-1]
    return atoms


def _coerce_to_atoms(obj: Any) -> Atoms:
    """Best-effort conversion of various result shapes into an ASE Atoms.

    Parameters
    ----------
    obj : Any
        Object to convert to Atoms. Can be:
        - ASE Atoms object (returned as-is)
        - Dictionary with 'optimized_atoms' key
        - List/tuple of Atoms objects (first one returned)

    Returns:
    -------
    Atoms
        ASE Atoms object

    Raises:
    ------
    ValueError
        If obj cannot be converted to Atoms

    """
    if isinstance(obj, Atoms):
        return obj
    # Strategy dict result
    if isinstance(obj, dict) and "optimized_atoms" in obj:
        return obj["optimized_atoms"]
    # List/tuple of Atoms (take first)
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], Atoms):
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
    out_path : Optional[str]
        Output file path. If None, no file is written.

    Returns:
    -------
    Optional[str]
        Output path if file was written, None otherwise

    Raises:
    ------
    ValueError
        If atoms cannot be converted to valid structure
    OSError
        If file cannot be written

    """
    if not out_path:
        return None
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

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


def print_frequency_summary(frequency_analysis: dict[str, Any], target: str = "minima") -> None:
    """Print a formatted summary of frequency analysis results.

    Parameters
    ----------
    frequency_analysis : dict[str, Any]
        Frequency analysis results from Explorer.calculate_frequencies()
    target : str, default="minima"
        Target type ("minima" or "ts") for appropriate validation messages

    """
    if not frequency_analysis:
        return

    frequencies = frequency_analysis.get("frequencies", [])
    n_modes = len(frequencies)
    zpe = frequency_analysis.get("zero_point_energy", 0.0)
    thermo = frequency_analysis.get("thermodynamic_properties", {})
    temperature = thermo.get("temperature", 298.15)

    # Calculate free energy correction (G - E) = H - TS - E = -TS (for ideal gas)
    entropy = thermo.get("entropy", 0.0)
    free_energy_correction = -entropy * temperature / 1000.0  # Convert K to eV/K

    click.echo(f"Frequency analysis completed ({n_modes} modes):")
    click.echo(f"  Zero-point energy: {zpe:.4f} eV")
    click.echo(f"  Free energy correction ({temperature:.1f} K): {free_energy_correction:.4f} eV")

    # Validation status
    if target == "ts":
        is_ts = frequency_analysis.get("is_ts", False)
        ts_analysis = frequency_analysis.get("ts_analysis", {})
        n_imaginary = ts_analysis.get("n_imaginary_frequencies", 0)
        if is_ts:
            click.echo(f"  Status: ✓ Valid transition state ({n_imaginary} imaginary frequency)")
        else:
            click.echo(
                f"  Status: ✗ Invalid transition state ({n_imaginary} imaginary frequencies)",
            )
    else:  # minima
        is_minimum = frequency_analysis.get("is_minimum", False)
        minima_analysis = frequency_analysis.get("minima_analysis", {})
        n_imaginary = minima_analysis.get("n_significant_imaginary_frequencies", 0)
        if is_minimum:
            click.echo(f"  Status: ✓ Valid minimum ({n_imaginary} imaginary frequencies)")
        else:
            click.echo(
                f"  Status: ✗ Invalid minimum ({n_imaginary} significant imaginary frequencies)",
            )

    # Show key frequencies
    if frequencies:
        # Show first few real frequencies and any imaginary ones
        real_freqs = [f for f in frequencies if f > 0]
        imaginary_freqs = [f for f in frequencies if f < 0]

        if real_freqs:
            lowest_real = real_freqs[:3]  # First 3 real frequencies
            freq_str = ", ".join(f"{f:.1f}" for f in lowest_real)
            click.echo(f"  Lowest frequencies: {freq_str} cm⁻¹")

        if imaginary_freqs:
            # Show all imaginary frequencies
            imag_str = ", ".join(f"{f:.1f}" for f in imaginary_freqs)
            click.echo(f"  Imaginary frequencies: {imag_str} cm⁻¹")


def save_results_json(results: dict[str, Any], output_path: str) -> None:
    """Save complete results dictionary to JSON file.

    Parameters
    ----------
    results : dict[str, Any]
        Complete results dictionary from Explorer.run()
    output_path : str
        Path to save JSON file (will replace .xyz extension with .json)

    """
    # Convert output path to JSON
    json_path = os.path.splitext(output_path)[0] + ".json"

    # Create a serializable version of results
    serializable_results = {}
    for key, value in results.items():
        if key == "optimized_atoms":
            # Skip Atoms objects - they're saved separately as XYZ
            continue
        if key == "frequency_analysis" and value:
            # Ensure frequency analysis is serializable
            freq_data = dict(value)
            # Convert numpy arrays to lists
            for k, v in freq_data.items():
                if hasattr(v, "tolist"):
                    freq_data[k] = v.tolist()
            serializable_results[key] = freq_data
        else:
            # Try to make other values serializable
            try:
                if hasattr(value, "tolist"):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            except (TypeError, AttributeError):
                # Skip non-serializable values
                continue

    # Save to JSON
    with open(json_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    click.echo(f"Results saved to: {json_path}")


__all__ = [
    "load_atoms_from_xyz",
    "parse_kv_pairs",
    "print_frequency_summary",
    "save_results_json",
    "write_atoms",
]
