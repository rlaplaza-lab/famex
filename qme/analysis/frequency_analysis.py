"""Frequency and thermodynamics analysis."""

from typing import Any, Dict, List, Optional

from ase import Atoms

from qme.analysis.frequency import FrequencyAnalysis


def calculate_frequencies(
    self,
    atoms: Optional[Atoms] = None,
    delta: float = 0.01,
    method: str = "auto",
    nfree: Optional[int] = None,
    temperature: float = 298.15,
    save_hessian: bool = True,
    indices: Optional[List[int]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Calculate vibrational frequencies and normal modes.

    Parameters:
    -----------
    atoms : Atoms, optional
        Structure to analyze. Uses self.atoms if None.
    delta : float
        Displacement for finite differences (Å)
    method : str
        Hessian calculation method: 'auto', 'direct', or 'finite_differences'
    nfree : int, optional
        Number of degrees of freedom to remove. Auto-determined if None.
    temperature : float
        Temperature for thermodynamic properties (K)
    save_hessian : bool
        Whether to save Hessian matrix in results
    indices : List[int], optional
        Indices of atoms to include. All atoms if None.
    **kwargs
        Additional arguments for FrequencyAnalysis

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - frequencies: List of frequencies in cm^-1
        - normal_modes: Normal mode vectors
        - hessian: Hessian matrix (if save_hessian=True)
        - is_ts: Boolean indicating if structure is transition state
        - zero_point_energy: Zero-point vibrational energy
        - thermodynamic_properties: Dict with entropy, heat capacity, etc.
    """

    if atoms is None:
        if self.atoms is None:
            raise ValueError("No structure loaded. Use load_structure() first.")
        atoms = self.atoms.copy()
    else:
        atoms = atoms.copy()

    atoms.calc = self.calculator

    # Initialize frequency analysis
    freq_analysis = FrequencyAnalysis(
        atoms=atoms,
        calculator=self.calculator,
        delta=delta,
        nfree=nfree,
        indices=indices,
    )

    # Calculate Hessian and frequencies
    print(f"Calculating frequencies using {method} method...")
    hessian = freq_analysis.calculate_hessian(method=method)
    frequencies, normal_modes = freq_analysis.diagonalize_hessian()

    # Get vibrational frequencies (excluding trans/rot modes)
    vib_frequencies = freq_analysis.get_frequencies()
    vib_normal_modes = freq_analysis.get_normal_modes()

    # Calculate thermodynamic properties
    thermo_props = freq_analysis.get_thermodynamic_properties(temperature)

    # Transition state verification
    ts_analysis = freq_analysis.is_transition_state()

    # Prepare results
    results = {
        "frequencies": vib_frequencies.tolist(),
        "all_frequencies": frequencies.tolist(),  # Including trans/rot modes
        "normal_modes": vib_normal_modes.tolist(),
        "zero_point_energy": freq_analysis.get_zero_point_energy(),
        "thermodynamic_properties": thermo_props,
        "ts_analysis": ts_analysis,
        "is_ts": ts_analysis["is_transition_state"],
        "method_used": method,
        "delta": delta,
        "temperature": temperature,
        "n_atoms": len(atoms),
        "indices": indices if indices is not None else list(range(len(atoms))),
    }

    if save_hessian:
        results["hessian"] = hessian.tolist()

    self.results["frequency_analysis"] = results
    print(
        f"Frequency analysis completed. Found {len(vib_frequencies)} vibrational modes."
    )

    return results


def verify_transition_state(
    self, atoms: Optional[Atoms] = None, freq_threshold: float = 50.0, **freq_kwargs
) -> Dict[str, Any]:
    """
    Verify that a structure is a transition state by checking frequencies.

    Parameters:
    -----------
    atoms : Atoms, optional
        Structure to verify. Uses self.atoms if None.
    freq_threshold : float
        Minimum frequency magnitude in cm^-1 to consider significant
    **freq_kwargs
        Additional arguments passed to calculate_frequencies

    Returns:
    --------
    Dict[str, Any]
        Dictionary with verification results including number of imaginary
        frequencies and their values.
    """

    if atoms is None:
        if self.atoms is None:
            raise ValueError("No structure loaded. Use load_structure() first.")
        atoms = self.atoms.copy()
    else:
        atoms = atoms.copy()

    atoms.calc = self.calculator

    # Get frequency analysis parameters
    delta = freq_kwargs.get("delta", 0.01)
    method = freq_kwargs.get("method", "auto")
    nfree = freq_kwargs.get("nfree", None)
    indices = freq_kwargs.get("indices", None)

    print("Verifying transition state by frequency analysis...")

    # Initialize frequency analysis
    freq_analysis = FrequencyAnalysis(
        atoms=atoms,
        calculator=self.calculator,
        delta=delta,
        nfree=nfree,
        indices=indices,
    )

    # Calculate frequencies
    freq_analysis.calculate_hessian(method=method)
    freq_analysis.diagonalize_hessian()

    # Verify transition state
    ts_results = freq_analysis.is_transition_state(threshold=freq_threshold)

    # Add summary information
    ts_results.update(
        {
            "structure_verified": True,
            "method_used": method,
            "freq_threshold": freq_threshold,
            "verification_summary": _format_ts_verification(ts_results),
        }
    )

    self.results["ts_verification"] = ts_results

    return ts_results


def _format_ts_verification(ts_results: Dict[str, Any]) -> str:
    """Format TS verification results as readable summary."""
    n_imag = ts_results["n_imaginary_frequencies"]
    assessment = ts_results["assessment"]

    summary = f"Structure Assessment: {assessment}\n"
    summary += f"Number of imaginary frequencies: {n_imag}\n"

    if n_imag > 0:
        imag_freqs = ts_results["imaginary_frequencies"]
        freq_str = ", ".join([f"{f:.1f}" for f in imag_freqs])
        summary += f"Imaginary frequencies: {freq_str} cm⁻¹\n"

    if ts_results["is_transition_state"]:
        summary += (
            "✓ Structure is a valid transition state (exactly one imaginary frequency)"
        )
    elif n_imag == 0:
        summary += "✓ Structure is a minimum (no imaginary frequencies)"
    else:
        summary += (
            f"⚠ Structure is a higher-order saddle point "
            f"({n_imag} imaginary frequencies)"
        )

    return summary


def calculate_reaction_thermodynamics(
    self,
    reactant_atoms: Atoms,
    product_atoms: Atoms,
    ts_atoms: Optional[Atoms] = None,
    temperature: float = 298.15,
    **freq_kwargs,
) -> Dict[str, Any]:
    """
    Calculate reaction thermodynamics including activation barriers,
    reaction energies, and rate constants.

    Parameters:
    -----------
    reactant_atoms : Atoms
        Optimized reactant structure
    product_atoms : Atoms
        Optimized product structure
    ts_atoms : Atoms, optional
        Optimized transition state structure
    temperature : float
        Temperature for analysis (K)
    **freq_kwargs
        Additional arguments for frequency calculations

    Returns:
    --------
    Dict[str, Any]
        Dictionary with reaction thermodynamics
    """

    print(f"Calculating reaction thermodynamics at {temperature} K...")

    # Ensure all structures have the calculator attached
    reactant_atoms = reactant_atoms.copy()
    product_atoms = product_atoms.copy()
    reactant_atoms.calc = self.calculator
    product_atoms.calc = self.calculator

    results = {
        "temperature": temperature,
        "has_transition_state": ts_atoms is not None,
    }

    # Calculate properties for reactant
    print("Analyzing reactant...")
    reactant_freq = FrequencyAnalysis(reactant_atoms, self.calculator, **freq_kwargs)
    reactant_freq.calculate_hessian()
    reactant_freq.diagonalize_hessian()

    reactant_energy = reactant_atoms.get_potential_energy()
    reactant_zpe = reactant_freq.get_zero_point_energy()
    reactant_thermo = reactant_freq.get_thermodynamic_properties(temperature)

    results["reactant"] = {
        "electronic_energy": reactant_energy,
        "zero_point_energy": reactant_zpe,
        "thermodynamic_properties": reactant_thermo,
        "total_energy": reactant_energy + reactant_zpe,
        "frequencies": reactant_freq.get_frequencies().tolist(),
    }

    # Calculate properties for product
    print("Analyzing product...")
    product_freq = FrequencyAnalysis(product_atoms, self.calculator, **freq_kwargs)
    product_freq.calculate_hessian()
    product_freq.diagonalize_hessian()

    product_energy = product_atoms.get_potential_energy()
    product_zpe = product_freq.get_zero_point_energy()
    product_thermo = product_freq.get_thermodynamic_properties(temperature)

    results["product"] = {
        "electronic_energy": product_energy,
        "zero_point_energy": product_zpe,
        "thermodynamic_properties": product_thermo,
        "total_energy": product_energy + product_zpe,
        "frequencies": product_freq.get_frequencies().tolist(),
    }

    # Calculate reaction energies
    results["reaction_energy"] = {
        "electronic": product_energy - reactant_energy,
        "zero_point_corrected": (product_energy + product_zpe)
        - (reactant_energy + reactant_zpe),
        "enthalpy": (
            product_thermo["internal_energy"] - reactant_thermo["internal_energy"]
        ),
        "free_energy": (
            (
                product_thermo["internal_energy"]
                - temperature * product_thermo["entropy"]
            )
            - (
                reactant_thermo["internal_energy"]
                - temperature * reactant_thermo["entropy"]
            )
        ),
    }

    # Calculate properties for transition state if provided
    if ts_atoms is not None:
        ts_atoms = ts_atoms.copy()
        ts_atoms.calc = self.calculator

        print("Analyzing transition state...")
        ts_freq = FrequencyAnalysis(ts_atoms, self.calculator, **freq_kwargs)
        ts_freq.calculate_hessian()
        ts_freq.diagonalize_hessian()

        ts_energy = ts_atoms.get_potential_energy()
        ts_zpe = ts_freq.get_zero_point_energy()
        ts_thermo = ts_freq.get_thermodynamic_properties(temperature)
        ts_verification = ts_freq.is_transition_state()

        results["transition_state"] = {
            "electronic_energy": ts_energy,
            "zero_point_energy": ts_zpe,
            "thermodynamic_properties": ts_thermo,
            "total_energy": ts_energy + ts_zpe,
            "frequencies": ts_freq.get_frequencies().tolist(),
            "is_valid_ts": ts_verification["is_transition_state"],
            "imaginary_frequency": (
                ts_verification["imaginary_frequencies"][0]
                if ts_verification["imaginary_frequencies"]
                else None
            ),
        }

        # Calculate activation barriers
        results["activation_energy"] = {
            "electronic": ts_energy - reactant_energy,
            "zero_point_corrected": (ts_energy + ts_zpe)
            - (reactant_energy + reactant_zpe),
            "enthalpy": ts_thermo["internal_energy"]
            - reactant_thermo["internal_energy"],
            "free_energy": (
                (ts_thermo["internal_energy"] - temperature * ts_thermo["entropy"])
                - (
                    reactant_thermo["internal_energy"]
                    - temperature * reactant_thermo["entropy"]
                )
            ),
        }

        # Estimate rate constant using transition state theory
        if ts_verification["is_transition_state"]:
            results["rate_constant"] = _calculate_rate_constant(
                results["activation_energy"]["free_energy"], temperature
            )

    self.results["reaction_thermodynamics"] = results
    print("Reaction thermodynamics analysis completed.")

    return results


def _calculate_rate_constant(
    delta_g_act: float,
    temperature: float,
) -> Dict[str, float]:
    """Calculate rate constant using transition state theory."""
    import math

    # Physical constants (CODATA 2018 values)
    kB = 8.617333262145e-5  # eV/K (Boltzmann constant)
    h = 4.135667696e-15  # eV·s (Planck constant)

    # TST rate constant: k = (kB*T/h) * exp(-ΔG‡/kB*T)
    prefactor = kB * temperature / h  # in s^-1
    exponential = math.exp(-delta_g_act / (kB * temperature))
    rate_constant = prefactor * exponential

    return {
        "rate_constant_s-1": rate_constant,
        "prefactor_s-1": prefactor,
        "activation_free_energy_eV": delta_g_act,
        "temperature_K": temperature,
    }
