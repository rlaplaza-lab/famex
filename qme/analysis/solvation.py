"""Solvation corrections for solution-phase thermochemistry.

This module provides corrections for solution-phase entropy calculations
based on the accessible free space in solution.
"""

from __future__ import annotations

from ase import units

__all__ = ["SolvationHandler", "get_free_space", "SUPPORTED_SOLVENTS"]

# Physical constants from ASE units
AVOGADRO_CONSTANT = units._Nav  # 1/mol

# Supported solvents and their properties
# Based on Shakhnovich & Whitesides, J. Org. Chem. 1998, 63, 3821-3830
# molarity in mol/L, molecular volume in Angstrom^3
SUPPORTED_SOLVENTS = {
    "none": {"molarity": 1.0, "molecular_vol": 1.0},
    "H2O": {"molarity": 55.6, "molecular_vol": 27.944},
    "toluene": {"molarity": 9.4, "molecular_vol": 149.070},
    "DMF": {"molarity": 12.9, "molecular_vol": 77.442},
    "AcOH": {"molarity": 17.4, "molecular_vol": 86.10},
    "chloroform": {"molarity": 12.5, "molecular_vol": 97.0},
}


def get_free_space(solvent: str) -> float:
    """Get accessible free space in solution (mL per L).

    Calculates the free space in a liter of bulk solvent based on
    Shakhnovich & Whitesides (J. Org. Chem. 1998, 63, 3821-3830).
    This represents the volume not occupied by solvent molecules and
    accessible to a solute immersed in bulk solvent.

    Parameters
    ----------
    solvent : str
        Solvent name (must be in SUPPORTED_SOLVENTS)

    Returns:
    -------
    float
        Accessible free space in mL per L

    Raises:
    ------
    ValueError
        If solvent is not in supported solvents list

    Examples:
    --------
    >>> get_free_space("H2O")
    890.123...  # approximately
    >>> get_free_space("none")
    1000.0
    """
    if solvent not in SUPPORTED_SOLVENTS:
        supported = list(SUPPORTED_SOLVENTS.keys())
        msg = f"Unknown solvent: {solvent}. Supported solvents: {supported}"
        raise ValueError(msg)

    solvent_data = SUPPORTED_SOLVENTS[solvent]
    molarity = solvent_data["molarity"]
    molecular_vol = solvent_data["molecular_vol"]

    if solvent == "none":
        return 1000.0  # Gas phase

    # Calculate free space following Shakhnovich-Whitesides approach
    # GoodVibes formula: v_free = 8 * ((1E27 / (solv_molarity * N_A))^0.333333 - solv_volume^0.333333)^3
    # where volume is in Angstrom^3
    v_free = (
        8 * ((1e27 / (molarity * AVOGADRO_CONSTANT)) ** 0.333333 - molecular_vol**0.333333) ** 3
    )
    free_space_ml_per_l = v_free * molarity * AVOGADRO_CONSTANT * 1e-24

    return free_space_ml_per_l


class SolvationHandler:
    """Handles solvation corrections for thermodynamic calculations."""

    def __init__(
        self,
        solvent: str = "none",
        concentration: float = 1.0,
    ):
        """Initialize solvation handler.

        Parameters
        ----------
        solvent : str
            Solvent name (default: 'none' for gas phase)
        concentration : float
            Concentration in mol/L (default: 1.0 M)
        """
        self.solvent = solvent
        self.concentration = concentration
        self.free_space_ml_per_l = get_free_space(solvent)

    def is_gas_phase(self) -> bool:
        """Check if in gas phase.

        Returns:
        -------
        bool
            True if solvent is 'none' (gas phase)
        """
        return self.solvent == "none"

    def effective_concentration(self) -> float:
        """Calculate effective concentration accounting for free space.

        This is the concentration that would give the same number density
        in the accessible free space as the nominal concentration in the
        total solution volume.

        Returns:
        -------
        float
            Effective concentration in mol/L
        """
        if self.is_gas_phase():
            return self.concentration

        # Effective concentration = nominal concentration / (free_space / 1000)
        # This accounts for the reduced accessible volume in solution
        return self.concentration / (self.free_space_ml_per_l / 1000.0)

    def __repr__(self) -> str:
        """String representation."""
        phase = "gas" if self.is_gas_phase() else f"solution ({self.solvent})"
        return f"SolvationHandler({phase}, {self.concentration} M)"
