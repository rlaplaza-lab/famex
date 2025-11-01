"""Helper functions for local strategies.

This module contains utility functions used by local optimization strategies.
"""

from typing import Any

from ase import Atoms

from qme.utils.logging import get_qme_logger

logger = get_qme_logger(__name__)


def validate_ts_structure(atoms: Atoms, explorer: Any, threshold: float = 50.0) -> dict[str, Any]:
    """Validate that structure is a transition state via frequency analysis.

    Parameters
    ----------
    atoms : Atoms
        The structure to validate
    explorer : Explorer
        Explorer instance for calculator access
    threshold : float, default=50.0
        Minimum frequency magnitude in cm^-1 to consider significant

    Returns:
    -------
    dict[str, Any]
        Validation results dictionary from FrequencyAnalysis.is_transition_state()

    """
    from qme.analysis.frequency import FrequencyAnalysis

    # Ensure calculator is attached
    if getattr(atoms, "calc", None) is None:
        explorer._create_and_attach_calculator(atoms)

    # Run frequency analysis
    freq_analysis = FrequencyAnalysis(atoms=atoms, calculator=atoms.calc, verbose=0)
    freq_analysis.calculate_hessian(method="auto")
    freq_analysis.diagonalize_hessian()

    # Check if structure is a TS
    validation_result = freq_analysis.is_transition_state(threshold=threshold)

    # Log warning if validation fails
    if not validation_result["is_transition_state"]:
        logger.warning(
            "TS validation failed: %s. Expected exactly 1 imaginary frequency, "
            "found %d imaginary frequencies.",
            validation_result["assessment"],
            validation_result["n_imaginary_frequencies"],
        )
    else:
        logger.info("TS validation passed: structure has exactly 1 imaginary frequency")

    return validation_result


def _validate_ts_optimization_setup(backend: str, optimizer_name: str) -> None:
    """Validate that TS optimization is using appropriate calculator and optimizer.

    This function hardcodes restrictions to prevent using mock calculators or
    unsuitable optimizers for transition state optimization tasks.

    Parameters
    ----------
    backend : str
        The calculator backend being used
    optimizer_name : str
        The optimizer being used

    Raises:
    ------
    ValueError
        If the setup is unsuitable for TS optimization

    """
    # Hardcoded restrictions for TS optimization
    FORBIDDEN_BACKENDS_FOR_TS = {"mock"}
    FORBIDDEN_OPTIMIZERS_FOR_TS = {"lbfgs", "l-bfgs", "l_bfgs", "bfgs", "fire"}

    if backend.lower() in FORBIDDEN_BACKENDS_FOR_TS:
        msg = (
            f"Backend '{backend}' is not suitable for transition state optimization. "
            f"Use a real ML potential backend (uma, aimnet2, mace, so3lr) instead."
        )
        raise ValueError(
            msg,
        )

    normalized_name = optimizer_name.lower()
    if normalized_name in FORBIDDEN_OPTIMIZERS_FOR_TS:
        msg = (
            f"Optimizer '{optimizer_name}' is not suitable for transition state "
            "optimization. Use 'sella' or 'trust-krylov-ts' for TS searches."
        )
        raise ValueError(
            msg,
        )


def _get_local_optimizer_class(name: str) -> type[Any]:
    """Map a short name to an ASE optimizer class or SELLA's Sella.

    SELLA is preferred when requested and is now a core dependency.
    All optimizers now support verbosity control through QME's logging system.
    """
    name = (name or "").lower()

    if name == "sella":
        from qme.optimizers.ase_wrappers import VerboseSella

        return VerboseSella

    # SciPy Hessian-based optimizers
    if name in ("trust-krylov", "trustkrylov", "trust_krylov"):
        from qme.optimizers.scipy_optimizers import TrustKrylov

        return TrustKrylov
    if name in (
        "trust-krylov-ts",
        "trustkrylovts",
        "trust_krylov_ts",
        "trust-krylov-transition",
    ):
        from qme.optimizers.scipy_optimizers import TrustKrylovTS

        return TrustKrylovTS
    if name in ("trust-ncg", "trustncg", "trust_ncg"):
        from qme.optimizers.scipy_optimizers import TrustNCG

        return TrustNCG
    if name in ("trust-exact", "trustexact", "trust_exact"):
        from qme.optimizers.scipy_optimizers import TrustExact

        return TrustExact
    if name in ("newton-cg", "newtoncg", "newton_cg"):
        from qme.optimizers.scipy_optimizers import NewtonCG

        return NewtonCG

    try:
        if name in ("lbfgs", "l-bfgs", "l_bfgs"):
            from qme.optimizers.ase_wrappers import VerboseLBFGS

            return VerboseLBFGS
        if name in ("bfgs",):
            from qme.optimizers.ase_wrappers import VerboseBFGS

            return VerboseBFGS
        if name in ("fire",):
            from qme.optimizers.ase_wrappers import VerboseFIRE

            return VerboseFIRE
    except Exception as e:  # pragma: no cover - ASE optional in some envs
        msg = f"Requested optimizer '{name}' is not available: {e}"
        raise ImportError(msg)

    msg = f"Unknown optimizer name: {name}"
    raise ValueError(msg)
