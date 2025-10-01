"""Calculator creation and management."""

from typing import Optional

from qme.calculator_registry import calculator_registry
from qme.potentials.calculator_cache import cache_calculator, get_cached_calculator


def create_calculator(
    backend: str,
    model_name: Optional[str],
    model_path: Optional[str],
    device: Optional[str],
    default_charge: int,
    default_spin: int,
    charge: Optional[int] = None,
    mult: Optional[int] = None,
    use_cache: bool = True,
):
    """Create calculator based on backend using the registry.

    New parameters `charge` and `mult` (optional) are forwarded to
    backends that accept explicit molecular charge / multiplicity
    constructor arguments (for example AIMNet2). Older backends that
    expect `default_charge` / `default_spin` will continue to receive
    those values.
    """

    # Use the centralized calculator registry. Forward both naming
    # conventions so different backends can pick what they expect.
    factory_kwargs = {
        "default_charge": default_charge,
        "default_spin": default_spin,
    }

    # If explicit charge/multiplicity were provided (e.g. from a
    # Geometry object), forward them under the common names used by
    # some backends.
    if charge is not None:
        factory_kwargs["charge"] = charge
    if mult is not None:
        factory_kwargs["mult"] = mult

    # Try to get cached calculator first (exclude SO3LR due to state issues)
    if use_cache and backend.lower() != "so3lr":
        try:
            cached_calc = get_cached_calculator(
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                device=device,
                **factory_kwargs,
            )

            if cached_calc is not None:
                return cached_calc
        except ImportError:
            # Calculator cache not available, continue without caching
            pass

    # Create new calculator
    calculator = calculator_registry.create_calculator(
        backend=backend,
        model_name=model_name,
        model_path=model_path,
        device=device,
        **factory_kwargs,
    )

    # Cache the calculator if caching is enabled (exclude SO3LR due to state issues)
    if use_cache and backend.lower() != "so3lr":
        try:
            cache_calculator(
                calculator=calculator,
                backend=backend,
                model_name=model_name,
                model_path=model_path,
                device=device,
                **factory_kwargs,
            )
        except ImportError:
            # Calculator cache not available, continue without caching
            pass

    return calculator
