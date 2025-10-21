"""Multistructure strategy modules for multi-structure optimization.

This module contains strategies that require multiple input structures:
- ts:interpolate - TS guess from interpolated path with local TS refinement
- path:interpolate - Generate interpolated path only (no optimization)
- path:neb - NEB path optimization with geodesic interpolation
- path:cineb - CI-NEB path optimization with geodesic interpolation
- ts:growing_string - Growing string method for TS search
"""

import qme.core.strategies.multistructure.cineb  # noqa: F401
import qme.core.strategies.multistructure.growing_string  # noqa: F401
import qme.core.strategies.multistructure.neb  # noqa: F401

# Import multistructure strategies to register them
import qme.core.strategies.multistructure.neb_optimizer  # noqa: F401
import qme.core.strategies.multistructure.path_interpolate  # noqa: F401
import qme.core.strategies.multistructure.ts_interpolate  # noqa: F401
