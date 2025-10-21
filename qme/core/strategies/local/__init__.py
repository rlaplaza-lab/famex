"""Local strategy modules for single-structure optimization.

This module contains strategies that work on individual structures:
- minima:local - Direct local minima optimization
- ts:local - Direct local transition state optimization
- path:irc - IRC path calculation from transition state
"""

# Import local strategies to register them
import qme.core.strategies.local.helpers  # noqa: F401
import qme.core.strategies.local.irc  # noqa: F401
import qme.core.strategies.local.minima  # noqa: F401
import qme.core.strategies.local.ts  # noqa: F401
