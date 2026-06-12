"""Backend name constants for FAMEX.

This module provides centralized backend name constants to ensure consistency
across the codebase and avoid magic strings.
"""

# Backend name constants
BACKEND_MOCK = "mock"
BACKEND_AIMNET2 = "aimnet2"
BACKEND_UMA = "uma"
BACKEND_MACE = "mace"
BACKEND_SO3LR = "so3lr"
BACKEND_ORB = "orb"
BACKEND_TBLITE = "tblite"

DEFAULT_UMA_MODEL = "uma-s-1p2"
FAIRCHEM_INSTALL = "fairchem-core>=2.21.0"

# Backend categorization constants
ALL_BACKENDS = [
    BACKEND_MOCK,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
]

ML_BACKENDS = [
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
]

REGULAR_BACKENDS = [
    BACKEND_MOCK,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
]
