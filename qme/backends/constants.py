"""Backend name constants for QME.

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
BACKEND_TORCHSIM_MACE = "torchsim_mace"
BACKEND_TORCHSIM_UMA = "torchsim_uma"

# Backend categorization constants
ALL_BACKENDS = [
    BACKEND_MOCK,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
    BACKEND_TORCHSIM_MACE,
    BACKEND_TORCHSIM_UMA,
]

ML_BACKENDS = [
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
    BACKEND_TORCHSIM_MACE,
    BACKEND_TORCHSIM_UMA,
]

TORCHSIM_BACKENDS = [BACKEND_TORCHSIM_MACE, BACKEND_TORCHSIM_UMA]

REGULAR_BACKENDS = [
    BACKEND_MOCK,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
]
