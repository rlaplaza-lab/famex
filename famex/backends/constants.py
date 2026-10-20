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
BACKEND_PET = "pet"
BACKEND_PYSCF = "pyscf"

DEFAULT_UMA_MODEL = "uma-s-1p2"
DEFAULT_MACE_MODEL = "mace-omol-0"
DEFAULT_PET_MODEL = "pet-mad-s"
DEFAULT_TBLITE_MODEL = "GFN2-xTB"
DEFAULT_PYSCF_MODEL = "b3lyp-d3bj/def2-svp"
FAIRCHEM_INSTALL = "fairchem-core>=2.22.0"

# Backend categorization constants
ALL_BACKENDS = [
    BACKEND_MOCK,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
    BACKEND_PET,
    BACKEND_PYSCF,
]

ML_BACKENDS = [
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
    BACKEND_PET,
]

REGULAR_BACKENDS = [
    BACKEND_MOCK,
    BACKEND_AIMNET2,
    BACKEND_MACE,
    BACKEND_UMA,
    BACKEND_SO3LR,
    BACKEND_ORB,
    BACKEND_TBLITE,
    BACKEND_PET,
    BACKEND_PYSCF,
]
