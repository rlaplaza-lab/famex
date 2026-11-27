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
DEFAULT_UMA_TASK = "omol"
# oc22/oc25 need UMA-1.2+ checkpoints; fine-tunes use model_name (not allowlisted).
UMA_TASKS = ("omol", "omat", "omc", "oc20", "oc25", "odac", "oc22")
DEFAULT_MACE_MODEL = "mace-omol-0"
DEFAULT_PET_MODEL = "pet-mad-s"
DEFAULT_TBLITE_MODEL = "GFN2-xTB"
DEFAULT_PYSCF_MODEL = "b3lyp-d3bj/def2-svp"
FAIRCHEM_INSTALL = "fairchem-core>=2.25.0"

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

# Dependency requirements for backend availability (dispersion is optional for PySCF).
BACKEND_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    BACKEND_MOCK: (),
    BACKEND_AIMNET2: ("torch",),
    BACKEND_UMA: ("fairchem", "torch"),
    BACKEND_SO3LR: ("so3lr",),
    BACKEND_MACE: ("mace", "torch"),
    BACKEND_ORB: ("orb_models", "torch"),
    BACKEND_TBLITE: ("tblite",),
    BACKEND_PET: ("upet", "torch"),
    BACKEND_PYSCF: ("pyscf",),
}

BACKEND_PACKAGE_NAMES: dict[str, tuple[str, ...]] = {
    BACKEND_AIMNET2: ("torch",),
    BACKEND_UMA: ("fairchem-core", "torch"),
    BACKEND_SO3LR: ("so3lr",),
    BACKEND_MACE: ("mace-torch", "torch"),
    BACKEND_ORB: ("orb-models>=0.7.0", "torch"),
    BACKEND_TBLITE: ("tblite",),
    BACKEND_PET: ("upet", "torch"),
    BACKEND_PYSCF: ("pyscf",),
}

_FAIRCHEM_PACKAGE = FAIRCHEM_INSTALL.split(">=")[0]

# Pip install argument after ``pip install `` (package fragment).
INSTALL_COMMANDS: dict[str, str] = {
    "torch": "torch",
    "sella": "sella",
    BACKEND_AIMNET2: "torch",
    "fairchem": _FAIRCHEM_PACKAGE,
    BACKEND_UMA: _FAIRCHEM_PACKAGE,
    BACKEND_MACE: "mace-torch",
    BACKEND_SO3LR: "so3lr",
    "orb_models": "orb-models",
    BACKEND_ORB: '"orb-models>=0.7.0"',
    BACKEND_TBLITE: "tblite",
    "upet": "upet",
    BACKEND_PET: "upet",
    "pyscf": "famex[pyscf]",
    BACKEND_PYSCF: "famex[pyscf]",
    "pyscf_dispersion": "pyscf-dispersion",
}

# Full shell commands when a pip package name alone is insufficient.
BACKEND_FULL_INSTALL_COMMANDS: dict[str, str] = {
    BACKEND_SO3LR: (
        "git clone https://github.com/general-molecular-simulations/so3lr.git"
        " && cd so3lr && pip install ."
    ),
}


def install_package_spec(name: str) -> str:
    """Return pip package spec for ``name``."""
    if name in INSTALL_COMMANDS:
        return INSTALL_COMMANDS[name]
    return INSTALL_COMMANDS.get(name.lower(), name.lower())


def format_install_command(name: str) -> str:
    """Format a user-facing install command for a backend or dependency."""
    if name in BACKEND_FULL_INSTALL_COMMANDS:
        return BACKEND_FULL_INSTALL_COMMANDS[name]
    return f"pip install {install_package_spec(name)}"
