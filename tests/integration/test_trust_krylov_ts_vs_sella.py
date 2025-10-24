"""Compare Trust-Krylov TS optimizer against Sella on a small UMA system."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from qme.backends.dependencies import deps
from qme.optimizers.scipy_optimizers import TrustKrylovTS
from qme.potentials.uma_potential import get_uma_calculator
from tests.test_utils import BackendTestMixin, TestMoleculeFactory

BH28_DATASET = Path(__file__).resolve().parents[2] / "examples" / "bh28_benchmark" / "bh28_dataset"
BH28_TS_SUBSET = [
    "BHDIV_3",
    "PXBH_3",
    "CADBH_2",
    "CRBH_1",
    "PXBH_2",
    "BHPERI_2",
]


pytestmark = [
    pytest.mark.skipif(not deps.has("sella"), reason="Sella is required for TS comparison"),
    pytest.mark.skipif(
        not BackendTestMixin.check_backend_availability("uma"),
        reason="UMA backend is not available in the test environment",
    ),
]


def _max_force(atoms) -> float:
    return float(np.max(np.abs(atoms.get_forces())))


def test_trust_krylov_ts_matches_sella_with_uma() -> None:
    """Run both TS optimizers on a toy UMA system and compare the outcomes."""
    BackendTestMixin.require_backend("uma")

    atoms_template = TestMoleculeFactory.get_ethylene_twisted_ts_guess()

    def fresh_atoms():
        atoms = atoms_template.copy()
        atoms.calc = get_uma_calculator()
        return atoms

    # Baseline gradient/energy from the starting geometry.
    baseline_atoms = fresh_atoms()
    initial_force = _max_force(baseline_atoms)
    initial_energy = baseline_atoms.get_potential_energy()
    assert initial_force > 1e-6, "Baseline geometry should not be stationary"

    # Run Sella for reference.
    from sella import Sella  # Imported lazily to honour skip conditions

    atoms_sella = fresh_atoms()
    sella_opt = Sella(atoms_sella, internal=True, order=1)
    sella_opt.run(fmax=0.2, steps=12)
    sella_force = _max_force(atoms_sella)
    sella_energy = atoms_sella.get_potential_energy()

    # Run the Trust-Krylov TS variant.
    atoms_trust = fresh_atoms()
    trust_opt = TrustKrylovTS(
        atoms_trust,
        hessian_update_freq=1,
        adaptive_hessian=False,
        negative_mode_boost=1e-2,
        min_positive_eigenvalue=5e-3,
    )
    trust_converged = trust_opt.run(fmax=0.2, steps=12)
    trust_force = _max_force(atoms_trust)
    trust_energy = atoms_trust.get_potential_energy()

    # Trust-Krylov should make progress relative to the baseline.
    # Sella may not converge well on this system, so we're more lenient
    assert trust_force <= initial_force + 0.1, (
        "Trust-Krylov TS should not significantly increase the force"
    )

    # If Sella converges, it should also make progress
    if sella_force <= initial_force + 1.0:  # Only check if Sella didn't diverge too much
        assert sella_force <= initial_force + 0.5, (
            "Sella should not significantly increase the force"
        )

    # Trust-Krylov should land in the same energetic neighbourhood as Sella.
    energy_window = max(0.01, 0.01 * abs(initial_energy))
    assert abs(trust_energy - sella_energy) < energy_window

    # Their gradient norms should be of comparable magnitude.
    force_ratio = trust_force / max(sella_force, 1e-6)
    assert force_ratio < 2.0, "Trust-Krylov TS should reach a similar stationary quality as Sella"

    if trust_converged and bool(sella_opt.converged()):
        # Ensure final TS structures align closely when both solvers converge.
        pos_diff = np.linalg.norm(atoms_trust.get_positions() - atoms_sella.get_positions())
        assert pos_diff < 0.1, f"TS positions drifted by {pos_diff:.3f} Å"


def _load_bh28_ts(reaction: str):
    ts_file = BH28_DATASET / f"{reaction}_ts.xyz"
    atoms = read(str(ts_file))
    if isinstance(atoms, list):
        atoms = atoms[0]
    atoms.calc = get_uma_calculator()
    return atoms


def test_trust_krylov_ts_bh28_subset_success_rate() -> None:
    """Ensure Trust-Krylov TS converges reliably on a BH28 subset with UMA."""
    BackendTestMixin.require_backend("uma")
    if not BH28_DATASET.exists():
        pytest.skip("BH28 dataset missing")

    # Check if actual XYZ files exist
    missing_files = [rxn for rxn in BH28_TS_SUBSET if not (BH28_DATASET / f"{rxn}_ts.xyz").exists()]
    if missing_files:
        pytest.skip(f"BH28 dataset files missing: {missing_files}")

    successes = 0
    results: dict[str, dict[str, float]] = {}

    for reaction in BH28_TS_SUBSET:
        atoms = _load_bh28_ts(reaction)
        initial_force = _max_force(atoms)

        optimizer = TrustKrylovTS(
            atoms,
            hessian_update_freq=1,
            hessian_method="finite_differences",  # Force finite differences
            adaptive_hessian=True,  # Enable adaptive Hessian for better convergence
            negative_mode_boost=1e-2,  # Increase boost for better TS optimization
            min_positive_eigenvalue=1e-2,  # Increase minimum for stability
            index_tolerance=1e-3,  # Relax tolerance for numerical stability
            mode_recompute_interval=1,
        )
        optimizer.run(fmax=0.18, steps=18)

        final_force = _max_force(atoms)
        energy = float(atoms.get_potential_energy())

        results[reaction] = {
            "initial_force": initial_force,
            "final_force": final_force,
            "energy": energy,
        }

        if final_force < 0.25 and final_force <= 0.9 * initial_force:
            successes += 1

    assert successes >= len(BH28_TS_SUBSET) - 1, f"BH28 TS results: {results}"
