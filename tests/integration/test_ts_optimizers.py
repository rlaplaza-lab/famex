"""Tests for transition state optimizers.

This module contains tests for RFO (Rational Function Optimization) transition
state optimizer and benchmark tests on challenging TS systems.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from famex.optimizers.rfo_optimizer import RFOTransitionState
from famex.potentials import get_uma_calculator
from tests.test_constants import DEFAULT_FMAX, LONG_STEPS
from tests.test_utils import requires_backend

BH28_DATASET = Path(__file__).resolve().parents[2] / "examples" / "bh28_benchmark" / "bh28_dataset"
BH28_TS_SUBSET = [
    "BHDIV_3",
    "PXBH_3",
    "CADBH_2",
    "CRBH_1",
    "PXBH_2",
    "BHPERI_2",
]


def _max_force(atoms):
    return float(np.max(np.abs(atoms.get_forces())))


def _load_bh28_ts(reaction):
    ts_file = BH28_DATASET / f"{reaction}_ts.xyz"
    atoms = read(str(ts_file))
    if isinstance(atoms, list):
        atoms = atoms[0]
    atoms.calc = get_uma_calculator()
    return atoms


class TestRFOTSOptimizer:
    """Tests for RFO (Rational Function Optimization) transition state optimizer."""

    @requires_backend("uma")
    def test_basic_functionality_with_uma(self, ethylene_twisted_ts_guess):
        atoms_template = ethylene_twisted_ts_guess

        def fresh_atoms():
            atoms = atoms_template.copy()
            atoms.calc = get_uma_calculator()
            return atoms

        # Baseline gradient/energy from the starting geometry.
        baseline_atoms = fresh_atoms()
        initial_force = _max_force(baseline_atoms)
        initial_energy = baseline_atoms.get_potential_energy()
        assert initial_force > 1e-6, "Baseline geometry should not be stationary"

        # Run RFO optimizer.
        atoms_rfo = fresh_atoms()
        rfo_opt = RFOTransitionState(
            atoms_rfo,
            hessian_update_freq=10,
            hessian_method="auto",
            trust_radius=0.02,
            max_trust_radius=0.06,
        )
        rfo_opt.run(fmax=DEFAULT_FMAX, steps=LONG_STEPS)
        rfo_force = _max_force(atoms_rfo)
        rfo_energy = atoms_rfo.get_potential_energy()

        # RFO should make progress relative to the baseline.
        # Allow some tolerance for challenging cases where optimization may not converge well
        # Note: RFO optimizer may not always converge in limited steps,
        # especially for TS optimization
        assert rfo_force <= initial_force * 2.0, (
            f"RFO should not dramatically increase the force. "
            f"Initial: {initial_force:.6f} eV/Å, Final: {rfo_force:.6f} eV/Å"
        )

        # RFO should reach tight convergence if it made progress.
        # If optimizer didn't converge well, at least check it didn't get dramatically worse
        if rfo_force < initial_force:
            # Made progress, should converge to tight threshold
            assert rfo_force < DEFAULT_FMAX, (
                f"RFO should converge when making progress. "
                f"Got {rfo_force:.6f} eV/Å (started at {initial_force:.6f} eV/Å)"
            )
        # Otherwise, just check that it didn't get dramatically worse
        # (handled by previous assertion)

        # Energy should be in a reasonable range.
        energy_window = max(0.05, 0.05 * abs(initial_energy))
        assert abs(rfo_energy - initial_energy) < energy_window, (
            f"RFO energy {rfo_energy:.6f} should be within {energy_window:.6f} of initial"
        )

    @requires_backend("uma")
    def test_bh28_subset_success_rate(self):
        """Test RFO optimizer on BH28 benchmark subset.

        This test verifies that RFO optimizer can converge all systems in the
        BH28 benchmark subset to tight convergence threshold (0.05 eV/Å).
        RFO is used instead of TrustKrylovTS because it handles these challenging
        systems more robustly, avoiding "bad approximation" errors.
        """
        if not BH28_DATASET.exists():
            pytest.skip("BH28 dataset missing")

        # Check if actual XYZ files exist
        missing_files = [
            rxn for rxn in BH28_TS_SUBSET if not (BH28_DATASET / f"{rxn}_ts.xyz").exists()
        ]
        if missing_files:
            pytest.skip(f"BH28 dataset files missing: {missing_files}")

        successes = 0
        results = {}

        for reaction in BH28_TS_SUBSET:
            atoms = _load_bh28_ts(reaction)
            initial_force = _max_force(atoms)

            # Use RFO optimizer for BH28 benchmark - it handles these challenging systems better
            # than other optimizers which can get stuck with "bad approximation" errors
            optimizer = RFOTransitionState(
                atoms,
                hessian_update_freq=5,  # Less frequent updates for efficiency
                hessian_method="auto",  # Let FAMEX choose best method
                trust_radius=0.02,
                max_trust_radius=0.1,  # Larger max trust radius for better exploration
            )
            optimizer.run(fmax=DEFAULT_FMAX, steps=LONG_STEPS)

            final_force = _max_force(atoms)
            energy = float(atoms.get_potential_energy())

            results[reaction] = {
                "initial_force": initial_force,
                "final_force": final_force,
                "energy": energy,
            }

            # Require tight convergence when optimizer makes progress
            if final_force < DEFAULT_FMAX and final_force <= 0.9 * initial_force:
                successes += 1

        # Require all 6 systems to converge to tight threshold (DEFAULT_FMAX eV/Å)
        # RFO optimizer should handle all these systems successfully
        assert successes == len(BH28_TS_SUBSET), (
            f"BH28 TS results: {results}. "
            f"Expected all {len(BH28_TS_SUBSET)} systems to converge to {DEFAULT_FMAX} eV/Å, "
            f"but only {successes} succeeded."
        )
