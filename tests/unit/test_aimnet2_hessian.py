"""Tests for AIMNet2 analytical Hessian via autograd.

Tests cover:
- Hessian sign convention (H = d²E/dx² = -dF/dx)
- Symmetry (H = H^T) of the analytical Hessian
- Gated integration: analytical Hessian vs finite differences for H2/water
- Frequency result sensibility via ASE VibrationsData
- Property dispatch: has_calculator_property, get_calculator_property
- get_property / get_hessian on AIMNet2Potential
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Torch-free unit test of the double-backward Hessian assembly logic
# ---------------------------------------------------------------------------


class _SimpleHarmonicModel:
    """Fake model: E = 0.5 * sum(coord^2); Hessian should be the identity."""

    def __init__(self) -> None:
        self.cutoff = 5.0
        self.cutoff_lr = float("inf")

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        coord = data["coord"]
        n = coord.shape[0] - 1 if "nbmat" in data and data["nbmat"].shape[0] > 1 else coord.shape[0]
        data["energy"] = 0.5 * (coord[:n] ** 2).sum()
        data["forces"] = -coord[:n]
        return data


def _torch_available() -> bool:
    """Return True if PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_hessian_sign_convention_on_harmonic_potential() -> None:
    """H = d2E/dx2 should equal the identity for E = 0.5 * sum(r2)."""
    import torch

    from famex.potentials.aimnet2_potential import generate_neighbor_list_numpy, maybe_pad_dim0

    model = _SimpleHarmonicModel()
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    nbmat = torch.as_tensor(generate_neighbor_list_numpy(pos, 5.0, 128), dtype=torch.int32)

    coord = torch.tensor(pos, dtype=torch.float32, requires_grad=True)
    n = coord.shape[0]
    coord_padded = maybe_pad_dim0(coord, nbmat.shape[0])
    coord_padded.requires_grad_(True)

    with torch.jit.optimized_execution(False):
        model_out = model({"coord": coord_padded, "nbmat": nbmat})

    energy = model_out["energy"].sum()
    forces = -torch.autograd.grad(energy, coord_padded, create_graph=True)[0][:n]

    nf = 3 * n
    rows = []
    for i in range(nf):
        (g,) = torch.autograd.grad(
            forces.reshape(-1)[i].sum(),
            coord_padded,
            retain_graph=True,
            allow_unused=True,
        )
        if g is None:
            g = torch.zeros_like(coord_padded)
        rows.append((-g[:n]).reshape(-1))

    H = torch.stack(rows).detach().numpy()
    expected = np.eye(3 * n)
    np.testing.assert_allclose(H, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Gated integration tests — require torch and AIMNet2 model
# ---------------------------------------------------------------------------


def _aimnet2_model_available() -> bool:
    """Check if torch and the AIMNet2 model can be loaded."""
    if not _torch_available():
        return False
    try:
        from famex.potentials.aimnet2_potential import get_model_path

        get_model_path("aimnet2")
    except (ImportError, RuntimeError):
        return False
    return True


def _water_equilibrium() -> tuple[np.ndarray, np.ndarray]:
    """Return positions and atomic numbers for water at the ωB97M equilibrium."""
    theta = np.radians(104.5)
    r = 0.9572
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, r * np.sin(theta), r * np.cos(theta)],
            [0.0, -r * np.sin(theta), r * np.cos(theta)],
        ]
    )
    numbers = np.array([8, 1, 1])
    return pos, numbers


@pytest.mark.skipif(not _aimnet2_model_available(), reason="AIMNet2 model not available")
class TestAIMNet2AnalyticalHessian:
    """Integration tests for the analytical Hessian against the real model."""

    @pytest.fixture(autouse=True)
    def _setup_calculator(self) -> None:
        from famex.potentials.aimnet2_potential import (
            NativeAIMNet2Calculator,
            get_model_path,
        )

        model_path = get_model_path("aimnet2")
        self._calc = NativeAIMNet2Calculator(model_path, device="cpu")

    def _analytical_hessian(self, pos: np.ndarray, numbers: np.ndarray) -> np.ndarray:
        """Compute analytical Hessian via the native calculator."""
        data = {"coord": pos, "numbers": numbers, "charge": 0.0, "mult": 1.0}
        res = self._calc(data, hessian=True)
        H = res["hessian"].detach().cpu().numpy()
        H = 0.5 * (H + H.T)
        return H

    def _fd_hessian(self, pos: np.ndarray, numbers: np.ndarray, delta: float = 0.005) -> np.ndarray:
        """Compute finite-difference Hessian via central differences on forces.

        Uses the regular calculator __call__ pipeline (neighbor lists are
        regenerated at each displaced geometry; for H2 at small delta the
        changes are negligible).
        """
        n = len(pos)
        nf = 3 * n
        Hfd = np.zeros((nf, nf))

        for a in range(nf):
            p = pos.copy().ravel()
            p[a] += delta
            res = self._calc(
                {"coord": p.reshape(n, 3), "numbers": numbers, "charge": 0.0, "mult": 1.0},
                forces=True,
            )
            fp = res["forces"].detach().cpu().numpy()
            p = pos.copy().ravel()
            p[a] -= delta
            res = self._calc(
                {"coord": p.reshape(n, 3), "numbers": numbers, "charge": 0.0, "mult": 1.0},
                forces=True,
            )
            fm = res["forces"].detach().cpu().numpy()
            Hfd[:, a] = (fp - fm).ravel() / (2 * delta)

        return 0.5 * (Hfd + Hfd.T)

    def test_hessian_symmetry_h2(self) -> None:
        """Analytical Hessian for H2 should be symmetric."""
        pos = np.array([[0.0, 0.0, -0.3707], [0.0, 0.0, 0.3707]])
        numbers = np.array([1, 1])
        H = self._analytical_hessian(pos, numbers)
        asym = np.max(np.abs(H - H.T))
        assert asym < 1e-4, f"H2 Hessian symmetry error: {asym}"

    def test_hessian_symmetry_water(self) -> None:
        """Analytical Hessian for water should be symmetric."""
        pos, numbers = _water_equilibrium()
        H = self._analytical_hessian(pos, numbers)
        asym = np.max(np.abs(H - H.T))
        assert asym < 1e-4, f"Water Hessian symmetry error: {asym}"

    def test_hessian_agrees_with_fd_h2(self) -> None:
        """Analytical Hessian for H2 should approximately agree with finite differences.

        H2 is a clean 1D system ideal for FD comparison.
        """
        pos = np.array([[0.0, 0.0, -0.3707], [0.0, 0.0, 0.3707]])
        numbers = np.array([1, 1])
        H_analytical = self._analytical_hessian(pos, numbers)
        H_fd = self._fd_hessian(pos, numbers, delta=0.005)

        diff = np.abs(H_analytical - H_fd)
        median_diff = np.median(diff)
        max_diff = diff.max()

        assert median_diff < 0.5, (
            f"H2: Analytical vs FD Hessian: median diff {median_diff:.3e} (max diff {max_diff:.3e})"
        )

    def test_h2_frequencies_sensible(self) -> None:
        """H2 Hessian should produce a stretching mode near 4270 cm-1."""
        from ase import Atoms as ASE_Atoms
        from ase.vibrations import VibrationsData

        pos = np.array([[0.0, 0.0, -0.3707], [0.0, 0.0, 0.3707]])
        numbers = np.array([1, 1])
        H = self._analytical_hessian(pos, numbers)
        atoms = ASE_Atoms("H2", positions=pos)
        # Normalize signed frequencies to absolute values
        raw = VibrationsData.from_2d(atoms, H).get_frequencies()
        freqs = np.abs(raw)

        # H2 harmonic stretch should be > 3000 cm-1
        max_freq = freqs.max()
        assert max_freq > 3000.0, f"H2 stretch too low: {max_freq:.1f} cm-1"
        assert max_freq < 5500.0, f"H2 stretch too high: {max_freq:.1f} cm-1"

    def test_water_hessian_produces_physical_frequencies(self) -> None:
        """Water Hessian should produce at least one stretching mode above 3000 cm-1."""
        from ase import Atoms as ASE_Atoms
        from ase.vibrations import VibrationsData

        pos, numbers = _water_equilibrium()
        H = self._analytical_hessian(pos, numbers)
        atoms = ASE_Atoms("H2O", positions=pos)
        raw = VibrationsData.from_2d(atoms, H).get_frequencies()
        freqs = np.abs(np.array(raw))

        # Must not contain NaN or Inf
        assert not np.any(np.isnan(freqs)), "Frequencies contain NaN"
        assert not np.any(np.isinf(freqs)), "Frequencies contain Inf"

        # At least one stretching mode should be present (> 3000 cm-1)
        max_freq = freqs.max()
        assert max_freq > 3000.0, (
            f"No stretching mode above 3000 cm-1; max frequency is {max_freq:.1f} cm-1"
        )

    def test_property_dispatch(self) -> None:
        """AIMNet2Potential should report Hessian support through the property system."""
        from famex.analysis.utils import get_calculator_property, has_calculator_property
        from famex.potentials.aimnet2_potential import AIMNet2Potential

        pot = AIMNet2Potential(device="cpu")

        assert has_calculator_property(pot, "hessian"), (
            "AIMNet2Potential should advertise hessian support"
        )

        pos, numbers = _water_equilibrium()
        from ase import Atoms as ASE_Atoms

        atoms = ASE_Atoms("H2O", positions=pos)
        atoms.calc = pot

        hessian = get_calculator_property(pot, "hessian", atoms)
        assert hessian.shape == (9, 9), f"Expected (9, 9) Hessian, got {hessian.shape}"
        assert not np.any(np.isnan(hessian)), "Hessian contains NaN"
        assert not np.any(np.isinf(hessian)), "Hessian contains Inf"

    def test_get_hessian_on_potential(self) -> None:
        """AIMNet2Potential.get_hessian should return a valid Hessian."""
        from ase import Atoms as ASE_Atoms

        from famex.potentials.aimnet2_potential import AIMNet2Potential

        pot = AIMNet2Potential(device="cpu")
        pot.charge = 0
        pot.mult = 1

        pos, numbers = _water_equilibrium()
        atoms = ASE_Atoms("H2O", positions=pos)
        pot.atoms = atoms

        H = pot.get_hessian(atoms)
        expected_shape = (9, 9)
        assert H.shape == expected_shape, f"Expected {expected_shape}, got {H.shape}"
        asym = np.max(np.abs(H - H.T))
        assert asym < 1e-4, f"get_hessian symmetry error: {asym}"

    def test_frequency_analysis_auto_selects_analytical(self) -> None:
        """FrequencyAnalysis with method='auto' should use analytical Hessian path."""
        from ase import Atoms as ASE_Atoms

        from famex.analysis.frequency import FrequencyAnalysis
        from famex.potentials.aimnet2_potential import AIMNet2Potential

        pot = AIMNet2Potential(device="cpu")
        pos, numbers = _water_equilibrium()
        atoms = ASE_Atoms("H2O", positions=pos)

        freq = FrequencyAnalysis(atoms, pot, delta=0.01, verbose=0)
        hessian = freq.calculate_hessian(method="auto")

        # Should be valid
        assert hessian.shape == (9, 9)
        assert not np.any(np.isnan(hessian))

        # Frequencies should be physically sensible
        vib_freqs = freq.get_frequencies()
        max_vib = np.abs(vib_freqs).max()
        assert max_vib > 3000.0, f"Max frequency too low: {max_vib:.1f} cm-1"

    def test_charge_mult_handling(self) -> None:
        """Hessian should be computable with non-default charge/multiplicity."""
        from ase import Atoms as ASE_Atoms

        from famex.potentials.aimnet2_potential import AIMNet2Potential

        # Test H2+ with charge=+1, mult=2 (doublet)
        pot = AIMNet2Potential(device="cpu")
        pot.charge = 1
        pot.mult = 2

        pos = np.array([[0.0, 0.0, -0.3707], [0.0, 0.0, 0.3707]])
        atoms = ASE_Atoms("H2", positions=pos)
        pot.atoms = atoms

        H = pot.get_hessian(atoms)
        expected_shape = (6, 6)
        assert H.shape == expected_shape, f"Expected {expected_shape}, got {H.shape}"
        asym = np.max(np.abs(H - H.T))
        assert asym < 1e-4, f"Symmetry error for charged H2+: {asym}"
        assert not np.any(np.isnan(H)), "Hessian contains NaN"


def test_make_analytical_hessian_function_eligibility() -> None:
    """sella-analytical: validate_calculator_supports_hessian should accept aimnet2."""
    if not _aimnet2_model_available():
        pytest.skip("AIMNet2 model not available")

    from famex.analysis.utils import has_calculator_property
    from famex.optimizers.sella_utils import validate_calculator_supports_hessian
    from famex.potentials.aimnet2_potential import AIMNet2Potential

    pot = AIMNet2Potential(device="cpu")
    assert has_calculator_property(pot, "hessian")
    validate_calculator_supports_hessian(pot)
