from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

import qme
from qme.analysis.frequency import FrequencyAnalysis
from qme.analysis.hessian import HessianCalculator
from qme.analysis.hessian_energy import EnergyBasedHessianCalculator
from qme.analysis.noise_estimation import (
    estimate_force_noise,
    estimate_optimal_delta,
    estimate_richardson_noise,
)
from qme.backends.availability import is_backend_available
from tests.test_utils import HarmonicCalculator, NoisyCalculator


class TestNoiseEstimation:
    @pytest.fixture
    def harmonic_atoms(self):
        return Atoms(
            symbols="HH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        )

    def test_estimate_richardson_noise(self):
        # Create two slightly different Hessians
        h1 = np.eye(3) + 0.01 * np.random.randn(3, 3)
        h2 = np.eye(3) + 0.01 * np.random.randn(3, 3)

        noise = estimate_richardson_noise(h1, h2)
        assert isinstance(noise, float)
        assert noise > 0

    def test_estimate_force_noise(self):
        atoms = Atoms(
            symbols="HH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        )

        # Test with noiseless calculator
        calc_noiseless = HarmonicCalculator()
        atoms.calc = calc_noiseless
        noise = estimate_force_noise(atoms, calc_noiseless, n_samples=5)

        assert isinstance(noise, float)
        assert noise >= 0

        # Test with noisy calculator
        calc_noisy = NoisyCalculator(noise_level=0.001)
        atoms.calc = calc_noisy
        noise = estimate_force_noise(atoms, calc_noisy, n_samples=5)

        assert isinstance(noise, float)
        assert noise >= 0

    def test_estimate_optimal_delta(self, harmonic_atoms):
        calc = HarmonicCalculator()
        harmonic_atoms.calc = calc

        optimal_delta, noise = estimate_optimal_delta(
            harmonic_atoms, calc, delta_range=(0.001, 0.05), max_iterations=3, verbose=0
        )

        assert isinstance(optimal_delta, float)
        assert isinstance(noise, float)
        assert optimal_delta >= 0.001
        assert optimal_delta <= 0.05
        assert noise >= 0


class TestAdaptiveHessianCalculator:
    @pytest.fixture
    def harmonic_atoms(self):
        return Atoms(
            symbols="HH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        )

    def test_adaptive_delta_basic(self, harmonic_atoms):
        calc = HarmonicCalculator()
        harmonic_atoms.calc = calc

        hessian_calc = HessianCalculator(
            harmonic_atoms,
            calc,
            delta=0.01,
            adaptive_delta=True,
            max_iterations=2,
            verbose=0,
        )

        hessian = hessian_calc.calculate_numerical_hessian()
        assert hessian is not None
        assert hessian.shape == (6, 6)

    def test_adaptive_delta_vs_fixed(self, harmonic_atoms):
        calc = HarmonicCalculator()
        harmonic_atoms.calc = calc

        # Fixed delta
        hessian_calc_fixed = HessianCalculator(
            harmonic_atoms,
            calc,
            delta=0.01,
            adaptive_delta=False,
            verbose=0,
        )
        hessian_fixed = hessian_calc_fixed.calculate_numerical_hessian()

        # Adaptive delta
        hessian_calc_adaptive = HessianCalculator(
            harmonic_atoms,
            calc,
            delta=0.01,
            adaptive_delta=True,
            max_iterations=2,
            verbose=0,
        )
        hessian_adaptive = hessian_calc_adaptive.calculate_numerical_hessian()

        # Should produce similar results
        np.testing.assert_allclose(hessian_fixed, hessian_adaptive, rtol=0.1, atol=0.1)

    def test_adaptive_delta_warnings(self, harmonic_atoms):
        calc = HarmonicCalculator()
        harmonic_atoms.calc = calc

        # Test with very small delta range
        hessian_calc = HessianCalculator(
            harmonic_atoms,
            calc,
            delta=0.01,
            adaptive_delta=True,
            delta_range=(0.0001, 0.001),  # Very small range
            max_iterations=1,
            verbose=0,
        )
        hessian = hessian_calc.calculate_numerical_hessian()
        assert hessian is not None


class TestEnergyBasedHessian:
    @pytest.fixture
    def harmonic_atoms(self):
        return Atoms(
            symbols="HH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        )

    def test_energy_based_basic(self, harmonic_atoms):
        calc = HarmonicCalculator()
        harmonic_atoms.calc = calc

        energy_calc = EnergyBasedHessianCalculator(harmonic_atoms, calc, delta=0.01, verbose=0)

        hessian = energy_calc.calculate_energy_hessian()
        assert hessian is not None
        assert hessian.shape == (6, 6)

    def test_energy_vs_force_consistency(self, harmonic_atoms):
        calc = HarmonicCalculator()
        harmonic_atoms.calc = calc

        # Energy-based FD
        energy_calc = EnergyBasedHessianCalculator(harmonic_atoms, calc, delta=0.01, verbose=0)
        hessian_energy = energy_calc.calculate_energy_hessian()

        # Force-based FD
        hessian_calc = HessianCalculator(harmonic_atoms, calc, delta=0.01, verbose=0)
        hessian_force = hessian_calc.calculate_numerical_hessian()

        # Should be very close for harmonic system
        np.testing.assert_allclose(hessian_energy, hessian_force, rtol=0.01, atol=0.1)


class TestAutoselectMethod:
    def test_autoselect_analytical(self):
        calc = HarmonicCalculator()
        atoms = Atoms(
            symbols="HH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        )
        atoms.calc = calc
        calc.atoms = atoms  # Set atoms for when get_hessian is called without argument

        freq_analysis = FrequencyAnalysis(atoms, calc, verbose=0)
        hessian = freq_analysis.calculate_hessian(method="autoselect")

        assert hessian is not None
        assert hessian.shape == (6, 6)

    def test_autoselect_adaptive_fd(self):
        calc = NoisyCalculator(noise_level=1e-6)  # Low noise
        atoms = Atoms(
            symbols="HH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        )
        atoms.calc = calc
        calc.atoms = atoms

        freq_analysis = FrequencyAnalysis(atoms, calc, delta=0.01, verbose=0)
        hessian = freq_analysis.calculate_hessian(method="autoselect")

        assert hessian is not None
        assert hessian.shape == (6, 6)

    @pytest.mark.skipif(not is_backend_available("uma"), reason="UMA backend not available")
    def test_autoselect_uma_integration(self, water_molecule):
        atoms = water_molecule.copy()
        atoms.calc = qme.get_uma_calculator(model_name="uma-s-1p1")
        atoms.calc.ensure_loaded()

        freq_analysis = FrequencyAnalysis(atoms, atoms.calc, verbose=0)
        hessian = freq_analysis.calculate_hessian(method="autoselect")

        assert hessian is not None
        assert hessian.shape == (9, 9)
        assert not np.any(np.isnan(hessian))
        assert not np.any(np.isinf(hessian))

        # Should be symmetric
        asymmetry = np.max(np.abs(hessian - hessian.T))
        assert asymmetry < 0.01  # Reasonable tolerance

    def test_autoselect_logging(self, water_molecule):
        calc = NoisyCalculator(noise_level=1e-6)
        atoms = water_molecule.copy()
        atoms.calc = calc
        calc.atoms = atoms

        freq_analysis = FrequencyAnalysis(atoms, calc, verbose=1)
        hessian = freq_analysis.calculate_hessian(method="autoselect")

        assert hessian is not None


class TestNoisySystems:
    @pytest.fixture
    def noisy_atoms(self):
        atoms = Atoms(
            symbols="HHH",
            positions=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        )
        return atoms

    def test_adaptive_delta_noise_handling(self, noisy_atoms):
        # Low noise
        calc_low = NoisyCalculator(noise_level=1e-6)
        noisy_atoms.calc = calc_low
        calc_low.atoms = noisy_atoms

        hessian_calc = HessianCalculator(
            noisy_atoms,
            calc_low,
            delta=0.01,
            adaptive_delta=True,
            max_iterations=2,
            verbose=0,
        )
        hessian = hessian_calc.calculate_numerical_hessian()

        assert hessian is not None
        assert hessian.shape == (9, 9)

    def test_adaptive_delta_very_noisy(self, noisy_atoms):
        # High noise
        calc_high = NoisyCalculator(noise_level=0.01)
        noisy_atoms.calc = calc_high
        calc_high.atoms = noisy_atoms

        hessian_calc = HessianCalculator(
            noisy_atoms,
            calc_high,
            delta=0.01,
            adaptive_delta=True,
            max_iterations=2,
            verbose=0,
        )

        # Should still produce a result (though quality will be poor)
        hessian = hessian_calc.calculate_numerical_hessian()
        assert hessian is not None

    def test_energy_based_noisy(self, noisy_atoms):
        calc = NoisyCalculator(noise_level=1e-5)
        noisy_atoms.calc = calc
        calc.atoms = noisy_atoms

        energy_calc = EnergyBasedHessianCalculator(noisy_atoms, calc, delta=0.01, verbose=0)
        hessian = energy_calc.calculate_energy_hessian()

        assert hessian is not None
        assert hessian.shape == (9, 9)
