from __future__ import annotations

import math

import numpy as np
import pytest
from ase import Atoms, units

from famex.analysis.frequency import FrequencyAnalysis
from famex.analysis.molecular_properties import rotational_temperatures_from_atoms
from famex.analysis.physics_constants import (
    AMU_TO_KG,
    AVOGADRO_CONSTANT,
    BOLTZMANN_CONSTANT,
    GAS_CONSTANT,
    J_PER_MOL_TO_EV,
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    gas_phase_concentration_1atm,
    ideal_gas_concentration,
)
from famex.analysis.quasiharmonic import (
    QuasiHarmonicHandler,
    calculate_damping_function,
    calculate_qRRHO_energy,
    calculate_rrho_energy,
    calculate_rrho_entropy,
)
from famex.analysis.solvation import SolvationHandler, get_free_space
from famex.analysis.statistical_thermo import (
    StatisticalThermodynamics,
    calculate_electronic_entropy,
    calculate_rotational_energy,
    calculate_rotational_entropy,
    calculate_translational_energy,
    calculate_translational_entropy,
)
from famex.analysis.symmetry import (
    SymmetryHandler,
    detect_symmetry_from_atoms,
    get_point_group_symmetry_number,
)
from famex.analysis.thermodynamics import ThermodynamicProperties, read_free_energy_correction

_TEMPERATURE = 298.15


def _thermal_factor(frequency_cm: float, temperature: float = _TEMPERATURE) -> float:
    return PLANCK_CONSTANT * frequency_cm * SPEED_OF_LIGHT / (BOLTZMANN_CONSTANT * temperature)


def _rrho_entropy(frequency_cm: float, temperature: float = _TEMPERATURE) -> float:
    x = _thermal_factor(frequency_cm, temperature)
    return GAS_CONSTANT * (x / math.expm1(x) - math.log(-math.expm1(-x)))


def _rrho_energy(frequency_cm: float, temperature: float = _TEMPERATURE) -> float:
    x = _thermal_factor(frequency_cm, temperature)
    return GAS_CONSTANT * temperature * (0.5 * x + x / math.expm1(x))


def _sackur_tetrode(
    mass_amu: float,
    temperature: float,
    concentration: float,
    free_space_ml_per_l: float = 1000.0,
) -> float:
    mass_kg = mass_amu * AMU_TO_KG
    inverse_wavelength_cubed = (
        2.0 * math.pi * mass_kg * BOLTZMANN_CONSTANT * temperature / PLANCK_CONSTANT**2
    ) ** 1.5
    number_density = concentration * 1000.0 * AVOGADRO_CONSTANT / (free_space_ml_per_l / 1000.0)
    return GAS_CONSTANT * (math.log(inverse_wavelength_cubed / number_density) + 2.5)


def _free_rotor_entropy(
    frequency_cm: float,
    temperature: float = _TEMPERATURE,
    average_moment: float = 1.0e-44,
) -> float:
    mu = PLANCK_CONSTANT / (8.0 * math.pi**2 * frequency_cm * SPEED_OF_LIGHT)
    mu_primed = mu * average_moment / (mu + average_moment)
    factor = 8.0 * math.pi**3 * mu_primed * BOLTZMANN_CONSTANT * temperature / PLANCK_CONSTANT**2
    return (0.5 + math.log(math.sqrt(factor))) * GAS_CONSTANT


def _damping(frequency_cm: float, cutoff: float, alpha: float = 4.0) -> float:
    return 1.0 / (1.0 + (cutoff / frequency_cm) ** alpha)


def _h2() -> Atoms:
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])


def _h2_rotational_temperature(atoms: Atoms) -> float:
    mass_kg = float(atoms.get_masses()[0]) * AMU_TO_KG
    inertia = 0.5 * mass_kg * (0.74e-10) ** 2
    return (PLANCK_CONSTANT**2) / (8.0 * math.pi**2 * inertia * BOLTZMANN_CONSTANT)


@pytest.fixture
def simple_molecule():
    return Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0.82, 0, -0.26]])


@pytest.fixture
def water_c2v():
    return Atoms(
        "OHH",
        positions=[
            [0.0, 0.0, 0.1173],
            [0.0, 0.7572, -0.4692],
            [0.0, -0.7572, -0.4692],
        ],
    )


class TestQuasiHarmonic:
    def test_rrho_entropy_matches_partition_function(self):
        frequencies = np.array([100.0, 500.0, 1000.0])
        entropy = calculate_rrho_entropy(frequencies, _TEMPERATURE)

        expected = np.array([_rrho_entropy(freq) for freq in frequencies])
        assert entropy == pytest.approx(expected, rel=1e-12)
        assert entropy[2] < entropy[1] < entropy[0]

    def test_rrho_energy_matches_zero_point_plus_thermal(self):
        energy = calculate_rrho_energy(np.array([1000.0]), _TEMPERATURE)
        assert energy[0] == pytest.approx(_rrho_energy(1000.0), rel=1e-12)

    def test_qrrho_energy_is_harmonic_energy(self):
        frequencies = np.array([50.0, 1000.0, 10000.0])
        assert calculate_qRRHO_energy(frequencies, _TEMPERATURE) == pytest.approx(
            calculate_rrho_energy(frequencies, _TEMPERATURE)
        )

    def test_high_frequency_entropy_is_finite_and_vanishes(self):
        entropy = calculate_rrho_entropy(np.array([1.0e6]), _TEMPERATURE)
        energy = calculate_rrho_energy(np.array([1.0e6]), _TEMPERATURE)
        zpe = 0.5 * PLANCK_CONSTANT * 1.0e6 * SPEED_OF_LIGHT * AVOGADRO_CONSTANT
        assert np.isfinite(entropy[0])
        assert entropy[0] == pytest.approx(0.0, abs=1e-12)
        assert energy[0] == pytest.approx(zpe, rel=1e-12)

    def test_damping_is_half_at_the_cutoff(self):
        damp = calculate_damping_function(np.array([50.0, 100.0, 500.0, 1000.0]), 100.0)
        assert damp[1] == pytest.approx(0.5)
        assert damp[3] > damp[2] > damp[1] > damp[0]
        assert np.all((damp >= 0.0) & (damp <= 1.0))

    def test_truhlar_raises_low_frequency_entropy_only(self):
        low = np.array([50.0])
        cutoff = 100.0
        handler = QuasiHarmonicHandler(method="truhlar", freq_cutoff=cutoff)
        entropy, _ = handler.vibrational_entropy(low, _TEMPERATURE)
        _, per_mode = handler.vibrational_energy(low, _TEMPERATURE)

        assert entropy == pytest.approx(_rrho_entropy(cutoff))
        assert per_mode[0] == pytest.approx(_rrho_energy(50.0))

    def test_scale_factor_is_applied_before_the_truhlar_cutoff(self):
        handler = QuasiHarmonicHandler(method="truhlar", freq_cutoff=100.0, freq_scale_factor=0.5)
        entropy, _ = handler.vibrational_entropy(np.array([150.0]), _TEMPERATURE)
        assert entropy == pytest.approx(_rrho_entropy(100.0))

    def test_grimme_damping_uses_scaled_frequencies(self):
        cutoff = 100.0
        scaled_frequency = 150.0 * 0.5
        weight = _damping(scaled_frequency, cutoff)
        expected_entropy = weight * _rrho_entropy(scaled_frequency) + (
            1.0 - weight
        ) * _free_rotor_entropy(scaled_frequency)
        expected_energy = (
            weight * _rrho_energy(scaled_frequency)
            + (1.0 - weight) * 0.5 * GAS_CONSTANT * _TEMPERATURE
        )
        handler = QuasiHarmonicHandler(
            method="grimme",
            freq_cutoff=cutoff,
            freq_scale_factor=0.5,
        )
        entropy, _ = handler.vibrational_entropy(np.array([150.0]), _TEMPERATURE)
        energy, _ = handler.vibrational_energy(np.array([150.0]), _TEMPERATURE)
        assert entropy == pytest.approx(expected_entropy, rel=1e-12)
        assert energy == pytest.approx(expected_energy, rel=1e-12)


class TestStatisticalThermodynamics:
    def test_translational_energy(self):
        energy = calculate_translational_energy(_TEMPERATURE)
        assert energy == pytest.approx(1.5 * GAS_CONSTANT * _TEMPERATURE)

    def test_translational_entropy_matches_sackur_tetrode(self):
        concentration = gas_phase_concentration_1atm(_TEMPERATURE)
        entropy = calculate_translational_entropy(18.0, _TEMPERATURE, concentration)
        assert entropy == pytest.approx(_sackur_tetrode(18.0, _TEMPERATURE, concentration))

    def test_rotational_energy(self):
        assert calculate_rotational_energy(_TEMPERATURE, linear=True) == pytest.approx(
            GAS_CONSTANT * _TEMPERATURE
        )
        assert calculate_rotational_energy(_TEMPERATURE, linear=False) == pytest.approx(
            1.5 * GAS_CONSTANT * _TEMPERATURE
        )
        assert calculate_rotational_energy(_TEMPERATURE, is_atom=True) == pytest.approx(0.0)

    def test_rotational_entropy_linear_pair_matches_unique_temperature(self):
        theta = 10.0
        one = calculate_rotational_entropy([theta], 2, _TEMPERATURE, linear=True)
        pair = calculate_rotational_entropy([theta, theta], 2, _TEMPERATURE, linear=True)
        with_axis = calculate_rotational_entropy([0.0, theta, theta], 2, _TEMPERATURE, linear=True)
        near_axis = calculate_rotational_entropy(
            [1.0e-4, theta, theta], 2, _TEMPERATURE, linear=True
        )
        expected = GAS_CONSTANT * (math.log((_TEMPERATURE / theta) / 2) + 1.0)
        assert one == pytest.approx(expected)
        assert pair == pytest.approx(one)
        assert with_axis == pytest.approx(one)
        assert near_axis == pytest.approx(one)

    def test_nonlinear_rotational_entropy_requires_three_temperatures(self):
        temps = np.array([10.0, 20.0, 30.0])
        entropy = calculate_rotational_entropy(temps, 1, _TEMPERATURE, linear=False)
        q_rot = math.sqrt(math.pi * _TEMPERATURE**3 / float(np.prod(temps)))
        assert entropy == pytest.approx(GAS_CONSTANT * (math.log(q_rot) + 1.5))
        with pytest.raises(ValueError, match="3 positive"):
            calculate_rotational_entropy([10.0, 10.0], 1, _TEMPERATURE, linear=False)

    def test_atom_rotational_entropy_is_zero(self):
        entropy = calculate_rotational_entropy(
            np.array([0.0, 0.0, 0.0]), 1, _TEMPERATURE, is_atom=True
        )
        assert entropy == pytest.approx(0.0)

    def test_electronic_entropy(self):
        assert calculate_electronic_entropy(1) == pytest.approx(0.0)
        assert calculate_electronic_entropy(3) == pytest.approx(GAS_CONSTANT * math.log(3))

    def test_rotational_constants_use_exact_gigahertz_conversion(self):
        atoms = _h2()
        stat = StatisticalThermodynamics(atoms, rotational_constants=[100.0], linear=True)
        wavenumber = 100.0 * 1e9 / SPEED_OF_LIGHT
        expected = PLANCK_CONSTANT * SPEED_OF_LIGHT * wavenumber / BOLTZMANN_CONSTANT
        assert stat.rotational_temperatures[0] == pytest.approx(expected)


class TestSolvation:
    def test_get_free_space(self):
        assert get_free_space("none") == pytest.approx(1000.0)
        assert 0.0 < get_free_space("H2O") < 1000.0
        for solvent in ("toluene", "DMF", "AcOH", "chloroform"):
            assert get_free_space(solvent) > 0.0
        with pytest.raises(ValueError, match="Unknown solvent"):
            get_free_space("unknown_solvent")

    def test_effective_concentration(self):
        assert SolvationHandler(solvent="none", concentration=1.0).effective_concentration() == (
            pytest.approx(1.0)
        )
        water = SolvationHandler(solvent="H2O", concentration=1.0)
        assert water.effective_concentration() == pytest.approx(
            1.0 / (water.free_space_ml_per_l / 1000.0)
        )

    def test_default_concentration_and_pressure(self):
        gas = SolvationHandler(solvent="none", temperature=_TEMPERATURE)
        assert gas.concentration == pytest.approx(ideal_gas_concentration(_TEMPERATURE, 101325.0))
        pressurized = SolvationHandler(solvent="none", temperature=_TEMPERATURE, pressure=1.0e6)
        assert pressurized.concentration == pytest.approx(
            ideal_gas_concentration(_TEMPERATURE, 1.0e6)
        )
        explicit = SolvationHandler(
            solvent="none",
            concentration=1.0,
            temperature=_TEMPERATURE,
            pressure=1.0e6,
        )
        assert explicit.concentration == pytest.approx(1.0)
        assert SolvationHandler(solvent="H2O", pressure=1.0e6).concentration == pytest.approx(1.0)


class TestSymmetry:
    def test_get_point_group_symmetry_number(self):
        assert get_point_group_symmetry_number("C1") == 1
        assert get_point_group_symmetry_number("C2v") == 2
        assert get_point_group_symmetry_number("Td") == 12
        assert get_point_group_symmetry_number("Oh") == 24

        with pytest.raises(ValueError):
            get_point_group_symmetry_number("unknown")

    def test_symmetry_handler_default(self):
        with pytest.warns(UserWarning, match="C1 symmetry"):
            handler = SymmetryHandler()
        assert handler.symmetry_number == 1
        assert handler.point_group == "C1"

    def test_symmetry_handler_explicit(self):
        handler = SymmetryHandler(symmetry_number=12)
        assert handler.symmetry_number == 12

    def test_symmetry_handler_point_group(self):
        handler = SymmetryHandler(point_group="C2v")
        assert handler.symmetry_number == 2
        assert handler.point_group == "C2v"

    def test_point_group_linear_symmetry_numbers(self):
        assert get_point_group_symmetry_number("Cinfv") == 1
        assert get_point_group_symmetry_number("Dinfh") == 2
        assert SymmetryHandler(point_group="Dinfh").symmetry_number == 2

    def test_molsymm_detects_water_c2v(self, water_c2v):
        sigma, point_group = detect_symmetry_from_atoms(water_c2v)
        assert sigma == 2
        assert point_group == "C2v"
        handler = SymmetryHandler(atoms=water_c2v)
        assert handler.symmetry_number == 2
        assert handler.point_group == "C2v"


class TestCompleteThermodynamics:
    def test_h2_rotational_temperature_and_thermochemistry(self):
        atoms = _h2()
        theta = _h2_rotational_temperature(atoms)
        assert rotational_temperatures_from_atoms(atoms, linear=True)[0] == pytest.approx(theta)

        frequencies = np.array([4401.0])
        results = ThermodynamicProperties(
            frequencies,
            atoms,
            temperature=_TEMPERATURE,
            symmetry_number=2,
            linear=True,
        ).calculate_complete_thermodynamics(energy=0.0)

        assert results["enthalpy_rot"] == pytest.approx(
            GAS_CONSTANT * _TEMPERATURE * J_PER_MOL_TO_EV
        )
        expected_s_rot = GAS_CONSTANT * (math.log((_TEMPERATURE / theta) / 2.0) + 1.0)
        assert results["entropy_rot"] == pytest.approx(expected_s_rot * J_PER_MOL_TO_EV)

        mass = float(np.sum(atoms.get_masses()))
        concentration = ideal_gas_concentration(_TEMPERATURE, 101325.0)
        u_vib = _rrho_energy(frequencies[0])
        enthalpy = (
            1.5 * GAS_CONSTANT * _TEMPERATURE
            + GAS_CONSTANT * _TEMPERATURE
            + u_vib
            + GAS_CONSTANT * _TEMPERATURE
        )
        entropy = (
            _sackur_tetrode(mass, _TEMPERATURE, concentration)
            + expected_s_rot
            + _rrho_entropy(frequencies[0])
        )
        assert results["enthalpy_total"] == pytest.approx(enthalpy * J_PER_MOL_TO_EV, rel=1e-12)
        assert results["entropy_total"] == pytest.approx(entropy * J_PER_MOL_TO_EV, rel=1e-12)
        assert results["gibbs_free_energy"] == pytest.approx(
            (enthalpy - _TEMPERATURE * entropy) * J_PER_MOL_TO_EV,
            rel=1e-12,
        )

    def test_symmetry_number_changes_free_energy_by_rt_log_sigma(self, water_c2v):
        frequencies = np.array([1595.0, 3657.0, 3756.0])
        common = {
            "temperature": _TEMPERATURE,
            "method": "rrho",
            "linear": False,
        }
        g_sigma1 = ThermodynamicProperties(
            frequencies, water_c2v, symmetry_number=1, **common
        ).calculate_complete_thermodynamics(energy=0.0)["gibbs_free_energy"]
        g_sigma2 = ThermodynamicProperties(
            frequencies, water_c2v, symmetry_number=2, **common
        ).calculate_complete_thermodynamics(energy=0.0)["gibbs_free_energy"]
        delta = GAS_CONSTANT * _TEMPERATURE * math.log(2.0) * J_PER_MOL_TO_EV
        assert g_sigma2 - g_sigma1 == pytest.approx(delta, rel=1e-10)

    def test_pressure_changes_gas_phase_free_energy(self, water_c2v):
        frequencies = np.array([1595.0, 3657.0, 3756.0])
        low = ThermodynamicProperties(
            frequencies,
            water_c2v,
            temperature=_TEMPERATURE,
            pressure=101325.0,
            symmetry_number=2,
        ).calculate_complete_thermodynamics(energy=0.0)
        high = ThermodynamicProperties(
            frequencies,
            water_c2v,
            temperature=_TEMPERATURE,
            pressure=1013250.0,
            symmetry_number=2,
        ).calculate_complete_thermodynamics(energy=0.0)
        delta = GAS_CONSTANT * _TEMPERATURE * math.log(10.0) * J_PER_MOL_TO_EV
        assert high["concentration"] == pytest.approx(10.0 * low["concentration"])
        assert high["gibbs_free_energy"] - low["gibbs_free_energy"] == pytest.approx(delta)

        fixed = ThermodynamicProperties(
            frequencies,
            water_c2v,
            temperature=_TEMPERATURE,
            pressure=1013250.0,
            concentration=1.0,
            symmetry_number=2,
        ).calculate_complete_thermodynamics(energy=0.0)
        reference = ThermodynamicProperties(
            frequencies,
            water_c2v,
            temperature=_TEMPERATURE,
            pressure=101325.0,
            concentration=1.0,
            symmetry_number=2,
        ).calculate_complete_thermodynamics(energy=0.0)
        assert fixed["gibbs_free_energy"] == pytest.approx(reference["gibbs_free_energy"])

        solution_low = ThermodynamicProperties(
            frequencies, water_c2v, solvent="H2O", pressure=101325.0, symmetry_number=2
        ).calculate_complete_thermodynamics(energy=0.0)
        solution_high = ThermodynamicProperties(
            frequencies, water_c2v, solvent="H2O", pressure=1.0e6, symmetry_number=2
        ).calculate_complete_thermodynamics(energy=0.0)
        assert solution_low["concentration"] == pytest.approx(1.0)
        assert solution_high["gibbs_free_energy"] == pytest.approx(
            solution_low["gibbs_free_energy"]
        )

    def test_imaginary_modes_are_excluded_and_scale_factor_scales_zpe(self, simple_molecule):
        real = np.array([1000.0])
        with_imaginary = np.array([1000.0, -200.0, 0.0, 0.0 + 250.0j])
        baseline = ThermodynamicProperties(real, simple_molecule, symmetry_number=1)
        contaminated = ThermodynamicProperties(with_imaginary, simple_molecule, symmetry_number=1)
        assert contaminated.calculate_zero_point_energy() == pytest.approx(
            baseline.calculate_zero_point_energy()
        )
        assert contaminated.entropy_vibrational() == pytest.approx(baseline.entropy_vibrational())
        assert contaminated.heat_capacity_vibrational() == pytest.approx(
            baseline.heat_capacity_vibrational()
        )

        scaled = ThermodynamicProperties(
            real, simple_molecule, symmetry_number=1, freq_scale_factor=0.96
        )
        scaled_frequency = 1000.0 * 0.96
        x_scaled = _thermal_factor(scaled_frequency)
        assert scaled.calculate_zero_point_energy() == pytest.approx(
            0.96 * baseline.calculate_zero_point_energy()
        )
        assert scaled.entropy_vibrational() == pytest.approx(
            _rrho_entropy(scaled_frequency) * J_PER_MOL_TO_EV
        )
        assert scaled.heat_capacity_vibrational() == pytest.approx(
            units.kB * x_scaled**2 * math.exp(x_scaled) / math.expm1(x_scaled) ** 2
        )

    def test_vibrational_breakdown_reconstructs_enthalpy(self, simple_molecule):
        frequencies = np.array([50.0, 100.0, 500.0, 1000.0, 1650.0, 3650.0])
        for method in ("rrho", "truhlar", "grimme"):
            thermo = ThermodynamicProperties(
                frequencies,
                simple_molecule,
                temperature=_TEMPERATURE,
                method=method,
                symmetry_number=1,
            )
            results = thermo.calculate_complete_thermodynamics(energy=0.0)
            u_vib, _ = thermo.qh_handler.vibrational_energy(frequencies, _TEMPERATURE)
            assert results["zpe"] + results["enthalpy_vib"] == pytest.approx(
                u_vib * J_PER_MOL_TO_EV, rel=1e-12
            )
            assert results["contributions"]["zero_point"] == pytest.approx(results["zpe"])
            if method == "grimme":
                assert results["zpe"] == pytest.approx(0.0)
                assert thermo.calculate_zero_point_energy() > 0.0
            else:
                assert results["zpe"] == pytest.approx(thermo.calculate_zero_point_energy())

    def test_heat_capacity_matches_harmonic_oscillator(self, simple_molecule):
        frequency = 1000.0
        thermo = ThermodynamicProperties(
            np.array([frequency]), simple_molecule, temperature=_TEMPERATURE, symmetry_number=1
        )
        x = _thermal_factor(frequency)
        expected = units.kB * x**2 * math.exp(x) / math.expm1(x) ** 2
        assert thermo.heat_capacity_vibrational() == pytest.approx(expected, rel=1e-12)

    def test_atom_has_no_rotational_contribution(self):
        atom = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        results = ThermodynamicProperties(
            np.array([]),
            atom,
            temperature=_TEMPERATURE,
            symmetry_number=1,
            linear=True,
        ).calculate_complete_thermodynamics(energy=0.0)
        assert results["enthalpy_rot"] == pytest.approx(0.0)
        assert results["entropy_rot"] == pytest.approx(0.0)
        assert results["zpe"] == pytest.approx(0.0)

    def test_frequency_report_keeps_harmonic_zpe_for_grimme(self):
        atoms = _h2()
        frequencies = np.array([4401.0])

        class _FixedFrequencies(FrequencyAnalysis):
            def get_frequencies(
                self,
                unit: str = "cm-1",
                imag_threshold: float = 50.0,
            ) -> np.ndarray:
                return frequencies

        reported = _FixedFrequencies(
            atoms, calculator=object(), nfree=0
        ).get_thermodynamic_properties(
            temperature=_TEMPERATURE,
            method="grimme",
            symmetry_number=2,
            linear=True,
        )
        thermo = ThermodynamicProperties(
            frequencies,
            atoms,
            temperature=_TEMPERATURE,
            method="grimme",
            symmetry_number=2,
            linear=True,
        )
        assert reported["zero_point_energy"] == pytest.approx(thermo.calculate_zero_point_energy())
        assert reported["zero_point_energy"] > 0.0
        assert thermo.calculate_complete_thermodynamics(energy=0.0)["zpe"] == pytest.approx(0.0)

    def test_read_free_energy_correction(self):
        assert read_free_energy_correction(None) is None
        assert read_free_energy_correction(
            {"thermodynamic_properties": {"free_energy_correction": 0.25}}
        ) == pytest.approx(0.25)
