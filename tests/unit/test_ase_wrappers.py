"""Tests for qme.optimizers.ase_wrappers module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from qme.optimizers.ase_wrappers import (
    LoggingFile,
    ProfilerCalculatorWrapper,
    VerboseOptimizerWrapper,
)
from tests.test_constants import LOOSE_FMAX, QUICK_STEPS


class TestLoggingFile:
    """Test LoggingFile class."""

    def test_write_with_newlines(self):
        """Test write() method with newlines in text (lines 67-75)."""
        with patch("qme.optimizers.ase_wrappers.logger") as mock_logger:
            mock_logger.getEffectiveLevel.return_value = 10  # INFO level
            logging_file = LoggingFile()
            # Write text with newlines
            result = logging_file.write("line1\nline2\nline3")
            assert result == len("line1\nline2\nline3")
            # Buffer should contain only incomplete line
            assert logging_file.buffer == "line3"

    def test_write_multiple_lines(self):
        """Test write() method with multiple complete lines."""
        with patch("qme.optimizers.ase_wrappers.logger") as mock_logger:
            mock_logger.getEffectiveLevel.return_value = 10  # INFO level
            logging_file = LoggingFile()
            logging_file.write("line1\nline2\n")
            # All lines should be processed, buffer should be empty
            assert logging_file.buffer == ""

    def test_write_partial_line(self):
        """Test write() method with partial line (no newline)."""
        with patch("qme.optimizers.ase_wrappers.logger") as mock_logger:
            mock_logger.getEffectiveLevel.return_value = 10  # INFO level
            logging_file = LoggingFile()
            logging_file.write("partial line")
            # Should accumulate in buffer
            assert logging_file.buffer == "partial line"

    def test_write_empty_lines(self):
        """Test write() method with empty lines."""
        logging_file = LoggingFile()
        logging_file.write("\n\n\n")
        # Empty lines should be skipped
        assert logging_file.buffer == ""

    def test_flush(self):
        """Test flush() method."""
        logging_file = LoggingFile()
        logging_file.write("partial")
        logging_file.flush()
        # Buffer should be cleared after flush
        assert logging_file.buffer == ""


class TestProfilerCalculatorWrapper:
    """Test ProfilerCalculatorWrapper class."""

    def test_calculate_with_energy(self, mock_backend, water_molecule):
        """Test calculate() method with energy property (lines 131-147)."""
        mock_profiler = MagicMock()
        # Ensure mock_backend has calculate method
        mock_backend.calculate = MagicMock(return_value={"energy": 0.0})
        wrapper = ProfilerCalculatorWrapper(mock_backend, mock_profiler)
        water_molecule.calc = wrapper

        result = wrapper.calculate(water_molecule, properties=["energy"])
        assert "energy" in result
        # Profiler should track energy call
        mock_profiler.increment_call.assert_any_call("energy")

    def test_calculate_with_forces(self, mock_backend, water_molecule):
        """Test calculate() method with forces property."""
        mock_profiler = MagicMock()
        mock_backend.calculate = MagicMock(
            return_value={"forces": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}
        )
        wrapper = ProfilerCalculatorWrapper(mock_backend, mock_profiler)
        water_molecule.calc = wrapper

        result = wrapper.calculate(water_molecule, properties=["forces"])
        assert "forces" in result
        # Profiler should track forces call
        mock_profiler.increment_call.assert_any_call("forces")

    def test_calculate_with_hessian(self, mock_backend, water_molecule):
        """Test calculate() method with hessian property."""
        import numpy as np

        mock_profiler = MagicMock()
        mock_backend.calculate = MagicMock(return_value={"hessian": np.eye(9)})
        wrapper = ProfilerCalculatorWrapper(mock_backend, mock_profiler)
        water_molecule.calc = wrapper

        result = wrapper.calculate(water_molecule, properties=["hessian"])
        assert "hessian" in result
        # Profiler should track hessian call
        mock_profiler.increment_call.assert_any_call("hessian")

    def test_calculate_with_multiple_properties(self, mock_backend, water_molecule):
        """Test calculate() method with multiple properties."""
        import numpy as np

        mock_profiler = MagicMock()
        mock_backend.calculate = MagicMock(return_value={"energy": 0.0, "forces": np.zeros((3, 3))})
        wrapper = ProfilerCalculatorWrapper(mock_backend, mock_profiler)
        water_molecule.calc = wrapper

        result = wrapper.calculate(water_molecule, properties=["energy", "forces"])
        assert "energy" in result
        assert "forces" in result
        # Profiler should track both calls
        mock_profiler.increment_call.assert_any_call("energy")
        mock_profiler.increment_call.assert_any_call("forces")

    def test_calculate_with_none_properties(self, mock_backend, water_molecule):
        """Test calculate() method with None properties (defaults to energy)."""
        mock_profiler = MagicMock()
        wrapper = ProfilerCalculatorWrapper(mock_backend, mock_profiler)
        water_molecule.calc = wrapper

        wrapper.calculate(water_molecule, properties=None)
        # Should default to energy
        mock_profiler.increment_call.assert_called_with("energy")


class TestVerboseOptimizerWrapper:
    """Test VerboseOptimizerWrapper class."""

    def test_init_with_logfile_none(self, water_molecule, mock_backend):
        """Test initialization with logfile=None (lines 261-275)."""
        from ase.optimize import BFGS

        water_molecule.calc = mock_backend
        wrapper = VerboseOptimizerWrapper(
            water_molecule,
            BFGS,
            logfile=None,
            verbose=1,
        )
        assert wrapper.verbose == 1
        assert wrapper._logging_file is not None

    def test_init_with_logfile_dash(self, water_molecule, mock_backend):
        """Test initialization with logfile='-'."""
        from ase.optimize import BFGS

        water_molecule.calc = mock_backend
        wrapper = VerboseOptimizerWrapper(
            water_molecule,
            BFGS,
            logfile="-",
            verbose=1,
        )
        assert wrapper.verbose == 1
        assert wrapper._logging_file is not None

    def test_init_with_logfile_path(self, water_molecule, mock_backend, tmp_path):
        """Test initialization with logfile as file path."""
        from ase.optimize import BFGS

        logfile_path = tmp_path / "test.log"
        water_molecule.calc = mock_backend
        wrapper = VerboseOptimizerWrapper(
            water_molecule,
            BFGS,
            logfile=str(logfile_path),
            verbose=1,
        )
        assert wrapper.verbose == 1

    def test_init_with_verbose_0(self, water_molecule, mock_backend):
        """Test initialization with verbose=0 (quiet mode)."""
        from ase.optimize import BFGS

        water_molecule.calc = mock_backend
        wrapper = VerboseOptimizerWrapper(
            water_molecule,
            BFGS,
            verbose=0,
        )
        assert wrapper.verbose == 0
        # logfile should be None in quiet mode, or /dev/null, or a Log object
        # Different ASE versions handle logfile=None differently:
        # - Some versions set logfile=None or create TextIOWrapper pointing to /dev/null
        # - Some versions (e.g., Python 3.10) create a Log object without a .name attribute
        logfile = wrapper.wrapped_optimizer.logfile
        assert (
            logfile is None
            or (hasattr(logfile, "name") and logfile.name == "/dev/null")
            or (
                hasattr(logfile, "__class__")
                and logfile.__class__.__module__ == "ase.optimize.optimize"
                and logfile.__class__.__name__ == "Log"
            )
        )  # Accept ASE Log objects created in some versions

    def test_init_atoms_calc_none_but_hasattr(self, water_molecule):
        """Test initialization when atoms.calc is None but hasattr returns True (lines 299-305)."""
        from ase.optimize import BFGS

        # Create atoms object where hasattr(atoms, "calc") is True but atoms.calc is None
        water_molecule.calc = None
        wrapper = VerboseOptimizerWrapper(
            water_molecule,
            BFGS,
            verbose=1,
        )
        # Should handle gracefully without error
        assert wrapper.verbose == 1

    def test_init_with_profiler(self, water_molecule, mock_backend):
        """Test initialization with profiler."""
        from ase.optimize import BFGS

        from qme.utils.profiler import PerformanceProfiler

        water_molecule.calc = mock_backend
        profiler = PerformanceProfiler()
        wrapper = VerboseOptimizerWrapper(
            water_molecule,
            BFGS,
            verbose=1,
            profiler=profiler,
        )
        # Calculator should be wrapped with ProfilerCalculatorWrapper
        assert isinstance(wrapper.atoms.calc, ProfilerCalculatorWrapper)

    def test_init_with_verbose_2(self, water_molecule, mock_backend):
        """Test initialization with verbose=2 (verbose mode)."""
        from ase.optimize import BFGS

        water_molecule.calc = mock_backend
        with patch("qme.optimizers.ase_wrappers.logger") as mock_logger:
            mock_logger.info = MagicMock()  # Make it callable
            mock_logger.getEffectiveLevel = MagicMock(return_value=10)  # INFO level
            VerboseOptimizerWrapper(
                water_molecule,
                BFGS,
                verbose=2,
            )
            # Should log initialization message
            assert mock_logger.info.called

    def test_run_with_verbose_2(self, water_molecule, mock_backend):
        """Test run() method with verbose=2."""
        from ase.optimize import BFGS

        water_molecule.calc = mock_backend
        wrapper = VerboseOptimizerWrapper(
            water_molecule,
            BFGS,
            verbose=2,
        )
        with patch("qme.optimizers.ase_wrappers.logger") as mock_logger:
            wrapper.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS)
            # Should log optimization start
            mock_logger.info.assert_called()

    def test_run_non_converged_with_scipy_reason(self, water_molecule, mock_backend):
        """Test run() method when optimization doesn't converge with scipy reason."""
        from unittest.mock import MagicMock

        from ase.optimize import BFGS

        water_molecule.calc = mock_backend
        wrapper = VerboseOptimizerWrapper(
            water_molecule,
            BFGS,
            verbose=1,
        )
        # Mock wrapped optimizer to return False and have _scipy_result
        wrapper.wrapped_optimizer.run = MagicMock(return_value=False)
        wrapper.wrapped_optimizer.get_number_of_steps = MagicMock(return_value=5)
        mock_scipy_result = MagicMock()
        mock_scipy_result.message = "Test reason"
        wrapper.wrapped_optimizer._scipy_result = mock_scipy_result

        with patch("qme.optimizers.ase_wrappers.logger") as mock_logger:
            wrapper.run(fmax=LOOSE_FMAX, steps=QUICK_STEPS)
            # Should log warning about non-convergence
            mock_logger.warning.assert_called()
