"""Unit tests for QME utility modules: logging, profiler, and ml_warnings."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

import qme
from qme.utils.logging import (
    get_qme_log_level,
    get_qme_logger,
    is_in_quiet_context,
    print_model_info,
    setup_qme_logging,
)
from qme.utils.ml_warnings import VerboseFilter, quiet_backend_loading, suppress_ml_warnings
from qme.utils.profiler import (
    MemoryInfo,
    PerformanceProfiler,
    TimingEntry,
    profile_call,
    profile_optimizer_step,
)


class TestLogging:
    """Test logging configuration and utilities."""

    def test_setup_qme_logging_default(self) -> None:
        """Test default logging setup."""
        setup_qme_logging(force=True)

        logger = logging.getLogger("qme")
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    @pytest.mark.parametrize(
        ("verbosity", "expected_level"),
        [
            (0, logging.WARNING),
            (1, logging.INFO),
            (2, logging.DEBUG),
        ],
    )
    def test_setup_qme_logging_verbosity(self, verbosity: int, expected_level: int) -> None:
        """Test logging setup with different verbosity levels."""
        setup_qme_logging(verbosity=verbosity, force=True)

        logger = logging.getLogger("qme")
        assert logger.level == expected_level

    def test_setup_qme_logging_no_force(self) -> None:
        """Test that logging setup doesn't reconfigure if already configured."""
        setup_qme_logging(force=True)
        initial_handlers = len(logging.getLogger("qme").handlers)

        # Setup again without force - should not add new handlers
        setup_qme_logging(force=False)
        final_handlers = len(logging.getLogger("qme").handlers)

        # Should have same number of handlers (or fewer if cleared)
        assert final_handlers <= initial_handlers

    def test_get_qme_logger_main(self) -> None:
        """Test getting logger for __main__."""
        logger = get_qme_logger("__main__")
        assert logger.name == "qme"

    def test_get_qme_logger_module_name(self) -> None:
        """Test getting logger for module names."""
        logger = get_qme_logger("test_module")
        assert logger.name == "qme.test_module"

        logger2 = get_qme_logger("qme.already.qme.module")
        assert logger2.name == "qme.already.qme.module"

    def test_get_qme_log_level(self) -> None:
        """Test getting current log level."""
        setup_qme_logging(verbosity=2, force=True)
        level = get_qme_log_level()
        assert level == logging.DEBUG

        setup_qme_logging(verbosity=0, force=True)
        level = get_qme_log_level()
        assert level == logging.WARNING

    def test_is_in_quiet_context_default(self) -> None:
        """Test quiet context check returns False by default."""
        # By default, should not be in quiet context
        result = is_in_quiet_context()
        assert result is False

    @patch("click.echo")
    @patch("qme.utils.device.get_device_info")
    def test_print_model_info_minimal(
        self, mock_get_device_info: MagicMock, mock_echo: MagicMock
    ) -> None:
        """Test print_model_info with minimal information."""
        print_model_info("test_backend")

        # Should have printed backend name
        assert mock_echo.call_count >= 2  # Header and separator at minimum
        call_args = [str(call) for call in mock_echo.call_args_list]
        assert any("test_backend" in arg.upper() or "TEST_BACKEND" in arg for arg in call_args)

    @patch("click.echo")
    @patch("qme.utils.device.get_device_info")
    def test_print_model_info_full(
        self, mock_get_device_info: MagicMock, mock_echo: MagicMock
    ) -> None:
        """Test print_model_info with all information."""
        mock_get_device_info.return_value = {"gpu_name": "Test GPU", "cuda_available": True}

        print_model_info(
            backend="test_backend",
            model_name="test_model",
            model_path="/path/to/model",
            device="cuda",
        )

        # Should have printed all information
        call_args = [str(call) for call in mock_echo.call_args_list]
        assert any("test_backend" in arg.upper() or "TEST_BACKEND" in arg for arg in call_args)
        assert any("test_model" in arg for arg in call_args)
        assert any("/path/to/model" in arg for arg in call_args)
        assert any("cuda" in arg.lower() for arg in call_args)

    @patch("click.echo")
    @patch("qme.utils.device.get_device_info")
    def test_print_model_info_device_error_handling(
        self,
        mock_get_device_info: MagicMock,
        mock_echo: MagicMock,
    ) -> None:
        """Test that print_model_info handles device info errors gracefully."""
        mock_get_device_info.side_effect = Exception("GPU info error")

        # Should not raise exception
        print_model_info(backend="test", device="cuda")
        assert mock_echo.called


class TestProfiler:
    """Test performance profiler functionality."""

    def test_profiler_initialization(self) -> None:
        """Test profiler initialization."""
        profiler = PerformanceProfiler()

        assert profiler._gpu_available is not None  # Should detect GPU availability
        assert len(profiler._memory_snapshots) >= 1  # Should have initial snapshot
        assert profiler._calculator_calls["energy"] == 0
        assert profiler._calculator_calls["forces"] == 0

    def test_increment_call_existing(self) -> None:
        """Test incrementing existing call counter."""
        profiler = PerformanceProfiler()

        profiler.increment_call("energy", count=3)
        assert profiler._calculator_calls["energy"] == 3

        profiler.increment_call("energy", count=2)
        assert profiler._calculator_calls["energy"] == 5

    def test_increment_call_new(self) -> None:
        """Test incrementing new call counter."""
        profiler = PerformanceProfiler()

        profiler.increment_call("custom_call", count=7)
        assert profiler._calculator_calls["custom_call"] == 7

    def test_profile_section_context_manager(self) -> None:
        """Test profile_section context manager."""
        profiler = PerformanceProfiler()

        with profiler.profile_section("test_section"):
            pass

        assert "test_section" in profiler._timings
        assert len(profiler._timings["test_section"]) == 1
        timing = profiler._timings["test_section"][0]
        assert timing.duration is not None
        assert timing.duration >= 0

    def test_profile_section_with_parent(self) -> None:
        """Test profile_section with parent."""
        profiler = PerformanceProfiler()

        with profiler.profile_section("parent"):  # noqa: SIM117
            with profiler.profile_section("child", parent="parent"):
                pass

        assert "parent" in profiler._timings
        assert "child" in profiler._timings
        child_timing = profiler._timings["child"][0]
        assert child_timing.parent == "parent"

    def test_start_end_timing(self) -> None:
        """Test manual timing with start_timing and end_timing."""
        profiler = PerformanceProfiler()

        timing = profiler.start_timing("manual_timing")
        assert timing.name == "manual_timing"
        assert timing.start_time > 0

        duration = profiler.end_timing("manual_timing")
        assert duration is not None
        assert duration >= 0
        assert "manual_timing" in profiler._timings

    def test_end_timing_nonexistent(self) -> None:
        """Test ending timing that doesn't exist."""
        profiler = PerformanceProfiler()

        duration = profiler.end_timing("nonexistent")
        assert duration is None

    def test_snapshot_memory(self) -> None:
        """Test memory snapshot functionality."""
        profiler = PerformanceProfiler()

        initial_snapshots = len(profiler._memory_snapshots)
        memory_info = profiler.snapshot_memory()

        assert isinstance(memory_info, MemoryInfo)
        assert memory_info.ram_mb > 0
        assert len(profiler._memory_snapshots) == initial_snapshots + 1

    def test_get_summary(self) -> None:
        """Test getting profiler summary."""
        profiler = PerformanceProfiler()

        # Add some activity
        profiler.increment_call("energy", count=5)
        profiler.increment_call("forces", count=10)
        with profiler.profile_section("test"):
            pass

        summary = profiler.get_summary()

        assert "total_time" in summary
        assert "timings" in summary
        assert "memory" in summary
        assert "calculator_calls" in summary
        assert "resources" in summary
        assert "gpu_available" in summary

        assert summary["calculator_calls"]["energy"] == 5
        assert summary["calculator_calls"]["forces"] == 10
        assert "test" in summary["timings"]

    def test_get_summary_timing_stats(self) -> None:
        """Test timing statistics in summary."""
        profiler = PerformanceProfiler()

        # Multiple timings of same section
        for _ in range(3):
            with profiler.profile_section("repeated"):
                pass

        summary = profiler.get_summary()
        timing_stats = summary["timings"]["repeated"]

        assert timing_stats["count"] == 3
        assert timing_stats["total_time"] > 0
        assert timing_stats["avg_time"] > 0
        assert timing_stats["min_time"] >= 0
        assert timing_stats["max_time"] >= 0

    def test_timing_entry_finish(self) -> None:
        """Test TimingEntry finish method."""
        import time

        timing = TimingEntry(name="test", start_time=time.perf_counter())
        assert timing.end_time is None
        assert timing.duration is None

        duration = timing.finish()

        assert timing.end_time is not None
        assert timing.duration is not None
        assert duration == timing.duration
        assert duration >= 0

    def test_profile_call_decorator_with_profiler(self) -> None:
        """Test profile_call decorator when profiler is available."""
        profiler = PerformanceProfiler()

        class TestClass:
            def __init__(self) -> None:
                self.profiler = profiler

            @profile_call
            def test_method(self) -> str:
                return "result"

        obj = TestClass()
        result = obj.test_method()

        assert result == "result"
        # Check that timing was recorded
        summary = profiler.get_summary()
        assert len(summary["timings"]) > 0

    def test_profile_call_decorator_without_profiler(self) -> None:
        """Test profile_call decorator when profiler is not available."""

        class TestClass:
            @profile_call
            def test_method(self) -> str:
                return "result"

        obj = TestClass()
        result = obj.test_method()

        assert result == "result"

    def test_profile_optimizer_step_decorator(self) -> None:
        """Test profile_optimizer_step decorator."""
        profiler = PerformanceProfiler()

        class Optimizer:
            def __init__(self) -> None:
                self.profiler = profiler

            @profile_optimizer_step
            def step(self) -> None:
                pass

        optimizer = Optimizer()
        optimizer.step()

        summary = profiler.get_summary()
        assert summary["calculator_calls"]["optimizer_steps"] == 1
        assert "optimizer_step" in summary["timings"]


class TestMLWarnings:
    """Test ML warning suppression utilities."""

    def test_verbose_filter_logger_name(self) -> None:
        """Test VerboseFilter filtering by logger name."""
        filter_obj = VerboseFilter()
        record = logging.LogRecord(
            name="torch.some.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Some message",
            args=(),
            exc_info=None,
        )

        # Should filter out torch logger
        assert filter_obj.filter(record) is False

    def test_verbose_filter_suppressed_pattern(self) -> None:
        """Test VerboseFilter filtering by message pattern."""
        filter_obj = VerboseFilter()
        record = logging.LogRecord(
            name="some.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="NumExpr defaulting to something",
            args=(),
            exc_info=None,
        )

        # Should filter out messages with suppressed pattern
        assert filter_obj.filter(record) is False

    def test_verbose_filter_normal_message(self) -> None:
        """Test VerboseFilter allows normal messages."""
        filter_obj = VerboseFilter()
        record = logging.LogRecord(
            name="qme.some.module",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Normal QME message",
            args=(),
            exc_info=None,
        )

        # Should allow normal messages
        assert filter_obj.filter(record) is True

    def test_suppress_ml_warnings_context(self) -> None:
        """Test suppress_ml_warnings context manager."""
        original_stderr = sys.stderr

        with suppress_ml_warnings() as captured:
            # Write something to stderr
            sys.stderr.write("Normal message\n")
            sys.stderr.write("NumExpr defaulting to something\n")

        # Should restore stderr
        assert sys.stderr == original_stderr

        # Should have captured messages
        assert isinstance(captured, list)

    def test_suppress_ml_warnings_restores_stderr(self) -> None:
        """Test that suppress_ml_warnings properly restores stderr even on exception."""
        original_stderr = sys.stderr

        try:
            with suppress_ml_warnings():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still restore stderr
        assert sys.stderr == original_stderr

    @patch("qme.utils.logging.print_model_info")
    @patch("qme.utils.ml_warnings.suppress_ml_warnings")
    def test_quiet_backend_loading_basic(
        self,
        mock_suppress: MagicMock,
        mock_print_info: MagicMock,
    ) -> None:
        """Test quiet_backend_loading context manager."""
        mock_suppress.return_value.__enter__ = MagicMock(return_value=[])
        mock_suppress.return_value.__exit__ = MagicMock(return_value=None)

        with quiet_backend_loading("test_backend"):
            pass

        # Should call suppress_ml_warnings
        assert mock_suppress.called
        # Should call print_model_info by default
        assert mock_print_info.called

    @patch("qme.utils.logging.print_model_info")
    @patch("qme.utils.ml_warnings.suppress_ml_warnings")
    def test_quiet_backend_loading_no_info(
        self,
        mock_suppress: MagicMock,
        mock_print_info: MagicMock,
    ) -> None:
        """Test quiet_backend_loading with show_model_info=False."""
        mock_suppress.return_value.__enter__ = MagicMock(return_value=[])
        mock_suppress.return_value.__exit__ = MagicMock(return_value=None)

        with quiet_backend_loading("test_backend", show_model_info=False):
            pass

        # Should call suppress_ml_warnings
        assert mock_suppress.called
        # Should NOT call print_model_info
        assert not mock_print_info.called

    @patch("qme.utils.logging.is_in_quiet_context")
    @patch("qme.utils.logging.print_model_info")
    @patch("qme.utils.ml_warnings.suppress_ml_warnings")
    def test_quiet_backend_loading_nested_context(
        self,
        mock_suppress: MagicMock,
        mock_print_info: MagicMock,
        mock_is_quiet: MagicMock,
    ) -> None:
        """Test quiet_backend_loading when already in quiet context."""
        mock_suppress.return_value.__enter__ = MagicMock(return_value=[])
        mock_suppress.return_value.__exit__ = MagicMock(return_value=None)
        mock_is_quiet.return_value = True  # Already in quiet context

        with quiet_backend_loading("test_backend", show_model_info=True):
            pass

        # Should call suppress_ml_warnings
        assert mock_suppress.called
        # Should NOT call print_model_info if already in quiet context
        assert not mock_print_info.called


class TestQMEInitModule:
    """Test qme.__init__ module lazy imports."""

    def test_explorer_lazy_import(self):
        """Test lazy import of Explorer."""
        from qme import Explorer

        assert Explorer is not None

    def test_profiler_lazy_import(self):
        """Test lazy import of PerformanceProfiler."""
        from qme import PerformanceProfiler

        assert PerformanceProfiler is not None

    def test_deps_lazy_import(self):
        """Test lazy import of deps."""
        from qme import deps

        assert deps is not None

    def test_geometry_lazy_import(self):
        """Test lazy import of Geometry."""
        from qme import Geometry

        assert Geometry is not None

    def test_invalid_attribute_error(self):
        """Test that invalid attribute raises AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute 'invalid_attr_xyz'"):
            _ = qme.invalid_attr_xyz  # noqa: F841
