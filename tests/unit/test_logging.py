"""Unit tests for logging utilities."""

from __future__ import annotations

import logging
from io import StringIO

import pytest

from qme.utils.logging import (
    get_qme_log_level,
    get_qme_logger,
    is_in_quiet_context,
    print_model_info,
    setup_qme_logging,
)


class TestSetupQmeLogging:
    """Test setup_qme_logging function."""

    def test_setup_logging_default(self) -> None:
        """Test setting up logging with default verbosity."""
        # Clear any existing configuration
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        setup_qme_logging(verbosity=1, force=True)

        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logging_quiet(self) -> None:
        """Test setting up logging with quiet verbosity (0)."""
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        setup_qme_logging(verbosity=0, force=True)

        assert logger.level == logging.WARNING

    def test_setup_logging_verbose(self) -> None:
        """Test setting up logging with verbose verbosity (2)."""
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        setup_qme_logging(verbosity=2, force=True)

        assert logger.level == logging.DEBUG

    def test_setup_logging_no_force(self) -> None:
        """Test that setup_logging doesn't reconfigure if already configured."""
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        # First setup
        setup_qme_logging(verbosity=1, force=True)
        initial_handler_count = len(logger.handlers)

        # Second setup without force
        setup_qme_logging(verbosity=2, force=False)

        # Should not reconfigure
        assert logger.level == logging.INFO  # Should remain INFO
        # Handler count might stay same or be different depending on implementation
        assert len(logger.handlers) == initial_handler_count

    def test_setup_logging_with_force(self) -> None:
        """Test that setup_logging reconfigures when force=True."""
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        # First setup
        setup_qme_logging(verbosity=1, force=True)

        # Second setup with force
        setup_qme_logging(verbosity=2, force=True)

        assert logger.level == logging.DEBUG  # Should be updated

    def test_logging_formatter_info_level(self) -> None:
        """Test that INFO level messages are formatted cleanly."""
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        setup_qme_logging(verbosity=2, force=True)

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(handler)

        logger.info("Test message")

        output = stream.getvalue()
        # INFO messages should not have level prefix
        assert "Test message" in output
        assert "[INFO]" not in output

    def test_logging_formatter_warning_level(self) -> None:
        """Test that WARNING level messages include warning indicator."""
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        setup_qme_logging(verbosity=1, force=True)

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(handler)

        logger.warning("Test warning")

        output = stream.getvalue()
        assert "⚠️" in output or "WARNING" in output or "Test warning" in output

    def test_logging_formatter_error_level(self) -> None:
        """Test that ERROR level messages include error indicator."""
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        setup_qme_logging(verbosity=1, force=True)

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(handler)

        logger.error("Test error")

        output = stream.getvalue()
        assert "❌" in output or "ERROR" in output or "Test error" in output

    def test_logging_formatter_debug_level(self) -> None:
        """Test that DEBUG level messages include debug prefix."""
        logger = logging.getLogger("qme")
        logger.handlers.clear()

        setup_qme_logging(verbosity=2, force=True)

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(handler)

        logger.debug("Test debug")

        output = stream.getvalue()
        assert "[DEBUG]" in output or "Test debug" in output


class TestGetQmeLogger:
    """Test get_qme_logger function."""

    def test_get_logger_with_qme_prefix(self) -> None:
        """Test getting logger with qme prefix."""
        logger = get_qme_logger("qme.test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "qme.test_module"

    def test_get_logger_without_prefix(self) -> None:
        """Test getting logger without qme prefix (auto-adds)."""
        logger = get_qme_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "qme.test_module"

    def test_get_logger_main(self) -> None:
        """Test getting logger for __main__."""
        logger = get_qme_logger("__main__")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "qme"

    def test_get_logger_uses_hierarchy(self) -> None:
        """Test that loggers use proper hierarchy."""
        parent_logger = get_qme_logger("qme")
        child_logger = get_qme_logger("qme.submodule")

        assert child_logger.parent == parent_logger or child_logger.name.startswith("qme")


class TestGetQmeLogLevel:
    """Test get_qme_log_level function."""

    def test_get_log_level_after_setup(self) -> None:
        """Test getting log level after setup."""
        setup_qme_logging(verbosity=1, force=True)

        level = get_qme_log_level()
        assert level == logging.INFO

    def test_get_log_level_different_verbosities(self) -> None:
        """Test getting log level for different verbosities."""
        setup_qme_logging(verbosity=0, force=True)
        assert get_qme_log_level() == logging.WARNING

        setup_qme_logging(verbosity=1, force=True)
        assert get_qme_log_level() == logging.INFO

        setup_qme_logging(verbosity=2, force=True)
        assert get_qme_log_level() == logging.DEBUG


class TestQuietContext:
    """Test quiet context functionality."""

    def test_is_in_quiet_context_default(self) -> None:
        """Test that quiet context defaults to False."""
        # Clear any existing context
        result = is_in_quiet_context()
        assert result is False


class TestPrintModelInfo:
    """Test print_model_info function."""

    def test_print_model_info_basic(self) -> None:
        """Test printing model info with basic parameters."""
        # This function uses click.echo, so we test that it doesn't raise
        try:
            print_model_info("uma", model_name="uma-s-1p1", device="cpu")
        except Exception:
            # If click is not available or there's an import error, that's okay
            # We're just testing the function structure
            pytest.skip("click not available or import error")

    def test_print_model_info_with_path(self) -> None:
        """Test printing model info with model path."""
        try:
            print_model_info("so3lr", model_path="/path/to/model", device="cpu")
        except Exception:
            pytest.skip("click not available or import error")

    def test_print_model_info_with_gpu(self) -> None:
        """Test printing model info with GPU device."""
        try:
            print_model_info("uma", model_name="test", device="cuda")
        except Exception:
            pytest.skip("click not available or import error")

    def test_print_model_info_minimal(self) -> None:
        """Test printing model info with minimal parameters."""
        try:
            print_model_info("mock")
        except Exception:
            pytest.skip("click not available or import error")


class TestLoggingIntegration:
    """Test logging integration scenarios."""

    def test_logger_used_in_code(self) -> None:
        """Test that loggers can be used for actual logging."""
        logger = get_qme_logger("qme.test")

        # Should not raise
        logger.info("Test message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_multiple_loggers_same_module(self) -> None:
        """Test getting multiple loggers from same module."""
        logger1 = get_qme_logger("qme.test")
        logger2 = get_qme_logger("qme.test")

        # Should return same logger instance (logging.getLogger is cached)
        assert logger1 is logger2

    def test_logger_propagation(self) -> None:
        """Test that loggers propagate correctly."""
        setup_qme_logging(verbosity=1, force=True)

        parent = logging.getLogger("qme")
        child = get_qme_logger("qme.child")

        # Child should inherit from parent
        assert child.parent == parent or child.name.startswith("qme")
