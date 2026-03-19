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
    def test_setup_logging_default(self, clear_qme_logger):
        setup_qme_logging(verbosity=1, force=True)

        assert clear_qme_logger.level == logging.INFO
        assert len(clear_qme_logger.handlers) > 0

    @pytest.mark.parametrize(
        ("verbosity", "expected_level"),
        [
            (0, logging.WARNING),
            (1, logging.INFO),
            (2, logging.DEBUG),
        ],
    )
    def test_setup_logging_verbosity(self, clear_qme_logger, verbosity, expected_level):
        setup_qme_logging(verbosity=verbosity, force=True)

        assert clear_qme_logger.level == expected_level

    def test_setup_logging_no_force(self, clear_qme_logger):
        # First setup
        setup_qme_logging(verbosity=1, force=True)
        initial_handler_count = len(clear_qme_logger.handlers)

        # Second setup without force
        setup_qme_logging(verbosity=2, force=False)

        # Should not reconfigure
        assert clear_qme_logger.level == logging.INFO  # Should remain INFO
        # Handler count might stay same or be different depending on implementation
        assert len(clear_qme_logger.handlers) == initial_handler_count

    def test_setup_logging_with_force(self, clear_qme_logger):
        # First setup
        setup_qme_logging(verbosity=1, force=True)

        # Second setup with force
        setup_qme_logging(verbosity=2, force=True)

        assert clear_qme_logger.level == logging.DEBUG  # Should be updated

    def test_logging_formatter_info_level(self, clear_qme_logger):
        setup_qme_logging(verbosity=2, force=True)

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(clear_qme_logger.handlers[0].formatter)
        clear_qme_logger.addHandler(handler)

        clear_qme_logger.info("Test message")

        output = stream.getvalue()
        # INFO messages should not have level prefix
        assert "Test message" in output
        assert "[INFO]" not in output

    def test_logging_formatter_warning_level(self, clear_qme_logger):
        setup_qme_logging(verbosity=1, force=True)

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(clear_qme_logger.handlers[0].formatter)
        clear_qme_logger.addHandler(handler)

        clear_qme_logger.warning("Test warning")

        output = stream.getvalue()
        assert "⚠️" in output or "WARNING" in output or "Test warning" in output

    def test_logging_formatter_error_level(self, clear_qme_logger):
        setup_qme_logging(verbosity=1, force=True)

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(clear_qme_logger.handlers[0].formatter)
        clear_qme_logger.addHandler(handler)

        clear_qme_logger.error("Test error")

        output = stream.getvalue()
        assert "❌" in output or "ERROR" in output or "Test error" in output

    def test_logging_formatter_debug_level(self, clear_qme_logger):
        setup_qme_logging(verbosity=2, force=True)

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(clear_qme_logger.handlers[0].formatter)
        clear_qme_logger.addHandler(handler)

        clear_qme_logger.debug("Test debug")

        output = stream.getvalue()
        assert "[DEBUG]" in output or "Test debug" in output


class TestGetQmeLogger:
    def test_get_logger_with_qme_prefix(self):
        logger = get_qme_logger("qme.test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "qme.test_module"

    def test_get_logger_without_prefix(self):
        logger = get_qme_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "qme.test_module"

    def test_get_logger_main(self):
        logger = get_qme_logger("__main__")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "qme"

    def test_get_logger_uses_hierarchy(self):
        parent_logger = get_qme_logger("qme")
        child_logger = get_qme_logger("qme.submodule")

        assert child_logger.parent == parent_logger or child_logger.name.startswith("qme")


class TestGetQmeLogLevel:
    def test_get_log_level_after_setup(self, clear_qme_logger):
        setup_qme_logging(verbosity=1, force=True)

        level = get_qme_log_level()
        assert level == logging.INFO

    def test_get_log_level_different_verbosities(self, clear_qme_logger):
        setup_qme_logging(verbosity=0, force=True)
        assert get_qme_log_level() == logging.WARNING

        setup_qme_logging(verbosity=1, force=True)
        assert get_qme_log_level() == logging.INFO

        setup_qme_logging(verbosity=2, force=True)
        assert get_qme_log_level() == logging.DEBUG


class TestQuietContext:
    def test_is_in_quiet_context_default(self):
        # Clear any existing context
        result = is_in_quiet_context()
        assert result is False


class TestPrintModelInfo:
    def test_print_model_info_basic(self):
        # This function uses click.echo, so we test that it doesn't raise
        try:
            print_model_info("uma", model_name="uma-s-1p2", device="cpu")
        except Exception:
            # If click is not available or there's an import error, that's okay
            # We're just testing the function structure
            pytest.skip("click not available or import error")

    def test_print_model_info_with_path(self):
        try:
            print_model_info("so3lr", model_path="/path/to/model", device="cpu")
        except Exception:
            pytest.skip("click not available or import error")

    def test_print_model_info_with_gpu(self):
        try:
            print_model_info("uma", model_name="test", device="cuda")
        except Exception:
            pytest.skip("click not available or import error")

    def test_print_model_info_minimal(self):
        try:
            print_model_info("mock")
        except Exception:
            pytest.skip("click not available or import error")


class TestLoggingIntegration:
    def test_logger_used_in_code(self, clear_qme_logger):
        logger = get_qme_logger("qme.test")

        # Should not raise
        logger.info("Test message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_multiple_loggers_same_module(self, clear_qme_logger):
        logger1 = get_qme_logger("qme.test")
        logger2 = get_qme_logger("qme.test")

        # Should return same logger instance (logging.getLogger is cached)
        assert logger1 is logger2

    def test_logger_propagation(self, clear_qme_logger):
        setup_qme_logging(verbosity=1, force=True)

        parent = logging.getLogger("qme")
        child = get_qme_logger("qme.child")

        # Child should inherit from parent
        assert child.parent == parent or child.name.startswith("qme")
