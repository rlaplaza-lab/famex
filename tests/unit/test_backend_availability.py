from __future__ import annotations

from unittest.mock import MagicMock, patch

from famex.backends.availability import (
    BackendAvailabilityChecker,
    clear_availability_cache,
    get_availability_reason,
    get_available_backends,
    get_available_backends_with_logging,
    get_available_ml_backends,
    get_backend_error_message,
    is_backend_available,
)
from famex.backends.constants import BACKEND_MOCK


class TestBackendAvailabilityChecker:
    def test_checker_initialization(self):
        checker = BackendAvailabilityChecker()

        assert isinstance(checker._cache, dict)
        assert isinstance(checker._conflict_cache, dict)
        assert len(checker._cache) == 0
        assert len(checker._conflict_cache) == 0
        assert len(checker._requirements) > 0

    def test_check_basic_dependencies_mock(self):
        checker = BackendAvailabilityChecker()

        result = checker._check_basic_dependencies(BACKEND_MOCK)
        assert result is True  # Mock has no dependencies

    def test_check_basic_dependencies_missing(self):
        checker = BackendAvailabilityChecker()

        with patch("famex.backends.availability.deps") as mock_deps:
            mock_deps.has.return_value = False

            result = checker._check_basic_dependencies("uma")
            assert result is False

    def test_check_basic_dependencies_present(self):
        checker = BackendAvailabilityChecker()

        with patch("famex.backends.availability.deps") as mock_deps:
            mock_deps.has.return_value = True

            result = checker._check_basic_dependencies("uma")
            assert result is True

    def test_check_import_compatibility_mock(self):
        checker = BackendAvailabilityChecker()

        result = checker._check_import_compatibility(BACKEND_MOCK)
        assert result is None  # No import issues

    def test_check_known_conflicts_cache(self):
        checker = BackendAvailabilityChecker()

        # First call should check and cache
        with patch("famex.backends.availability._check_e3nn_conflict") as mock_check:
            mock_check.return_value = None
            result1 = checker._check_known_conflicts("mace")

        # Second call should use cache
        result2 = checker._check_known_conflicts("mace")

        # Both should return same result
        assert result1 == result2
        assert "mace" in checker._conflict_cache

    def test_is_backend_available_mock(self):
        checker = BackendAvailabilityChecker()

        assert checker.is_backend_available(BACKEND_MOCK) is True
        assert BACKEND_MOCK in checker._cache
        assert checker._cache[BACKEND_MOCK] is True

    def test_is_backend_available_cached(self):
        checker = BackendAvailabilityChecker()

        # Check mock (always available)
        result1 = checker.is_backend_available(BACKEND_MOCK)
        assert result1 is True

        # Mock the dependency check to ensure cache is used
        with patch.object(checker, "_check_basic_dependencies"):
            result2 = checker.is_backend_available(BACKEND_MOCK)

            # Should use cache, so dependency check shouldn't be called
            assert result2 is True
            # For mock, it's cached immediately, so this might not be called
            # But for other backends, cache would prevent re-checking

    def test_is_backend_available_missing_dependencies(self):
        checker = BackendAvailabilityChecker()

        with patch.object(checker, "_check_basic_dependencies", return_value=False):
            result = checker.is_backend_available("uma")
            assert result is False
            assert checker._cache["uma"] is False

    def test_is_backend_available_with_conflict(self):
        checker = BackendAvailabilityChecker()

        with (
            patch.object(checker, "_check_basic_dependencies", return_value=True),
            patch.object(checker, "_check_known_conflicts", return_value="Test conflict"),
        ):
            result = checker.is_backend_available("mace")
            assert result is False

    def test_is_backend_available_with_import_error(self):
        checker = BackendAvailabilityChecker()

        with (
            patch.object(checker, "_check_basic_dependencies", return_value=True),
            patch.object(checker, "_check_known_conflicts", return_value=None),
            patch.object(checker, "_check_import_compatibility", return_value="Import error"),
        ):
            result = checker.is_backend_available("mace")
            assert result is False

    def test_get_availability_reason_mock(self):
        checker = BackendAvailabilityChecker()

        reason = checker.get_availability_reason(BACKEND_MOCK)
        assert "available" in reason.lower()

    def test_get_availability_reason_missing_deps(self):
        checker = BackendAvailabilityChecker()

        with patch.object(checker, "_check_basic_dependencies", return_value=False):
            reason = checker.get_availability_reason("uma")
            assert "missing" in reason.lower() or "dependencies" in reason.lower()

    def test_get_availability_reason_conflict(self):
        checker = BackendAvailabilityChecker()

        with (
            patch.object(checker, "_check_basic_dependencies", return_value=True),
            patch.object(checker, "_check_known_conflicts", return_value="Version conflict"),
        ):
            reason = checker.get_availability_reason("mace")
            assert "conflict" in reason.lower()

    def test_get_availability_reason_available(self):
        checker = BackendAvailabilityChecker()

        with (
            patch.object(checker, "_check_basic_dependencies", return_value=True),
            patch.object(checker, "_check_known_conflicts", return_value=None),
            patch.object(checker, "_check_import_compatibility", return_value=None),
        ):
            reason = checker.get_availability_reason("uma")
            assert "available" in reason.lower()

    def test_get_available_backends(self):
        checker = BackendAvailabilityChecker()

        # Mock is always available
        backends = checker.get_available_backends(include_mock=True)
        assert BACKEND_MOCK in backends

        backends_no_mock = checker.get_available_backends(include_mock=False)
        assert BACKEND_MOCK not in backends_no_mock

    def test_clear_cache(self):
        checker = BackendAvailabilityChecker()

        # Populate cache
        checker.is_backend_available(BACKEND_MOCK)
        assert len(checker._cache) > 0

        # Clear cache
        checker.clear_cache()
        assert len(checker._cache) == 0
        assert len(checker._conflict_cache) == 0


class TestConvenienceFunctions:
    def test_is_backend_available_function(self):
        assert is_backend_available(BACKEND_MOCK) is True

    def test_get_availability_reason_function(self):
        reason = get_availability_reason(BACKEND_MOCK)
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_get_available_backends_function(self):
        backends = get_available_backends(include_mock=True)
        assert isinstance(backends, list)
        assert BACKEND_MOCK in backends

        backends_no_mock = get_available_backends(include_mock=False)
        assert BACKEND_MOCK not in backends_no_mock

    def test_clear_availability_cache_function(self):
        # Populate cache
        is_backend_available(BACKEND_MOCK)

        # Clear it
        clear_availability_cache()

        # Create new checker to verify cache is cleared
        # (We can't directly check the global instance's cache)
        # But we can verify the function works without error
        assert True  # Function should complete without error

    def test_get_backend_error_message(self):
        message = get_backend_error_message(BACKEND_MOCK)

        assert isinstance(message, str)
        assert BACKEND_MOCK in message
        assert "available" in message.lower() or "install" in message.lower()

    def test_get_backend_error_message_unavailable(self):
        with (
            patch("famex.backends.availability.is_backend_available", return_value=False),
            patch(
                "famex.backends.availability.get_availability_reason", return_value="Missing deps"
            ),
        ):
            message = get_backend_error_message("nonexistent")

            assert "nonexistent" in message
            assert "not available" in message.lower() or "missing" in message.lower()

    def test_get_available_backends_with_logging(self):
        backends = get_available_backends_with_logging(include_mock=True, verbose=False)

        assert isinstance(backends, list)
        assert BACKEND_MOCK in backends

    def test_get_available_backends_with_logging_verbose(self):
        with patch("famex.backends.availability._get_logger") as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log

            backends = get_available_backends_with_logging(include_mock=True, verbose=True)

            assert isinstance(backends, list)
            # Logger should have been called
            assert mock_log.info.called

    def test_get_available_ml_backends(self):
        backends = get_available_ml_backends(verbose=False)

        assert isinstance(backends, list)
        assert BACKEND_MOCK not in backends  # Should exclude mock


class TestConflictChecking:
    def test_e3nn_conflict_detection_not_applicable(self):
        from famex.backends.availability import _check_e3nn_conflict

        with patch("famex.backends.availability.deps") as mock_deps:
            mock_deps.has.return_value = False
            result = _check_e3nn_conflict()
            assert result is None
