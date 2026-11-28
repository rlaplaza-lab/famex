"""Tests for shared HTTP helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from famex.io.http import USER_AGENT, create_session, get_with_retries


def test_create_session_sets_user_agent() -> None:
    session = create_session()
    try:
        assert session.headers["User-Agent"] == USER_AGENT
    finally:
        session.close()


def test_get_with_retries_retries_transient_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("famex.io.http.time.sleep", lambda _delay: None)
    responses = [
        MagicMock(status_code=503),
        MagicMock(status_code=200, content=b"ok"),
    ]
    session = MagicMock()
    session.get.side_effect = responses

    result = get_with_retries("http://example.com/x", session=session)
    assert result.status_code == 200
    assert session.get.call_count == 2


def test_get_with_retries_raises_after_transport_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("famex.io.http.time.sleep", lambda _delay: None)
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError):
        get_with_retries("http://example.com/x", session=session, retry_delays=(0.0, 0.0))
