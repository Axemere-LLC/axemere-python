"""Unit tests for axemere.gateway.google._client.genai_client."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from axemere.gateway import AiGatewayConfig, PLACEHOLDER_API_KEY
from axemere.gateway.google._client import genai_client


def _make_config(**overrides: object) -> AiGatewayConfig:
    defaults: dict = {
        "gateway_url": "https://us.gw.axemere.ai",
        "gateway_token": "tok_test",
        "workload_id": "wl_test",
        "project_id": "",
        "account_id": "",
        "customer_id": "",
    }
    defaults.update(overrides)
    return AiGatewayConfig(**defaults)


@pytest.fixture()
def mock_genai() -> MagicMock:
    """Substitute google.genai with a MagicMock for the duration of each test."""
    fake_instance = MagicMock()
    fake_instance._api_client._http_options.headers = {"x-goog-api-key": PLACEHOLDER_API_KEY}

    module = MagicMock()
    module.Client.return_value = fake_instance

    with patch.dict(sys.modules, {"google.genai": module}):
        yield module


def test_returns_genai_client_instance(mock_genai: MagicMock) -> None:
    result = genai_client(config=_make_config())
    assert result is mock_genai.Client.return_value


def test_default_base_url_is_gateway_url(mock_genai: MagicMock) -> None:
    genai_client(config=_make_config(gateway_url="https://us.gw.axemere.ai"))
    _, kwargs = mock_genai.Client.call_args
    assert kwargs["http_options"]["base_url"] == "https://us.gw.axemere.ai"


def test_caller_can_override_base_url(mock_genai: MagicMock) -> None:
    genai_client(config=_make_config(), http_options={"base_url": "https://custom.endpoint"})
    _, kwargs = mock_genai.Client.call_args
    assert kwargs["http_options"]["base_url"] == "https://custom.endpoint"


def test_sets_authorization_header(mock_genai: MagicMock) -> None:
    genai_client(config=_make_config(gateway_token="tok_test"))
    _, kwargs = mock_genai.Client.call_args
    assert kwargs["http_options"]["headers"].get("Authorization") == "Bearer tok_test"


def test_sets_target_host_header(mock_genai: MagicMock) -> None:
    genai_client(config=_make_config())
    _, kwargs = mock_genai.Client.call_args
    assert (
        kwargs["http_options"]["headers"].get("X-MVGC-Target-Host")
        == "generativelanguage.googleapis.com"
    )


def test_sets_placeholder_api_key(mock_genai: MagicMock) -> None:
    genai_client(config=_make_config())
    _, kwargs = mock_genai.Client.call_args
    assert kwargs.get("api_key") == PLACEHOLDER_API_KEY


def test_strips_xgoog_api_key_after_init(mock_genai: MagicMock) -> None:
    headers = {"x-goog-api-key": PLACEHOLDER_API_KEY, "Authorization": "Bearer tok_test"}
    mock_genai.Client.return_value._api_client._http_options.headers = headers

    genai_client(config=_make_config())

    assert "x-goog-api-key" not in headers
    assert "Authorization" in headers  # other headers preserved


def test_warns_when_xgoog_strip_fails(mock_genai: MagicMock) -> None:
    # Simulate the SDK changing its internal _api_client structure
    mock_genai.Client.return_value._api_client = MagicMock(spec=[])

    with pytest.warns(RuntimeWarning, match="Could not strip x-goog-api-key"):
        genai_client(config=_make_config())


def test_caller_http_options_merged(mock_genai: MagicMock) -> None:
    genai_client(config=_make_config(), http_options={"timeout": 30})
    _, kwargs = mock_genai.Client.call_args
    opts = kwargs["http_options"]
    assert opts["timeout"] == 30
    assert "base_url" in opts
    assert "headers" in opts


def test_caller_headers_win_over_gateway_headers(mock_genai: MagicMock) -> None:
    genai_client(config=_make_config(), http_options={"headers": {"X-Custom": "yes"}})
    _, kwargs = mock_genai.Client.call_args
    hdrs = kwargs["http_options"]["headers"]
    assert hdrs.get("X-Custom") == "yes"
    assert "Authorization" in hdrs  # gateway headers still present


def test_reads_env_vars_when_no_config(mock_genai: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXEMERE_GATEWAY_URL", "https://env.gw.axemere.ai")
    monkeypatch.setenv("AXEMERE_GATEWAY_TOKEN", "env_tok")

    genai_client()

    _, kwargs = mock_genai.Client.call_args
    assert kwargs["http_options"]["base_url"] == "https://env.gw.axemere.ai"
    assert kwargs["http_options"]["headers"].get("Authorization") == "Bearer env_tok"
