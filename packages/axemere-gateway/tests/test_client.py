"""Tests for the core axemere-gateway client, config, and headers."""

from __future__ import annotations

import os
from unittest import mock

import httpx
import pytest

from axemere.gateway import (
    AiGatewayClient,
    AiGatewayConfig,
    GatewayError,
    PolicyDeniedError,
    QuotaExceededError,
    ai_gateway_headers,
)


# ---------------------------------------------------------------------------
# AiGatewayConfig environment resolution
# ---------------------------------------------------------------------------


def test_config_reads_url_and_token_from_env():
    env = {
        "AXEMERE_GATEWAY_URL": "https://gw.example.com",
        "AXEMERE_GATEWAY_TOKEN": "tok-abc",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        cfg = AiGatewayConfig()
    assert cfg.gateway_url == "https://gw.example.com"
    assert cfg.gateway_token == "tok-abc"


def test_config_falls_back_to_workload_token():
    env = {"AXEMERE_WORKLOAD_TOKEN": "legacy-tok"}
    with mock.patch.dict(os.environ, env, clear=True):
        cfg = AiGatewayConfig()
    assert cfg.gateway_token == "legacy-tok"


def test_config_gateway_token_takes_precedence_over_workload_token():
    env = {
        "AXEMERE_GATEWAY_TOKEN": "primary",
        "AXEMERE_WORKLOAD_TOKEN": "legacy",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        cfg = AiGatewayConfig()
    assert cfg.gateway_token == "primary"


def test_config_defaults_when_env_absent():
    with mock.patch.dict(os.environ, {}, clear=True):
        cfg = AiGatewayConfig()
    assert cfg.gateway_url == "http://localhost:7080"
    assert cfg.gateway_token is None
    assert cfg.timeout == 120.0


def test_from_env_matches_bare_constructor():
    env = {"AXEMERE_GATEWAY_URL": "https://gw", "AXEMERE_GATEWAY_TOKEN": "t"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert AiGatewayConfig.from_env() == AiGatewayConfig()


# ---------------------------------------------------------------------------
# Header building
# ---------------------------------------------------------------------------


def test_headers_include_auth_when_token_set():
    cfg = AiGatewayConfig(gateway_url="http://gw", gateway_token="tok-xyz")
    headers = ai_gateway_headers(cfg, target_host="api.openai.com")
    assert headers["Authorization"] == "Bearer tok-xyz"
    assert headers["X-MVGC-Target-Host"] == "api.openai.com"


def test_headers_omit_auth_when_token_empty():
    cfg = AiGatewayConfig(gateway_url="http://gw", gateway_token=None)
    headers = ai_gateway_headers(cfg)
    assert "Authorization" not in headers


def test_client_build_headers_includes_and_omits_auth():
    with_token = AiGatewayClient(AiGatewayConfig(gateway_token="tok-1"))
    assert with_token._build_headers()["Authorization"] == "Bearer tok-1"

    without_token = AiGatewayClient(AiGatewayConfig(gateway_token=None))
    assert "Authorization" not in without_token._build_headers()


# ---------------------------------------------------------------------------
# execute_sync error handling
# ---------------------------------------------------------------------------


def _response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "http://gw/v1/actions:execute"),
    )


def _patch_async_client(response: httpx.Response):
    """Patch httpx.AsyncClient so execute_sync returns *response* without I/O."""

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return response

    return mock.patch("axemere.gateway._client.httpx.AsyncClient", _FakeAsyncClient)


def _call_execute_sync(client: AiGatewayClient) -> None:
    client.execute_sync(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )


def test_execute_sync_raises_gateway_error_when_url_empty():
    client = AiGatewayClient(AiGatewayConfig(gateway_url="", gateway_token="t"))
    with pytest.raises(GatewayError):
        _call_execute_sync(client)


def test_execute_sync_4xx_raises_gateway_error():
    client = AiGatewayClient(AiGatewayConfig(gateway_url="http://gw", gateway_token="t"))
    resp = _response(400, {"result": {"status_code": 400, "body": {"error": "bad request"}}})
    with _patch_async_client(resp):
        with pytest.raises(GatewayError) as exc_info:
            _call_execute_sync(client)
    assert exc_info.value.status_code == 400


def test_execute_sync_403_raises_policy_denied():
    client = AiGatewayClient(AiGatewayConfig(gateway_url="http://gw", gateway_token="t"))
    resp = _response(403, {"decision": "deny", "decision_trace": {"reason": "blocked by policy"}})
    with _patch_async_client(resp):
        with pytest.raises(PolicyDeniedError) as exc_info:
            _call_execute_sync(client)
    assert exc_info.value.reason == "blocked by policy"


def test_execute_sync_429_raises_quota_exceeded():
    client = AiGatewayClient(AiGatewayConfig(gateway_url="http://gw", gateway_token="t"))
    resp = _response(429, {"decision": "allow", "message": "over budget", "upgrade_url": "https://up"})
    with _patch_async_client(resp):
        with pytest.raises(QuotaExceededError) as exc_info:
            _call_execute_sync(client)
    assert exc_info.value.upgrade_url == "https://up"


def test_execute_sync_success_returns_content():
    client = AiGatewayClient(AiGatewayConfig(gateway_url="http://gw", gateway_token="t"))
    payload = {
        "decision": "allow",
        "record_id": "rec_1",
        "result": {
            "status_code": 200,
            "body": {
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "hello there"}}],
            },
        },
    }
    resp = _response(200, payload)
    with _patch_async_client(resp):
        result = client.execute_sync(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert result.content == "hello there"
    assert result.record_id == "rec_1"


# ---------------------------------------------------------------------------
# Provider host routing
# ---------------------------------------------------------------------------


def test_provider_hosts_correct_for_non_standard_providers():
    """Providers that don't follow the api.{name}.com pattern must have explicit entries."""
    from axemere.gateway._client import _PROVIDER_HOSTS

    cases = {
        "nvidia-nim": "integrate.api.nvidia.com",
        "openrouter": "openrouter.ai",
        "upstage": "api.upstage.ai",
        "moonshot": "api.moonshot.ai",
        "minimax": "api.minimax.chat",
        "zhipu": "api.z.ai",
    }
    for provider, expected_host in cases.items():
        assert _PROVIDER_HOSTS.get(provider) == expected_host, (
            f"provider {provider!r}: expected host {expected_host!r}, "
            f"got {_PROVIDER_HOSTS.get(provider)!r}"
        )
