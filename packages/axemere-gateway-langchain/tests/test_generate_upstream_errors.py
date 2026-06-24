"""Tests for _generate() upstream provider error surfacing."""

import json
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from axemere.gateway.langchain._client import ChatAiGateway
from axemere.gateway._errors import GatewayError


def _make_gateway_response(status_code: int, body: Any) -> MagicMock:
    """Build a mock httpx.Response wrapping an upstream connector result."""
    payload = {
        "schema": "v1",
        "request_id": "req_test",
        "record_id": "rec_test",
        "decision": "allow",
        "result": {
            "status_code": status_code,
            "body": body,
        },
    }
    resp = MagicMock()
    resp.status_code = 200  # gateway always returns 200
    resp.json.return_value = payload
    return resp


def _make_chat_ai_gateway(provider: str = "openai") -> ChatAiGateway:
    cfg = MagicMock()
    cfg.gateway_url = "http://localhost:7080"
    cfg.gateway_token = None
    cfg.workload_id = "wl_test"
    cfg.labels = {}
    cfg.timeout = 120
    llm = ChatAiGateway(provider=provider, model="test-model", config=cfg)
    return llm


MESSAGES = [HumanMessage(content="hello")]


class TestUpstreamErrorSurfacing:
    def test_openai_4xx_raises_with_message(self):
        llm = _make_chat_ai_gateway("openai")
        body = {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}}
        resp = _make_gateway_response(401, body)
        with patch("httpx.Client.post", return_value=resp):
            with pytest.raises(GatewayError) as exc_info:
                llm._generate(MESSAGES)
        assert "openai" in str(exc_info.value)
        assert "401" in str(exc_info.value)
        assert "Invalid API key" in str(exc_info.value)

    def test_anthropic_4xx_raises_with_message(self):
        llm = _make_chat_ai_gateway("anthropic")
        body = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        resp = _make_gateway_response(401, body)
        with patch("httpx.Client.post", return_value=resp):
            with pytest.raises(GatewayError) as exc_info:
                llm._generate(MESSAGES)
        assert "anthropic" in str(exc_info.value)
        assert "401" in str(exc_info.value)
        assert "invalid x-api-key" in str(exc_info.value)

    def test_gemini_503_raises_with_message(self):
        llm = _make_chat_ai_gateway("google")
        body = {"error": {"code": 503, "message": "The model is overloaded. Please try again later.", "status": "UNAVAILABLE"}}
        resp = _make_gateway_response(503, body)
        with patch("httpx.Client.post", return_value=resp):
            with pytest.raises(GatewayError) as exc_info:
                llm._generate(MESSAGES)
        assert "google" in str(exc_info.value)
        assert "503" in str(exc_info.value)
        assert "overloaded" in str(exc_info.value)

    def test_429_rate_limit_raises(self):
        llm = _make_chat_ai_gateway("openai")
        body = {"error": {"message": "Rate limit exceeded", "type": "requests", "code": "rate_limit_exceeded"}}
        resp = _make_gateway_response(429, body)
        with patch("httpx.Client.post", return_value=resp):
            with pytest.raises(GatewayError) as exc_info:
                llm._generate(MESSAGES)
        assert "429" in str(exc_info.value)
        assert "Rate limit exceeded" in str(exc_info.value)

    def test_non_dict_error_body_raises_without_crash(self):
        """A plain string error body should not cause an AttributeError."""
        llm = _make_chat_ai_gateway("openai")
        resp = _make_gateway_response(500, "internal server error")
        with patch("httpx.Client.post", return_value=resp):
            with pytest.raises(GatewayError) as exc_info:
                llm._generate(MESSAGES)
        assert "500" in str(exc_info.value)

    def test_200_upstream_does_not_raise(self):
        """A successful upstream response still parses normally."""
        llm = _make_chat_ai_gateway("openai")
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "gpt-4o",
        }
        payload = {
            "schema": "v1",
            "request_id": "req_test",
            "record_id": "rec_test",
            "decision": "allow",
            "metering": {"cost_usd": "0.000010", "tokens_in": 5, "tokens_out": 3},
            "result": {"status_code": 200, "body": body},
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = payload
        with patch("httpx.Client.post", return_value=resp):
            result = llm._generate(MESSAGES)
        assert result.generations[0].message.content == "hello"
