"""Thin Anthropic client subclasses that route through the Axemere AI Gateway."""

from __future__ import annotations

import anthropic as _anthropic

from axemere.gateway import AiGatewayConfig, PLACEHOLDER_API_KEY, ai_gateway_headers


class Anthropic(_anthropic.Anthropic):
    """Drop-in replacement for ``anthropic.Anthropic`` that routes through the Axemere AI Gateway."""

    def __init__(self, *, config: AiGatewayConfig | None = None, **kwargs) -> None:
        cfg = config or AiGatewayConfig.from_env()
        caller_headers = kwargs.pop("default_headers", None) or {}
        merged_headers = {**ai_gateway_headers(cfg, target_host="api.anthropic.com"), **caller_headers}
        kwargs.setdefault("base_url", cfg.gateway_url)
        kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
        super().__init__(default_headers=merged_headers, **kwargs)


class AsyncAnthropic(_anthropic.AsyncAnthropic):
    """Drop-in replacement for ``anthropic.AsyncAnthropic`` that routes through the Axemere AI Gateway."""

    def __init__(self, *, config: AiGatewayConfig | None = None, **kwargs) -> None:
        cfg = config or AiGatewayConfig.from_env()
        caller_headers = kwargs.pop("default_headers", None) or {}
        merged_headers = {**ai_gateway_headers(cfg, target_host="api.anthropic.com"), **caller_headers}
        kwargs.setdefault("base_url", cfg.gateway_url)
        kwargs.setdefault("api_key", PLACEHOLDER_API_KEY)
        super().__init__(default_headers=merged_headers, **kwargs)
